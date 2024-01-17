mod state;

use super::{
    kernel::{gemm, rmsnorm, rmsnorm_inplace, sigmoid, slice, softmax},
    tokenizer::utok,
};
use crate::arguments::{AllInOneBin, Arguments, SafeTensors};
use state::{Layer, RotaryEmbedder, RunState};
use std::{ffi::OsStr, fs::File, iter::zip, path::Path};

/// `upos` for position id.
#[allow(non_camel_case_types)]
pub(super) type upos = u32;

pub(super) struct Transformer {
    layers: Vec<Layer>,
    logits: Vec<f32>,
    embedder: RotaryEmbedder,
    arguments: Box<dyn Arguments>,
}

impl Transformer {
    pub fn read_checkpoint(checkpoint: impl AsRef<Path>) -> Self {
        let checkpoint = checkpoint.as_ref();
        let arguments: Box<dyn Arguments> = if checkpoint.extension()
            == Some(OsStr::new("safetensors"))
        {
            let config = checkpoint.parent().unwrap().join("config.json");
            let config = File::open(&config)
                .unwrap_or_else(|_| panic!("Could not open config {}", config.display()));
            let tensors = File::open(checkpoint)
                .unwrap_or_else(|_| panic!("Could not open safetensors {}", checkpoint.display()));
            Box::new(SafeTensors::new(config, tensors))
        } else {
            let file = File::open(checkpoint)
                .unwrap_or_else(|_| panic!("Could not open checkpoint {}", checkpoint.display()));
            Box::new(AllInOneBin::new(file))
        };

        Self {
            layers: vec![Layer::new(&*arguments); arguments.n_layers()],
            logits: vec![0.; arguments.vocab_size()],
            embedder: RotaryEmbedder::new(&*arguments),
            arguments,
        }
    }

    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.arguments.vocab_size()
    }

    pub fn update(&mut self, tokens: &[utok], pos: upos) -> Vec<f32> {
        let tok_len = tokens.len();
        let pos = pos as usize;

        let dim = self.arguments.dim();
        let hidden_dim = self.arguments.hidden_dim();
        let seq_len = self.arguments.seq_len();
        let kv_dim = self.arguments.kv_dim();

        let n_head = self.arguments.n_heads();
        let kv_mul = n_head / self.arguments.n_kv_heads();
        let head_size = dim / n_head;
        let head_div = 1. / (head_size as f32).sqrt();

        let mut s = RunState::new(tok_len, dim, hidden_dim, n_head, seq_len);
        let h = s.hidden.split_at_mut(tok_len * hidden_dim);

        for (i, &token) in tokens.iter().enumerate() {
            slice!(s.x0; dim; [i]).copy_from_slice(self.arguments.token_embedding_table(token));
        }

        for (l, layer) in self.layers.iter_mut().enumerate() {
            let Layer { k_cache, v_cache } = layer;

            // x1 = rmsnorm(x0, rms_att_weight[l]);
            rmsnorm(&mut s.x1, &s.x0, self.arguments.rms_att_weight(l));
            {
                let k = dim;
                let n = tok_len;
                let b = s.x1.as_ptr();
                let rsb = 1;
                let csb = k as _;
                let alpha = 1.;
                let beta = 0.;
                // q = wq[l] * x1;
                let m = dim;
                let a = self.arguments.wq(l).as_ptr();
                let rsa = k as _;
                let csa = 1;
                let c = s.q.as_mut_ptr();
                let rsc = 1;
                let csc = m as _;
                unsafe { gemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc) };
                // k = wk[l] * x1;
                let m = kv_dim;
                let a = self.arguments.wk(l).as_ptr();
                let rsa = k as _;
                let csa = 1;
                let c = slice!(k_cache; kv_dim; [pos]).as_mut_ptr();
                let rsc = 1;
                let csc = m as _;
                unsafe { gemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc) };
                // v = wv[l] * x1;
                let m = kv_dim;
                let a = self.arguments.wv(l).as_ptr();
                let rsa = k as _;
                let csa = 1;
                let c = slice!(v_cache; kv_dim; [pos]).as_mut_ptr();
                let rsc = 1;
                let csc = m as _;
                unsafe { gemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc) };
            }
            // rotary embeddings
            for i in 0..tok_len {
                let pos = pos + i;
                self.embedder.run(pos, &mut slice!(s.q    ; dim   ; [i]  ));
                self.embedder.run(pos, &mut slice!(k_cache; kv_dim; [pos]));
            }
            // clear x1 for multi-head attention.
            s.x1.fill(0.);
            for h in 0..n_head {
                let att = &mut slice!(s.attention; tok_len * seq_len; [h]);
                let att_len = pos + tok_len;
                // att = head_div * q * k;
                let m = tok_len;
                let k = head_size;
                let n = att_len;
                let alpha = head_div;
                let beta = 0.;
                let a = slice!(s.q; head_size; [h]).as_ptr();
                let b = slice!(k_cache; head_size; [h / kv_mul]).as_ptr();
                let c = att.as_mut_ptr();
                let rsa = kv_dim as _;
                let csa = 1;
                let rsb = 1;
                let csb = kv_dim as _;
                let rsc = seq_len as _;
                let csc = 1;
                unsafe { gemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc) };
                // att = softmax(att);
                for i in 0..tok_len {
                    let att = &mut slice!(att; seq_len; [i])[..att_len];
                    let (att, tail) = att.split_at_mut(pos + i + 1);
                    softmax(att);
                    tail.fill(0.);
                }
                // x1 += att * v;
                let m = head_size;
                let k = att_len;
                let n = tok_len;
                let alpha = 1.;
                let beta = 1.;
                let a = slice!(v_cache; head_size; [h / kv_mul]).as_ptr();
                let b = att.as_ptr();
                let c = slice!(s.x1; head_size; [h]).as_mut_ptr();
                let rsa = 1;
                let csa = kv_dim as _;
                let rsb = 1;
                let csb = seq_len as _;
                let rsc = 1;
                let csc = kv_dim as _;
                unsafe { gemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc) };
            }

            // x0 += wo[l] * x1;
            {
                let m = dim;
                let k = dim;
                let n = tok_len;
                let alpha = 1.;
                let beta = 1.;
                let a = self.arguments.wo(l).as_ptr();
                let b = s.x1.as_ptr();
                let c = s.x0.as_mut_ptr();
                let rsa = k as _;
                let csa = 1;
                let rsb = 1;
                let csb = k as _;
                let rsc = 1;
                let csc = m as _;
                unsafe { gemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc) };
            }
            // x1 = rmsnorm(x0, rms_ffn_weight[l]);
            rmsnorm(&mut s.x1, &s.x0, self.arguments.rms_ffn_weight(l));
            {
                let m = hidden_dim;
                let k = dim;
                let n = tok_len;
                let alpha = 1.;
                let beta = 0.;
                let b = s.x1.as_ptr();
                let rsa = k as _;
                let csa = 1;
                let rsb = 1;
                let csb = k as _;
                let rsc = 1;
                let csc = m as _;
                // h0 = w1[l] * x1;
                let a = self.arguments.w1(l).as_ptr();
                let c = h.0.as_mut_ptr();
                unsafe { gemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc) };
                // h1 = w3[l] * x1;
                let a = self.arguments.w3(l).as_ptr();
                let c = h.1.as_mut_ptr();
                unsafe { gemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc) };
            }
            // h0 *= sigmoid(h0) * h1;
            zip(&mut *h.0, &*h.1).for_each(|(h0, h1)| *h0 *= sigmoid(*h0) * *h1);
            // x0 += w2[l] * h0;
            {
                let m = dim;
                let k = hidden_dim;
                let n = tok_len;
                let alpha = 1.;
                let beta = 1.;
                let a = self.arguments.w2(l).as_ptr();
                let b = h.0.as_ptr();
                let c = s.x0.as_mut_ptr();
                let rsa = k as _;
                let csa = 1;
                let rsb = 1;
                let csb = k as _;
                let rsc = 1;
                let csc = m as _;
                unsafe { gemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc) };
            }
        }

        s.x0
    }

    pub fn forward(&mut self, token: utok, pos: upos) -> &mut [f32] {
        let mut x = self.update(&[token], pos);

        rmsnorm_inplace(&mut x, self.arguments.rms_final_weight());
        // logits = wcls * x;
        {
            let m = self.logits.len();
            let k = x.len();
            let n = 1;
            let alpha = 1.;
            let beta = 0.;
            let a = self.arguments.wcls().as_ptr();
            let b = x.as_ptr();
            let c = self.logits.as_mut_ptr();
            let rsa = k as _;
            let csa = 1;
            let rsb = 1;
            let csb = k as _;
            let rsc = 1;
            let csc = m as _;
            unsafe { gemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc) };
        }

        &mut self.logits
    }
}

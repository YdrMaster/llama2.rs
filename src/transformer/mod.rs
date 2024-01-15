mod config;
mod state;
mod weights;

use super::{
    kernel::{gemm, rmsnorm, rmsnorm_inplace, sigmoid, slice, softmax},
    tokenizer::utok,
};
use config::Config;
use memmap2::Mmap;
use state::{Layer, RotaryEmbedder, RunState};
use std::{fs::File, iter::zip, path::Path};
use weights::Weights;

/// `upos` for position id.
#[allow(non_camel_case_types)]
pub(super) type upos = u32;

pub(super) struct Transformer {
    layers: Vec<Layer>,
    logits: Vec<f32>,
    embedder: RotaryEmbedder,
    mmap: Mmap,
}

impl Transformer {
    pub fn read_checkpoint(checkpoint: impl AsRef<Path>) -> Self {
        let checkpoint = checkpoint.as_ref();
        let file = File::open(checkpoint)
            .unwrap_or_else(|_| panic!("Could not open checkpoint {}", checkpoint.display()));

        let mmap = unsafe { Mmap::map(&file) }.unwrap();
        let config = Config::map(&mmap).0;

        Self {
            layers: vec![Layer::new(config); config.n_layers()],
            logits: vec![0.; config.vocab_size()],
            embedder: RotaryEmbedder::new(config),
            mmap,
        }
    }

    #[inline]
    pub fn vocab_size(&self) -> usize {
        Config::map(&self.mmap).0.vocab_size()
    }

    pub fn update(&mut self, tokens: &[utok], pos: upos) -> Vec<f32> {
        let (config, _) = Config::map(&self.mmap);
        let tok_len = tokens.len();
        let pos = pos as usize;

        let dim = config.dim();
        let hidden_dim = config.hidden_dim();
        let seq_len = config.seq_len();
        let kv_dim = config.kv_dim();

        let n_head = config.n_heads();
        let kv_mul = n_head / config.n_kv_heads();
        let head_size = dim / n_head;
        let head_div = 1. / (head_size as f32).sqrt();

        let w = Weights::new(&self.mmap);
        let mut s = RunState::new(tok_len, config);
        let h = s.hidden.split_at_mut(tok_len * hidden_dim);

        for (i, token) in tokens.iter().map(|&t| t as usize).enumerate() {
            slice!(s.x0; dim; [i]).copy_from_slice(&slice!(w.token_embedding_table; dim; [token]));
        }

        for (l, layer) in self.layers.iter_mut().enumerate() {
            let Layer { k_cache, v_cache } = layer;

            rmsnorm(&mut s.x1, &s.x0, &slice!(w.rms_att_weight; dim; [l]));

            // matmul(q, &s.x1, &slice!(w.wq; dim * dim; [l]));
            {
                let m = dim;
                let k = dim;
                let n = tok_len;
                let alpha = 1.;
                let beta = 0.;
                let a = slice!(w.wq; dim * dim; [l]).as_ptr();
                let b = s.x1.as_ptr();
                let c = s.q.as_mut_ptr();
                let rsa = k as _;
                let csa = 1;
                let rsb = 1;
                let csb = k as _;
                let rsc = 1;
                let csc = m as _;
                unsafe { gemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc) };
            }
            // matmul(k, &s.x1, &slice!(w.wk; kv_dim * dim; [l]));
            {
                let m = kv_dim;
                let k = dim;
                let n = tok_len;
                let alpha = 1.;
                let beta = 0.;
                let a = slice!(w.wk; kv_dim * dim; [l]).as_ptr();
                let b = s.x1.as_ptr();
                let c = slice!(k_cache; kv_dim; [pos]).as_mut_ptr();
                let rsa = k as _;
                let csa = 1;
                let rsb = 1;
                let csb = k as _;
                let rsc = 1;
                let csc = m as _;
                unsafe { gemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc) };
            }
            // matmul(v, &s.x1, &slice!(w.wv; kv_dim * dim; [l]));
            {
                let m = kv_dim;
                let k = dim;
                let n = tok_len;
                let alpha = 1.;
                let beta = 0.;
                let a = slice!(w.wv; kv_dim * dim; [l]).as_ptr();
                let b = s.x1.as_ptr();
                let c = slice!(v_cache; kv_dim; [pos]).as_mut_ptr();
                let rsa = k as _;
                let csa = 1;
                let rsb = 1;
                let csb = k as _;
                let rsc = 1;
                let csc = m as _;
                unsafe { gemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc) };
            }
            for i in 0..tok_len {
                let pos = pos + i;
                self.embedder.run(pos, &mut slice!(s.q; dim; [i]));
                self.embedder.run(pos, &mut slice!(k_cache; kv_dim; [pos]));
            }

            s.x1.fill(0.);
            for h in 0..n_head {
                let att = &mut slice!(s.attention; tok_len * seq_len; [h]);

                {
                    let m = tok_len;
                    let k = head_size;
                    let n = pos + tok_len;
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
                }

                for i in 0..tok_len {
                    let att = &mut slice!(att; seq_len; [i]);
                    let (att, tail) = att.split_at_mut(pos + i + 1);
                    softmax(att);
                    tail.fill(0.);
                }

                {
                    let m = head_size;
                    let k = pos + tok_len;
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
            }

            // sgemm(&mut s.x0, &s.x1, &slice!(w.wo; dim * dim; [l]));
            {
                let m = dim;
                let k = dim;
                let n = tok_len;
                let alpha = 1.;
                let beta = 1.;
                let a = slice!(w.wo; dim * dim; [l]).as_ptr();
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

            rmsnorm(&mut s.x1, &s.x0, &slice!(w.rms_ffn_weight; dim; [l]));

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
                // matmul(h.0, &s.x1, &slice!(w.w1; hidden_dim * dim; [l]));
                let a = slice!(w.w1; hidden_dim * dim; [l]).as_ptr();
                let c = h.0.as_mut_ptr();
                unsafe { gemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc) };
                // matmul(h.1, &s.x1, &slice!(w.w3; hidden_dim * dim; [l]));
                let a = slice!(w.w3; hidden_dim * dim; [l]).as_ptr();
                let c = h.1.as_mut_ptr();
                unsafe { gemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc) };
            }

            zip(&mut *h.0, &*h.1).for_each(|(hb0, hb1)| *hb0 *= sigmoid(*hb0) * *hb1);

            // sgemm(&mut s.x0, h.0, &slice!(w.w2; dim * hidden_dim; [l]));
            {
                let m = dim;
                let k = hidden_dim;
                let n = tok_len;
                let alpha = 1.;
                let beta = 1.;
                let a = slice!(w.w2; dim * hidden_dim; [l]).as_ptr();
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
        let w = Weights::new(&self.mmap);

        rmsnorm_inplace(&mut x, w.rms_final_weight);
        // matmul(&mut self.logits, &x, w.wcls);
        {
            let m = self.logits.len();
            let k = x.len();
            let n = 1;
            let alpha = 1.;
            let beta = 0.;
            let a = w.wcls.as_ptr();
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

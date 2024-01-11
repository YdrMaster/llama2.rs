mod config;
mod weights;

use super::{
    kernel::{matmul, rmsnorm, rmsnorm_inplace, sgemm, sigmoid, softmax},
    tokenizer::utok,
};
use config::Config;
use memmap2::Mmap;
use std::{fs::File, iter::zip, path::Path};
use weights::Weights;

/// `upos` for position id.
#[allow(non_camel_case_types)]
pub(super) type upos = u32;

pub(super) struct Transformer {
    state: RunState,
    embedder: RotaryEmbedder,
    mmap: Mmap,
}

macro_rules! slice {
    ($blob:expr; $width:expr; [$line:expr]) => {
        $blob[$line * $width..][..$width]
    };
}

impl Transformer {
    pub fn read_checkpoint(checkpoint: impl AsRef<Path>) -> Self {
        let checkpoint = checkpoint.as_ref();
        let file = File::open(checkpoint)
            .expect(format!("Could not open checkpoint {}", checkpoint.display()).as_str());

        let mmap = unsafe { Mmap::map(&file) }.unwrap();
        let config = Config::map(&mmap).0;

        Self {
            state: RunState::new(config),
            embedder: RotaryEmbedder::new(config.seq_len(), config.dim(), config.n_heads()),
            mmap,
        }
    }

    #[inline]
    pub fn vocab_size(&self) -> usize {
        Config::map(&self.mmap).0.vocab_size()
    }

    pub fn update(&mut self, tokens: &[utok], pos: upos) {
        let (config, _) = Config::map(&self.mmap);
        let w = Weights::new(&self.mmap);
        let s = &mut self.state;

        let dim = config.dim();
        let hidden_dim = config.hidden_dim();
        let n_layer = config.n_layers();
        let seq_len = config.seq_len();
        let kv_dim = config.kv_dim();

        let n_head = config.n_heads();
        let kv_mul = n_head / config.n_kv_heads();
        let head_size = dim / n_head;
        let head_div = (head_size as f32).sqrt();

        for (i, &token) in tokens.iter().enumerate() {
            let token = token as usize;
            let pos = pos as usize + i;

            let content_row = &slice!(w.token_embedding_table; dim; [token]);
            s.x.copy_from_slice(content_row);

            let att = &mut s.att[..=pos];
            let hb = s.hb.split_at_mut(hidden_dim);

            for l in 0..n_layer {
                rmsnorm(&mut s.xb, &s.x, &slice!(w.rms_att_weight; dim; [l]));

                let q = &mut s.q[..];
                let k = &mut slice!(s.  key_cache; kv_dim; [l * seq_len + pos]);
                let v = &mut slice!(s.value_cache; kv_dim; [l * seq_len + pos]);

                matmul(q, &s.xb, &slice!(w.wq; dim * dim   ; [l]));
                matmul(k, &s.xb, &slice!(w.wk; dim * kv_dim; [l]));
                matmul(v, &s.xb, &slice!(w.wv; dim * kv_dim; [l]));

                self.embedder.run(pos, q);
                self.embedder.run(pos, k);

                s.xb.fill(0.);
                for h in 0..n_head {
                    let q = &slice!(q; head_size; [h]);
                    for (t, a) in att.iter_mut().enumerate() {
                        let k = &slice!(s.key_cache; kv_dim; [l * seq_len + t]);
                        let k = &slice!(k; head_size; [h / kv_mul]);
                        // score
                        *a = zip(q, k).map(|(&q, &k)| q * k).sum::<f32>() / head_div;
                    }

                    softmax(att);

                    let xb = &mut slice!(s.xb; head_size; [h]);
                    for (t, &a) in att.iter().enumerate() {
                        let v = &slice!(s.value_cache; kv_dim; [l * seq_len + t]);
                        let v = &slice!(v; head_size; [h / kv_mul]);
                        zip(&mut xb[..], v).for_each(|(xb, &v)| *xb += a * v);
                    }
                }

                sgemm(&mut s.x, &s.xb, &slice!(w.wo; dim * dim; [l]));

                rmsnorm(&mut s.xb, &s.x, &slice!(w.rms_ffn_weight; dim; [l]));

                matmul(hb.0, &s.xb, &slice!(w.w1; dim * hidden_dim; [l]));
                matmul(hb.1, &s.xb, &slice!(w.w3; dim * hidden_dim; [l]));

                zip(&mut hb.0[..], &hb.1[..]).for_each(|(hb0, hb1)| *hb0 *= sigmoid(*hb0) * *hb1);

                sgemm(&mut s.x, hb.0, &slice!(w.w2; dim * hidden_dim; [l]));
            }
        }
    }

    pub fn forward(&mut self, token: utok, pos: upos) -> &mut [f32] {
        self.update(&[token], pos);

        let w = Weights::new(&self.mmap);
        let s = &mut self.state;

        rmsnorm_inplace(&mut s.x, &w.rms_final_weight);
        matmul(&mut s.logits, &s.x, &w.wcls);

        &mut s.logits
    }
}

struct RunState {
    x: Vec<f32>,      // no cache
    xb: Vec<f32>,     // no cache
    hb: Vec<f32>,     // no cache
    q: Vec<f32>,      // no cache
    att: Vec<f32>,    // no cache
    logits: Vec<f32>, // no cache
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
}

impl RunState {
    fn new(config: &Config) -> Self {
        let dim = config.dim();
        let hidden_dim = config.hidden_dim();
        let n_layers = config.n_layers();
        let seq_len = config.seq_len();
        let kv_dim = config.kv_dim();
        Self {
            x: vec![0.; dim],
            xb: vec![0.; dim],
            hb: vec![0.; hidden_dim * 2],
            q: vec![0.; dim],
            key_cache: vec![0.; n_layers * seq_len * kv_dim],
            value_cache: vec![0.; n_layers * seq_len * kv_dim],
            att: vec![0.; seq_len],
            logits: vec![0.; config.vocab_size()],
        }
    }
}

struct RotaryEmbedder {
    dim: usize,
    rotary: Vec<f32>,
}

impl RotaryEmbedder {
    pub fn new(seq_len: usize, dim: usize, n_heads: usize) -> Self {
        let head_size = dim / n_heads;
        let mut rotary = Vec::with_capacity(seq_len * dim);
        for pos in 0..seq_len {
            for i in (0..dim).step_by(2) {
                let freq = 1e4f32.powf(-((i % head_size) as f32 / head_size as f32));
                let (sin, cos) = (pos as f32 * freq).sin_cos();
                rotary.push(cos);
                rotary.push(sin);
            }
        }
        Self { dim, rotary }
    }

    fn run(&self, pos: usize, data: &mut [f32]) {
        let rotary = &slice!(self.rotary; self.dim; [pos]);
        for i in 0..data.len() / 2 {
            let x = &mut slice!(data; 2; [i]);
            let w = &slice!(rotary; 2; [i]);
            x.copy_from_slice(&[
                x[0] * w[0] - x[1] * w[1], //
                x[1] * w[0] + x[0] * w[1],
            ]);
        }
    }
}

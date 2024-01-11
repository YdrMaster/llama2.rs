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
    layers: Vec<Layer>,
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
            layers: vec![Layer::new(config); config.n_layers()],
            embedder: RotaryEmbedder::new(config),
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
            s.x0.copy_from_slice(content_row);

            let h = s.hidden.split_at_mut(hidden_dim);

            for (l, layer) in self.layers.iter_mut().enumerate() {
                let Layer { k_cache, v_cache } = layer;

                rmsnorm(&mut s.x1, &s.x0, &slice!(w.rms_att_weight; dim; [l]));

                {
                    let q = &mut s.q[..];
                    let k = &mut slice!(k_cache; kv_dim; [pos]);
                    let v = &mut slice!(v_cache; kv_dim; [pos]);

                    matmul(q, &s.x1, &slice!(w.wq; dim * dim   ; [l]));
                    matmul(k, &s.x1, &slice!(w.wk; dim * kv_dim; [l]));
                    matmul(v, &s.x1, &slice!(w.wv; dim * kv_dim; [l]));

                    self.embedder.run(pos, q);
                    self.embedder.run(pos, k);
                }

                s.x1.fill(0.);
                for h in 0..n_head {
                    let att = &mut slice!(s.attention; seq_len; [h]);
                    let att = &mut att[..=pos];

                    let q = &slice!(s.q; head_size; [h]);
                    for (t, a) in att.iter_mut().enumerate() {
                        let k = &slice!(k_cache; kv_dim; [t]);
                        let k = &slice!(k; head_size; [h / kv_mul]);
                        // score
                        *a = zip(q, k).map(|(&q, &k)| q * k).sum::<f32>() / head_div;
                    }

                    softmax(att);

                    let xb = &mut slice!(s.x1; head_size; [h]);
                    for (t, &a) in att.iter().enumerate() {
                        let v = &slice!(v_cache; kv_dim; [t]);
                        let v = &slice!(v; head_size; [h / kv_mul]);
                        zip(&mut xb[..], v).for_each(|(xb, &v)| *xb += a * v);
                    }
                }

                sgemm(&mut s.x0, &s.x1, &slice!(w.wo; dim * dim; [l]));

                rmsnorm(&mut s.x1, &s.x0, &slice!(w.rms_ffn_weight; dim; [l]));

                matmul(h.0, &s.x1, &slice!(w.w1; dim * hidden_dim; [l]));
                matmul(h.1, &s.x1, &slice!(w.w3; dim * hidden_dim; [l]));

                zip(&mut h.0[..], &h.1[..]).for_each(|(hb0, hb1)| *hb0 *= sigmoid(*hb0) * *hb1);

                sgemm(&mut s.x0, h.0, &slice!(w.w2; dim * hidden_dim; [l]));
            }
        }
    }

    pub fn forward(&mut self, token: utok, pos: upos) -> &mut [f32] {
        self.update(&[token], pos);

        let w = Weights::new(&self.mmap);
        let s = &mut self.state;

        rmsnorm_inplace(&mut s.x0, &w.rms_final_weight);
        matmul(&mut s.logits, &s.x0, &w.wcls);

        &mut s.logits
    }
}

/// This struct is new for each inference.
struct RunState {
    x0: Vec<f32>,
    x1: Vec<f32>,
    q: Vec<f32>,
    hidden: Vec<f32>,
    attention: Vec<f32>,
    logits: Vec<f32>,
}

impl RunState {
    fn new(config: &Config) -> Self {
        let dim = config.dim();
        Self {
            x0: vec![0.; dim],
            x1: vec![0.; dim],
            q: vec![0.; dim],
            hidden: vec![0.; config.hidden_dim() * 2],
            attention: vec![0.; config.n_heads() * config.seq_len()],
            logits: vec![0.; config.vocab_size()],
        }
    }
}

#[derive(Clone)]
struct Layer {
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
}

impl Layer {
    fn new(config: &Config) -> Self {
        let len = config.seq_len() * config.kv_dim();
        Self {
            k_cache: vec![0.; len],
            v_cache: vec![0.; len],
        }
    }
}

struct RotaryEmbedder {
    dim: usize,
    rotary: Vec<f32>,
}

impl RotaryEmbedder {
    pub fn new(config: &Config) -> Self {
        let dim = config.dim();
        let n_heads = config.n_heads();
        let seq_len = config.seq_len();
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

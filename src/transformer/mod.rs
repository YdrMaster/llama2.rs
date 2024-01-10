mod config;
mod weights;

use super::{
    kernel::{matmul, rmsnorm, rmsnorm_inplace, softmax},
    tokenizer::utok,
};
use config::Config;
use memmap2::Mmap;
use std::{
    fs::File,
    ops::{Add, Div, Neg},
    path::Path,
};
use weights::Weights;

/// `upos` for position id.
#[allow(non_camel_case_types)]
pub(super) type upos = u32;

pub(super) struct Transformer {
    state: RunState,
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
        Self {
            state: RunState::new(Config::map(&mmap).0),
            mmap,
        }
    }

    #[inline]
    pub fn vocab_size(&self) -> usize {
        Config::map(&self.mmap).0.vocab_size()
    }

    pub fn forward(&mut self, token: utok, pos: upos) -> &mut [f32] {
        let (config, _) = Config::map(&self.mmap);
        let w = Weights::new(&self.mmap);
        let s = &mut self.state;

        let token = token as usize;
        let pos = pos as usize;

        let dim = config.dim();
        let hidden_dim = config.hidden_dim();
        let n_layer = config.n_layers();
        let seq_len = config.seq_len();
        let kv_dim = config.kv_dim();
        let head_size = config.head_size();

        let content_row = &slice!(w.token_embedding_table; dim; [token]);
        s.x[..dim].copy_from_slice(content_row);

        for l in 0..n_layer {
            rmsnorm(&mut s.xb, &s.x, &slice!(w.rms_att_weight; dim; [l]));

            let q = &mut s.q[..];
            let k = &mut slice!(s.  key_cache; kv_dim; [l * seq_len + pos]);
            let v = &mut slice!(s.value_cache; kv_dim; [l * seq_len + pos]);

            matmul(q, &s.xb, &slice!(w.wq; dim * dim   ; [l]));
            matmul(k, &s.xb, &slice!(w.wk; dim * kv_dim; [l]));
            matmul(v, &s.xb, &slice!(w.wv; dim * kv_dim; [l]));

            for i in (0..dim).step_by(2) {
                let freq = 1e4f32.powf(-((i % head_size) as f32 / head_size as f32));
                let w = {
                    let (fci, fcr) = (pos as f32 * freq).sin_cos();
                    [
                        fcr, -fci, //
                        fci, fcr,
                    ]
                };

                #[inline]
                fn rot(y: &mut [f32], w: &[f32]) {
                    let x = &[y[0], y[1]];
                    matmul(y, x, w);
                }

                {
                    rot(&mut q[i..][..2], &w);
                }
                if i < kv_dim {
                    rot(&mut k[i..][..2], &w);
                }
            }

            let n_head = config.n_heads();
            let kv_mul = config.n_heads() / config.n_kv_heads();
            let div = (head_size as f32).sqrt();

            for h in 0..n_head {
                let q = &mut slice!(q; head_size; [h]);
                let att = &mut slice!(s.att; seq_len; [h]);
                for (t, att) in att[..pos].iter_mut().enumerate() {
                    let k = &slice!(s.key_cache; kv_dim; [l * seq_len + t]);
                    let k = &slice!(k; head_size; [h / kv_mul]);
                    let score = q.iter().zip(k).map(|(&q, &k)| q * k).sum::<f32>().div(div);
                    *att = score;
                }

                softmax(&mut att[..pos + 1]);

                let xb = &mut slice!(s.xb; head_size; [h]);
                xb.fill(0.);
                for t in 0..=pos {
                    let v = &slice!(s.value_cache; kv_dim; [l * seq_len + t]);
                    let v = &slice!(v; head_size; [h / kv_mul]);
                    let att = att[t];
                    for (xb, &v) in xb.iter_mut().zip(v) {
                        *xb += att * v;
                    }
                }
            }

            matmul(&mut s.xb2, &s.xb, &slice!(w.wo; dim * dim; [l]));

            s.x.iter_mut().zip(&s.xb2).for_each(|(x, &xb2)| *x += xb2);

            rmsnorm(&mut s.xb, &s.x, &slice!(w.rms_ffn_weight; dim; [l]));

            matmul(&mut s.hb, &s.xb, &slice!(w.w1; dim * hidden_dim; [l]));
            matmul(&mut s.hb2, &s.xb, &slice!(w.w3; dim * hidden_dim; [l]));

            for (hb, hb2) in s.hb.iter_mut().zip(&s.hb2) {
                *hb *= hb.neg().exp().add(1.).recip() * hb2;
            }

            matmul(&mut s.xb, &s.hb, &slice!(w.w2; dim * hidden_dim; [l]));

            for (x, xb) in s.x.iter_mut().zip(&s.xb) {
                *x += xb;
            }
        }

        rmsnorm_inplace(&mut s.x, &w.rms_final_weight);

        matmul(&mut s.logits, &s.x, &w.wcls);

        return &mut s.logits;
    }
}

struct RunState {
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    q: Vec<f32>,
    att: Vec<f32>,
    logits: Vec<f32>,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
}

impl RunState {
    fn new(config: &Config) -> Self {
        let dim = config.dim();
        let hidden_dim = config.hidden_dim();
        let n_layers = config.n_layers();
        let n_heads = config.n_heads();
        let seq_len = config.seq_len();
        let kv_dim = config.kv_dim();
        Self {
            x: vec![0.; dim],
            xb: vec![0.; dim],
            xb2: vec![0.; dim],
            hb: vec![0.; hidden_dim],
            hb2: vec![0.; hidden_dim],
            q: vec![0.; dim],
            key_cache: vec![0.; n_layers * seq_len * kv_dim],
            value_cache: vec![0.; n_layers * seq_len * kv_dim],
            att: vec![0.; n_heads * seq_len],
            logits: vec![0.; config.vocab_size()],
        }
    }
}

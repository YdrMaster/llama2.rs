mod config;
mod weights;

use crate::tokenizer::utok;
use config::Config;
use memmap2::Mmap;
use std::{fs::File, path::Path};
use weights::Weights;

/// `upos` for position id.
#[allow(non_camel_case_types)]
pub(super) type upos = u32;

pub(super) struct Transformer {
    state: RunState,
    mmap: Mmap,
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

    pub fn forward(&mut self, token: utok, pos: upos) -> f32 {
        todo!()
    }
}

struct RunState {
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    q: Vec<f32>,
    // k: Vec<f32>,
    // v: Vec<f32>,
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
        let kv_dim = (dim * config.n_kv_heads()) / n_heads;
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

use super::config::Config;
use crate::kernel::slice;

pub(super) struct RunState {
    /// state buffer: `tok_len x dim`.
    pub x0: Vec<f32>,
    /// state buffer: `tok_len x dim`.
    pub x1: Vec<f32>,
    /// query buffer: `tok_len x dim`.
    pub q: Vec<f32>,
    /// hidden state buffer: `2 * tok_len x hidden_dim`.
    ///
    /// split to two buffers for using.
    pub hidden: Vec<f32>,
    /// attention buffer: `n_heads x tok_len x seq_len`.
    pub attention: Vec<f32>,
}

impl RunState {
    pub fn new(tok_len: usize, config: &Config) -> Self {
        let dim = config.dim();
        Self {
            x0: vec![0.; tok_len * dim],
            x1: vec![0.; tok_len * dim],
            q: vec![0.; tok_len * dim],
            hidden: vec![0.; tok_len * config.hidden_dim() * 2],
            attention: vec![0.; tok_len * config.n_heads() * config.seq_len()],
        }
    }
}

#[derive(Clone)]
pub(super) struct Layer {
    /// key cache: `seq_len x kv_dim`.
    pub k_cache: Vec<f32>,
    /// value cache: `seq_len x kv_dim`.
    pub v_cache: Vec<f32>,
}

impl Layer {
    pub fn new(config: &Config) -> Self {
        let len = config.seq_len() * config.kv_dim();
        Self {
            k_cache: vec![0.; len],
            v_cache: vec![0.; len],
        }
    }
}

pub(super) struct RotaryEmbedder {
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

    pub fn run(&self, pos: usize, data: &mut [f32]) {
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

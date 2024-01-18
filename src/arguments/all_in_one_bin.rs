use super::Arguments;
use crate::{kernel::slice, tokenizer::utok};
use memmap2::Mmap;
use std::fs::File;

pub(crate) struct AllInOneBin(Mmap);

impl AllInOneBin {
    #[inline]
    pub fn new(file: File) -> Self {
        Self(unsafe { Mmap::map(&file).expect("failed to map the weights file") })
    }

    #[inline(always)]
    fn config(&self) -> &Config {
        self.config_weights().0
    }

    #[inline(always)]
    fn weights(&self) -> Weights<'_> {
        let (config, weights) = self.config_weights();
        let (head, data, tail) = unsafe { weights.align_to::<f32>() };
        debug_assert!(head.is_empty() && tail.is_empty());
        Weights(config, data)
    }

    #[inline(always)]
    fn config_weights(&self) -> (&Config, &[u8]) {
        let (config, weights) = self.0.split_at(std::mem::size_of::<Config>());
        (unsafe { &*config.as_ptr().cast() }, weights)
    }
}

impl Arguments for AllInOneBin {
    #[inline]
    fn dim(&self) -> usize {
        self.config().dim()
    }

    #[inline]
    fn hidden_dim(&self) -> usize {
        self.config().hidden_dim()
    }

    #[inline]
    fn n_layers(&self) -> usize {
        self.config().n_layers()
    }

    #[inline]
    fn n_heads(&self) -> usize {
        self.config().n_heads()
    }

    #[inline]
    fn n_kv_heads(&self) -> usize {
        self.config().n_kv_heads()
    }

    #[inline]
    fn vocab_size(&self) -> usize {
        self.config().vocab_size()
    }

    #[inline]
    fn seq_len(&self) -> usize {
        self.config().seq_len()
    }

    fn token_embedding_table(&self, token: utok) -> &[f32] {
        let weights = self.weights();
        let data = weights.token_embedding_table().0;
        let dim = weights.0.dim();
        &slice!(data; dim; [token as usize])
    }

    fn rms_att_weight(&self, layer: usize) -> &[f32] {
        let weights = self.weights();
        let data = weights.rms_att_weight().0;
        let dim = weights.0.dim();
        &slice!(data; dim; [layer])
    }

    fn rms_ffn_weight(&self, layer: usize) -> &[f32] {
        let weights = self.weights();
        let data = weights.rms_ffn_weight().0;
        let dim = weights.0.dim();
        &slice!(data; dim; [layer])
    }

    fn wq(&self, layer: usize) -> &[f32] {
        let weights = self.weights();
        let data = weights.wq().0;
        let dim = weights.0.dim();
        &slice!(data; dim * dim; [layer])
    }

    fn wk(&self, layer: usize) -> &[f32] {
        let weights = self.weights();
        let data = weights.wk().0;
        let kv_dim = weights.0.kv_dim();
        let dim = weights.0.dim();
        &slice!(data; kv_dim * dim; [layer])
    }

    fn wv(&self, layer: usize) -> &[f32] {
        let weights = self.weights();
        let data = weights.wv().0;
        let kv_dim = weights.0.kv_dim();
        let dim = weights.0.dim();
        &slice!(data; kv_dim * dim; [layer])
    }

    fn wo(&self, layer: usize) -> &[f32] {
        let weights = self.weights();
        let data = weights.wo().0;
        let dim = weights.0.dim();
        &slice!(data; dim * dim; [layer])
    }

    fn w1(&self, layer: usize) -> &[f32] {
        let weights = self.weights();
        let data = weights.w1().0;
        let dim = weights.0.dim();
        let hidden_dim = weights.0.hidden_dim();
        &slice!(data; dim * hidden_dim; [layer])
    }

    fn w2(&self, layer: usize) -> &[f32] {
        let weights = self.weights();
        let data = weights.w2().0;
        let dim = weights.0.dim();
        let hidden_dim = weights.0.hidden_dim();
        &slice!(data; hidden_dim * dim; [layer])
    }

    fn w3(&self, layer: usize) -> &[f32] {
        let weights = self.weights();
        let data = weights.w3().0;
        let dim = weights.0.dim();
        let hidden_dim = weights.0.hidden_dim();
        &slice!(data; dim * hidden_dim; [layer])
    }

    fn rms_final_weight(&self) -> &[f32] {
        self.weights().rms_final_weight().0
    }

    fn wcls(&self) -> &[f32] {
        self.weights().wcls().0
    }
}

#[derive(Debug)]
#[repr(C)]
struct Config {
    dim: i32,
    hidden_dim: i32,
    n_layers: i32,
    n_heads: i32,
    n_kv_heads: i32,
    vocab_size: i32,
    seq_len: i32,
}

impl Config {
    #[inline]
    pub const fn dim(&self) -> usize {
        self.dim as _
    }

    #[inline]
    pub const fn hidden_dim(&self) -> usize {
        self.hidden_dim as _
    }

    #[inline]
    pub const fn n_layers(&self) -> usize {
        self.n_layers as _
    }

    #[inline]
    pub const fn n_heads(&self) -> usize {
        self.n_heads as _
    }

    #[inline]
    pub const fn n_kv_heads(&self) -> usize {
        self.n_kv_heads as _
    }

    #[inline]
    pub const fn shared_weight(&self) -> bool {
        self.vocab_size > 0
    }

    #[inline]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size.unsigned_abs() as _
    }

    #[inline]
    pub const fn seq_len(&self) -> usize {
        self.seq_len as _
    }

    #[inline]
    pub const fn kv_dim(&self) -> usize {
        (self.dim * self.n_kv_heads / self.n_heads) as _
    }
}

pub(super) struct Weights<'a>(&'a Config, &'a [f32]);

impl<'a> Weights<'a> {
    fn token_embedding_table(&self) -> (&'a [f32], &'a [f32]) {
        self.1.split_at(self.0.vocab_size() * self.0.dim())
    }

    fn rms_att_weight(&self) -> (&'a [f32], &'a [f32]) {
        self.token_embedding_table()
            .1
            .split_at(self.0.n_layers() * self.0.dim())
    }

    fn rms_ffn_weight(&self) -> (&'a [f32], &'a [f32]) {
        self.wo().1.split_at(self.0.n_layers() * self.0.dim())
    }

    fn wq(&self) -> (&'a [f32], &'a [f32]) {
        self.rms_att_weight()
            .1
            .split_at(self.0.n_layers() * self.0.dim() * self.0.dim())
    }

    fn wk(&self) -> (&'a [f32], &'a [f32]) {
        self.wq()
            .1
            .split_at(self.0.n_layers() * self.0.kv_dim() * self.0.dim())
    }

    fn wv(&self) -> (&'a [f32], &'a [f32]) {
        self.wk()
            .1
            .split_at(self.0.n_layers() * self.0.kv_dim() * self.0.dim())
    }

    fn wo(&self) -> (&'a [f32], &'a [f32]) {
        self.wv()
            .1
            .split_at(self.0.n_layers() * self.0.dim() * self.0.dim())
    }

    fn w1(&self) -> (&'a [f32], &'a [f32]) {
        self.rms_ffn_weight()
            .1
            .split_at(self.0.n_layers() * self.0.dim() * self.0.hidden_dim())
    }

    fn w2(&self) -> (&'a [f32], &'a [f32]) {
        self.w1()
            .1
            .split_at(self.0.n_layers() * self.0.hidden_dim() * self.0.dim())
    }

    fn w3(&self) -> (&'a [f32], &'a [f32]) {
        self.w2()
            .1
            .split_at(self.0.n_layers() * self.0.dim() * self.0.hidden_dim())
    }

    fn rms_final_weight(&self) -> (&'a [f32], &'a [f32]) {
        self.w3().1.split_at(self.0.dim())
    }

    fn wcls(&self) -> (&'a [f32], &'a [f32]) {
        let data = &self.rms_final_weight().1
            [self.0.seq_len() * (self.0.dim() / self.0.n_heads()) / 2 * 2..];
        let wcls = if self.0.shared_weight() {
            self.token_embedding_table().0
        } else {
            let (wcls, data) = data.split_at(self.0.vocab_size() * self.0.dim());
            debug_assert!(data.is_empty());
            wcls
        };
        (wcls, &[])
    }
}

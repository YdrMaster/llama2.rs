use crate::transformer::config::Config;
use memmap2::Mmap;

pub(super) struct Weights<'a> {
    pub token_embedding_table: &'a [f32],
    pub rms_att_weight: &'a [f32],
    pub rms_ffn_weight: &'a [f32],
    pub wq: &'a [f32],
    pub wk: &'a [f32],
    pub wv: &'a [f32],
    pub wo: &'a [f32],
    pub w1: &'a [f32],
    pub w2: &'a [f32],
    pub w3: &'a [f32],
    pub rms_final_weight: &'a [f32],
    pub wcls: &'a [f32],
}

impl<'a> Weights<'a> {
    pub fn new(mmap: &'a Mmap) -> Self {
        let (config, data) = Config::map(mmap);

        let dim = config.dim();
        let hidden_dim = config.hidden_dim();
        let n_layers = config.n_layers();
        let n_heads = config.n_heads();
        let shared_weight = config.shared_weight();
        let vocab_size = config.vocab_size();
        let seq_len = config.seq_len();
        let kv_dim = config.kv_dim();

        let (head, data, tail) = unsafe { data.align_to::<f32>() };
        assert!(head.is_empty() && tail.is_empty());

        let (token_embedding_table, data) = data.split_at(vocab_size * dim);
        let (rms_att_weight, data) = data.split_at(n_layers * dim);
        let (wq, data) = data.split_at(n_layers * dim * dim);
        let (wk, data) = data.split_at(n_layers * dim * kv_dim);
        let (wv, data) = data.split_at(n_layers * dim * kv_dim);
        let (wo, data) = data.split_at(n_layers * dim * dim);
        let (rms_ffn_weight, data) = data.split_at(n_layers * dim);
        let (w1, data) = data.split_at(n_layers * dim * hidden_dim);
        let (w2, data) = data.split_at(n_layers * hidden_dim * dim);
        let (w3, data) = data.split_at(n_layers * dim * hidden_dim);
        let (rms_final_weight, data) = data.split_at(dim);
        let data = &data[seq_len * (dim / n_heads) / 2 * 2..];
        let wcls = if shared_weight {
            token_embedding_table
        } else {
            let (wcls, data) = data.split_at(vocab_size * dim);
            debug_assert!(data.is_empty());
            wcls
        };

        Self {
            token_embedding_table,
            rms_att_weight,
            rms_ffn_weight,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            w3,
            rms_final_weight,
            wcls,
        }
    }
}

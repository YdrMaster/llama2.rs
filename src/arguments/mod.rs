mod all_in_one_bin;
mod safetensors;

use crate::tokenizer::utok;

pub(crate) use all_in_one_bin::AllInOneBin;
pub(crate) use safetensors::SafeTensors;

pub(crate) trait Arguments {
    fn dim(&self) -> usize;
    fn hidden_dim(&self) -> usize;
    fn n_layers(&self) -> usize;
    fn n_heads(&self) -> usize;
    fn n_kv_heads(&self) -> usize;
    fn vocab_size(&self) -> usize;
    fn seq_len(&self) -> usize;
    fn kv_dim(&self) -> usize {
        self.dim() * self.n_kv_heads() / self.n_heads()
    }

    /// `vocab_size * dim`.
    fn token_embedding_table(&self, token: utok) -> &[f32];
    /// `dim`.
    fn rms_att_weight(&self, layer: usize) -> &[f32];
    /// `dim`.
    fn rms_ffn_weight(&self, layer: usize) -> &[f32];
    /// `dim * dim`.
    fn wq(&self, layer: usize) -> &[f32];
    /// `kv_dim * dim`.
    fn wk(&self, layer: usize) -> &[f32];
    /// `kv_dim * dim`.
    fn wv(&self, layer: usize) -> &[f32];
    /// `dim * dim`.
    fn wo(&self, layer: usize) -> &[f32];
    /// `dim * hidden_dim`.
    fn w1(&self, layer: usize) -> &[f32];
    /// `hidden_dim * dim`.
    fn w2(&self, layer: usize) -> &[f32];
    /// `dim * hidden_dim`.
    fn w3(&self, layer: usize) -> &[f32];
    /// `dim`.
    fn rms_final_weight(&self) -> &[f32];
    fn wcls(&self) -> &[f32];
}

#[test]
fn test() {
    use std::fs::File;
    let Ok(bin) = File::open("test.bin") else {
        return;
    };
    let Ok(config) = File::open("config.json") else {
        return;
    };
    let Ok(safetensors) = File::open("model.safetensors") else {
        return;
    };
    let a = AllInOneBin::new(bin);
    let s = SafeTensors::new(config, safetensors);
    assert_eq!(a.dim(), s.dim());
    assert_eq!(a.hidden_dim(), s.hidden_dim());
    assert_eq!(a.n_layers(), s.n_layers());
    assert_eq!(a.n_heads(), s.n_heads());
    assert_eq!(a.n_kv_heads(), s.n_kv_heads());
    assert_eq!(a.vocab_size(), s.vocab_size());
    assert_eq!(a.seq_len(), s.seq_len());
    assert_eq!(a.kv_dim(), s.kv_dim());

    for i in 0..a.vocab_size() as utok {
        assert_eq!(a.token_embedding_table(i), s.token_embedding_table(i));
    }
    for i in 0..a.n_layers() {
        assert_eq!(a.rms_att_weight(i), s.rms_att_weight(i));
        assert_eq!(a.rms_ffn_weight(i), s.rms_ffn_weight(i));
        assert_eq!(a.wq(i), s.wq(i));
        assert_eq!(a.wk(i), s.wk(i));
        assert_eq!(a.wv(i), s.wv(i));
        assert_eq!(a.wo(i), s.wo(i));
        assert_eq!(a.w1(i), s.w1(i));
        assert_eq!(a.w2(i), s.w2(i));
        assert_eq!(a.w3(i), s.w3(i));
    }
    assert_eq!(a.rms_final_weight(), s.rms_final_weight());
    assert_eq!(a.wcls(), s.wcls());
}

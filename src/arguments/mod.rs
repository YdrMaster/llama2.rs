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

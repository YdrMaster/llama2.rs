use super::Arguments;
use crate::{kernel::slice, tokenizer::utok};
use memmap2::Mmap;
use std::{borrow::Cow, collections::HashMap, fs::File, io::Read, ops::Range};

pub(crate) struct SafeTensors {
    config: LLamaConfig,
    token_embedding_table: Vec<f32>,
    rms_att_weight: Vec<f32>,
    rms_ffn_weight: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    w1: Vec<f32>,
    w2: Vec<f32>,
    w3: Vec<f32>,
    rms_final_weight: Vec<f32>,
    wcls: Vec<f32>,
}

impl SafeTensors {
    pub fn new(mut config: File, safetensors: File) -> Self {
        let mut config_string = String::new();
        let _ = config.read_to_string(&mut config_string).unwrap();
        let config = serde_json::from_str::<LLamaConfig>(&config_string).unwrap();

        let mmap = unsafe { Mmap::map(&safetensors) }.unwrap();
        let (len, tail) = mmap.split_at(std::mem::size_of::<u64>());
        let len = unsafe { *len.as_ptr().cast::<u64>() } as usize;
        let (meta_json, data) = tail.split_at(len);
        let meta_json = serde_json::from_slice::<MetaJson>(meta_json).unwrap();

        let vocab_size = config.vocab_size;
        let n_layers = config.num_hidden_layers;
        let dim = config.hidden_size;
        let kv_dim = dim * config.num_key_value_heads / config.num_attention_heads;
        let hidden_dim = config.intermediate_size;

        let mut token_embedding_table = vec![0.; vocab_size * dim];
        let mut rms_att_weight = vec![0.; n_layers * dim];
        let mut rms_ffn_weight = vec![0.; n_layers * dim];
        let mut wq = vec![0.; n_layers * dim * dim];
        let mut wk = vec![0.; n_layers * kv_dim * dim];
        let mut wv = vec![0.; n_layers * kv_dim * dim];
        let mut wo = vec![0.; n_layers * dim * dim];
        let mut w1 = vec![0.; n_layers * dim * hidden_dim];
        let mut w2 = vec![0.; n_layers * hidden_dim * dim];
        let mut w3 = vec![0.; n_layers * dim * hidden_dim];
        let mut rms_final_weight = vec![0.; dim];
        let mut wcls = vec![0.; vocab_size * dim];

        for (name, tensor) in meta_json.tensors {
            let path = name.split('.').collect::<Vec<_>>();
            match path.as_slice() {
                ["model", "embed_tokens", "weight"] => {
                    assert_eq!(&tensor.shape, &[vocab_size, dim]);
                    let data = cast_slice(&tensor.dtype, &data[tensor.data_offsets.clone()]);
                    token_embedding_table.copy_from_slice(&data);
                }
                ["model", "layers", n, path @ .., "weight"] => {
                    let data = cast_slice(&tensor.dtype, &data[tensor.data_offsets.clone()]);
                    let layer = n.parse::<usize>().unwrap();

                    let copy_slice =
                        |dst: &mut [f32]| slice!(dst; data.len(); [layer]).copy_from_slice(&data);
                    let perm_copy = |dst: &mut [f32]| {
                        let dst = &mut slice!(dst; data.len(); [layer]);
                        let (head, part) = if tensor.shape[0] == tensor.shape[1] {
                            let n_head = config.num_attention_heads;
                            (n_head, dim / n_head / 2)
                        } else {
                            let n_kv_head = config.num_key_value_heads;
                            (n_kv_head, kv_dim / n_kv_head / 2)
                        };
                        for i in 0..head {
                            let t = i * part * 2;
                            for j in 0..part {
                                slice!(dst; dim; [t + 2 * j    ])
                                    .copy_from_slice(&slice!(data; dim; [t        + j]));
                                slice!(dst; dim; [t + 2 * j + 1])
                                    .copy_from_slice(&slice!(data; dim; [t + part + j]));
                            }
                        }
                    };

                    match path {
                        ["input_layernorm"] => {
                            assert_eq!(&tensor.shape, &[dim]);
                            copy_slice(&mut rms_att_weight);
                        }
                        ["self_attn", "q_proj"] => {
                            assert_eq!(&tensor.shape, &[dim, dim]);
                            perm_copy(&mut wq);
                        }
                        ["self_attn", "k_proj"] => {
                            assert_eq!(&tensor.shape, &[kv_dim, dim]);
                            perm_copy(&mut wk);
                        }
                        ["self_attn", "v_proj"] => {
                            assert_eq!(&tensor.shape, &[kv_dim, dim]);
                            copy_slice(&mut wv);
                        }
                        ["self_attn", "o_proj"] => {
                            assert_eq!(&tensor.shape, &[dim, dim]);
                            copy_slice(&mut wo);
                        }
                        ["post_attention_layernorm"] => {
                            assert_eq!(&tensor.shape, &[dim]);
                            copy_slice(&mut rms_ffn_weight);
                        }
                        ["mlp", "gate_proj"] => {
                            assert_eq!(&tensor.shape, &[hidden_dim, dim]);
                            copy_slice(&mut w1);
                        }
                        ["mlp", "down_proj"] => {
                            assert_eq!(&tensor.shape, &[dim, hidden_dim]);
                            copy_slice(&mut w2);
                        }
                        ["mlp", "up_proj"] => {
                            assert_eq!(&tensor.shape, &[hidden_dim, dim]);
                            copy_slice(&mut w3);
                        }
                        [..] => {}
                    };
                }
                ["model", "norm", "weight"] => {
                    assert_eq!(&tensor.shape, &[dim]);
                    let data = cast_slice(&tensor.dtype, &data[tensor.data_offsets.clone()]);
                    rms_final_weight.copy_from_slice(&data);
                }
                ["lm_head", "weight"] => {
                    assert_eq!(&tensor.shape, &[vocab_size, dim]);
                    let data = cast_slice(&tensor.dtype, &data[tensor.data_offsets.clone()]);
                    wcls.copy_from_slice(&data);
                }
                [..] => {}
            }
        }

        Self {
            config,
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

fn cast_slice<'a>(dtype: &str, slice: &'a [u8]) -> Cow<'a, [f32]> {
    #[inline]
    fn reslice<T>(slice: &[u8]) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(
                slice.as_ptr().cast(),
                slice.len() / std::mem::size_of::<T>(),
            )
        }
    }

    use half::{bf16, f16};
    match dtype {
        "F32" => Cow::Borrowed(reslice(slice)),
        "F16" => Cow::Owned(
            reslice::<f16>(slice)
                .iter()
                .map(|x| x.to_f32())
                .collect::<Vec<_>>(),
        ),
        "BF16" => Cow::Owned(
            reslice::<bf16>(slice)
                .iter()
                .map(|x| x.to_f32())
                .collect::<Vec<_>>(),
        ),
        _ => todo!(),
    }
}

impl Arguments for SafeTensors {
    fn dim(&self) -> usize {
        self.config.hidden_size
    }

    fn hidden_dim(&self) -> usize {
        self.config.intermediate_size
    }

    fn n_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn n_heads(&self) -> usize {
        self.config.num_attention_heads
    }

    fn n_kv_heads(&self) -> usize {
        self.config.num_key_value_heads
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn seq_len(&self) -> usize {
        self.config.max_position_embeddings
    }

    fn token_embedding_table(&self, token: utok) -> &[f32] {
        &slice!(self.token_embedding_table; self.dim(); [token as usize])
    }

    fn rms_att_weight(&self, layer: usize) -> &[f32] {
        &slice!(self.rms_att_weight; self.dim(); [layer])
    }

    fn rms_ffn_weight(&self, layer: usize) -> &[f32] {
        &slice!(self.rms_ffn_weight; self.dim(); [layer])
    }

    fn wq(&self, layer: usize) -> &[f32] {
        &slice!(self.wq; self.dim() * self.dim(); [layer])
    }

    fn wk(&self, layer: usize) -> &[f32] {
        &slice!(self.wk; self.kv_dim() * self.dim(); [layer])
    }

    fn wv(&self, layer: usize) -> &[f32] {
        &slice!(self.wv; self.kv_dim() * self.dim(); [layer])
    }

    fn wo(&self, layer: usize) -> &[f32] {
        &slice!(self.wo; self.dim() * self.dim(); [layer])
    }

    fn w1(&self, layer: usize) -> &[f32] {
        &slice!(self.w1; self.dim() * self.hidden_dim(); [layer])
    }

    fn w2(&self, layer: usize) -> &[f32] {
        &slice!(self.w2; self.hidden_dim() * self.dim(); [layer])
    }

    fn w3(&self, layer: usize) -> &[f32] {
        &slice!(self.w3; self.dim() * self.hidden_dim(); [layer])
    }

    fn rms_final_weight(&self) -> &[f32] {
        &self.rms_final_weight
    }

    fn wcls(&self) -> &[f32] {
        &self.wcls
    }
}

#[derive(serde::Deserialize, Debug)]
struct LLamaConfig {
    hidden_size: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    vocab_size: usize,
}

#[derive(serde::Deserialize, Debug)]
struct MetaJson {
    #[serde(flatten)]
    tensors: HashMap<String, Tensor>,
    #[serde(rename = "__metadata__")]
    #[allow(dead_code)]
    meta: HashMap<String, serde_json::Value>,
}

#[derive(serde::Deserialize, Debug)]
struct Tensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: Range<usize>,
}

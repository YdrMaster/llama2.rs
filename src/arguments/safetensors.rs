use super::Arguments;
use crate::{kernel::slice, tokenizer::utok};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::{tensor::TensorInfo, Dtype};
use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap},
    fs::File,
    io::{Read, Write},
    iter::zip,
};

pub struct SafeTensors {
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
            let data = cast_slice(data, &tensor);

            match path.as_slice() {
                ["model", "embed_tokens", "weight"] => {
                    assert_eq!(&tensor.shape, &[vocab_size, dim]);
                    token_embedding_table.copy_from_slice(&data);
                }
                ["model", "layers", n, path @ .., "weight"] => {
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
                    rms_final_weight.copy_from_slice(&data);
                }
                ["lm_head", "weight"] => {
                    assert_eq!(&tensor.shape, &[vocab_size, dim]);
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

    pub fn cast_f32(mut config: File, safetensors: File, stream: &mut dyn Write) -> String {
        let mut config_string = String::new();
        config.read_to_string(&mut config_string).unwrap();
        let config = serde_json::from_str::<LLamaConfig>(&config_string).unwrap();

        let mmap = unsafe { Mmap::map(&safetensors) }.unwrap();
        let (len, tail) = mmap.split_at(std::mem::size_of::<u64>());
        let len = unsafe { *len.as_ptr().cast::<u64>() } as usize;
        let (meta_json, data) = tail.split_at(len);
        let MetaJson { tensors, meta } = serde_json::from_slice::<MetaJson>(meta_json).unwrap();

        let mut out_meta = MetaJson {
            tensors: Default::default(),
            meta,
        };
        let mut range = 0usize..0usize;

        let order = tensors.keys().collect::<Vec<_>>();
        for &name in &order {
            let tensor = &tensors[name];
            let dtype = match tensor.dtype {
                Dtype::F16 | Dtype::BF16 => {
                    range.end += (tensor.data_offsets.1 - tensor.data_offsets.0) * 2;
                    Dtype::F32
                }
                others => {
                    range.end += tensor.data_offsets.1 - tensor.data_offsets.0;
                    others
                }
            };
            out_meta.tensors.insert(
                name.clone(),
                TensorInfo {
                    dtype,
                    shape: tensor.shape.clone(),
                    data_offsets: (range.start, range.end),
                },
            );
            range.start = range.end;
            println!("cast: \"{name}\".");
        }
        {
            let str = serde_json::to_string(&out_meta).unwrap();
            let len = str.len();
            const ALIGN: usize = std::mem::size_of::<usize>();
            let expand = (len + ALIGN - 1) & !(ALIGN - 1);
            stream.write_all(&(expand as u64).to_le_bytes()).unwrap();
            stream.write_all(str.as_bytes()).unwrap();
            for _ in len..expand {
                stream.write_all(&[32]).unwrap();
            }
        }
        for name in order {
            let tensor = &tensors[name];
            let data = &data[tensor.data_offsets.0..tensor.data_offsets.1];
            match tensor.dtype {
                Dtype::F16 => {
                    let src = reslice::<f16>(data);
                    let mut buf = vec![0u8; src.len() * std::mem::size_of::<f32>()];
                    let dst = reslice_mut::<f32>(&mut buf);
                    for (dst, src) in zip(dst, src) {
                        *dst = src.to_f32();
                    }
                    print!("writeing {:>10} bytes... ", buf.len());
                    stream.write_all(&buf).unwrap();
                }
                Dtype::BF16 => {
                    let src = reslice::<bf16>(data);
                    let mut buf = vec![0u8; src.len() * std::mem::size_of::<f32>()];
                    let dst = reslice_mut::<f32>(&mut buf);
                    for (dst, src) in zip(dst, src) {
                        *dst = src.to_f32();
                    }
                    print!("writeing {:>10} bytes... ", buf.len());
                    stream.write_all(&buf).unwrap();
                }
                _ => {
                    print!("writeing {:>10} bytes... ", data.len());
                    stream.write_all(data).unwrap();
                }
            }
            println!("copied: \"{name}\".");
        }

        serde_json::to_string_pretty(&LLamaConfig {
            torch_dtype: "float32".to_string(),
            ..config
        })
        .unwrap()
    }
}

#[inline]
fn reslice<T>(slice: &[u8]) -> &[T] {
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr().cast(),
            slice.len() / std::mem::size_of::<T>(),
        )
    }
}

#[inline]
fn reslice_mut<T>(slice: &mut [u8]) -> &mut [T] {
    unsafe {
        std::slice::from_raw_parts_mut(
            slice.as_mut_ptr().cast(),
            slice.len() / std::mem::size_of::<T>(),
        )
    }
}

fn cast_slice<'a>(data: &'a [u8], tensor: &TensorInfo) -> Cow<'a, [f32]> {
    let slice = &data[tensor.data_offsets.0..tensor.data_offsets.1];
    match tensor.dtype {
        Dtype::F32 => Cow::Borrowed(reslice(slice)),
        Dtype::F16 => Cow::Owned(
            reslice::<f16>(slice)
                .iter()
                .map(|x| x.to_f32())
                .collect::<Vec<_>>(),
        ),
        Dtype::BF16 => Cow::Owned(
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

#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct LLamaConfig {
    bos_token_id: utok,
    eos_token_id: utok,

    hidden_size: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    vocab_size: usize,

    torch_dtype: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct MetaJson {
    #[serde(flatten)]
    tensors: BTreeMap<String, TensorInfo>,
    #[serde(rename = "__metadata__")]
    meta: HashMap<String, serde_json::Value>,
}

mod bpe32000;
mod longest_prefix;

/// `utok` for token id.
#[allow(non_camel_case_types)]
pub(super) type utok = u32;

pub const _UNKNOWN: utok = 0;
pub const BOS: utok = 1;
pub const EOS: utok = 2;

pub trait Tokenizer {
    fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<utok>;
    fn decode(&self, token: utok, next: utok) -> &str;
}

pub use bpe32000::BpeTokenizer;
pub use longest_prefix::LongestPrefix;

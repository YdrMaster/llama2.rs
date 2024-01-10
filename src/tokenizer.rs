use memmap2::Mmap;
use std::{fs::File, path::Path};

pub(super) struct Tokenizer {
    mmap: Mmap,
    vocab_offset: Vec<usize>,
    sorted_vocab: Vec<TokenIndex>,
}

impl Tokenizer {
    pub fn new(tokenizer: impl AsRef<Path>, vocab_size: usize) -> Self {
        let mmap = unsafe { Mmap::map(&File::open(tokenizer).unwrap()) }.unwrap();

        let mut vocab_offset = Vec::<usize>::with_capacity(vocab_size);
        let mut sorted_vocab = Vec::<TokenIndex>::with_capacity(vocab_size);
        {
            let mut offset = std::mem::size_of::<u32>();
            for index in 0..vocab_size {
                let ptr = mmap.as_ref()[offset..].as_ptr();
                let header = unsafe { &*ptr.cast::<TokenHeader>() };
                vocab_offset.push(offset);
                sorted_vocab.push(TokenIndex { index, offset });
                offset += std::mem::size_of::<TokenHeader>() + header.len as usize;
            }
        }
        sorted_vocab.sort_by_key(|token| map_token(&mmap, token).1);

        Self {
            mmap,
            vocab_offset,
            sorted_vocab,
        }
    }

    #[inline]
    pub fn max_token_len(&self) -> usize {
        unsafe { *self.mmap.as_ptr().cast::<u32>() as usize }
    }

    pub fn encode(&mut self, text: &str, bos: bool, eos: bool) -> Vec<u32> {
        const _UNKNOWN: u32 = 0;
        const BOS: u32 = 1;
        const EOS: u32 = 2;
        #[inline(always)]
        const fn byte_index(b: u8) -> u32 {
            b as u32 + 3
        }

        let mut tokens = Vec::<u32>::with_capacity(text.len() + 2);
        if bos {
            tokens.push(BOS);
        }
        if !text.is_empty() {
            tokens.push(self.find_token(" ").unwrap().index as _)
        }

        text.chars().map(|c| c.to_string()).for_each(|c| {
            if let Some(token) = self.find_token(&c) {
                tokens.extend([token.index as u32]);
            } else {
                tokens.extend(c.bytes().map(byte_index));
            }
        });

        loop {
            let mut best_score = std::f32::NEG_INFINITY;
            let mut replacement = None;
            for (i, pair) in tokens.windows(2).enumerate() {
                let pair = format!("{}{}", self.map_str(pair[0]), self.map_str(pair[1]));
                if let Some(token) = self.find_token(&pair) {
                    let score = map_token(&self.mmap, token).0.score;
                    if score > best_score {
                        best_score = score;
                        replacement = Some((i, token.index as u32));
                    }
                }
            }
            match replacement {
                Some((i, j)) => {
                    tokens[i] = j;
                    tokens.remove(i + 1);
                }
                None => break,
            }
        }

        if bos {
            assert_eq!(tokens[0], BOS);
        }
        if eos {
            tokens.push(EOS);
        }
        tokens
    }

    #[inline]
    fn find_token(&self, token: &str) -> Option<&TokenIndex> {
        self.sorted_vocab
            .binary_search_by_key(&token, |token| map_token(&self.mmap, token).1)
            .ok()
            .map(|idx| &self.sorted_vocab[idx])
    }

    #[inline]
    fn map_str(&self, index: u32) -> &str {
        let offset = self.vocab_offset[index as usize];
        map_token(&self.mmap, &TokenIndex { index: 0, offset }).1
    }
}

#[repr(C)]
struct TokenHeader {
    score: f32,
    len: u32,
}

struct TokenIndex {
    index: usize,
    offset: usize,
}

#[inline]
fn map_token<'a>(mmap: &'a Mmap, token: &TokenIndex) -> (&'a TokenHeader, &'a str) {
    let slice = &mmap.as_ref()[token.offset..];
    let header = unsafe { &*slice.as_ptr().cast::<TokenHeader>() };
    let slice = &slice[std::mem::size_of::<TokenHeader>()..][..header.len as usize];
    (header, unsafe { std::str::from_utf8_unchecked(slice) })
}

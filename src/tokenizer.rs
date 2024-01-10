use memmap2::Mmap;
use std::{fs::File, path::Path};

/// `utok` for token id.
#[allow(non_camel_case_types)]
pub(super) type utok = u32;

pub(super) const _UNKNOWN: utok = 0;
pub(super) const BOS: utok = 1;
pub(super) const EOS: utok = 2;

/// Tokenizer 的功能是建立 token 字符串和一个序号之间的关系。
pub(super) struct Tokenizer {
    /// tokenizer 文件的内存映射。
    mmap: Mmap,
    /// 保存每个序号对应的对象在文件中的偏移，用于从序号查询 token 字符串。
    words_offset: Vec<usize>,
    /// 保存根据 token 字符串字典序排序的序号，用于从 token 字符串查询序号。
    sorted_indices: Vec<utok>,
}

impl Tokenizer {
    pub fn new(tokenizer: impl AsRef<Path>, vocab_size: usize) -> Self {
        let mmap = unsafe { Mmap::map(&File::open(tokenizer).unwrap()) }.unwrap();

        let mut words_offset = Vec::<usize>::with_capacity(vocab_size);
        let mut sorted_indices = Vec::<utok>::with_capacity(vocab_size);
        {
            let mut offset = std::mem::size_of::<u32>();
            for index in 0..vocab_size as utok {
                words_offset.push(offset);
                sorted_indices.push(index);
                offset += file::item_len(&mmap, offset);
            }
        }
        sorted_indices.sort_by_key(|&index| file::map(&mmap, words_offset[index as usize]).0);

        Self {
            mmap,
            words_offset,
            sorted_indices,
        }
    }

    #[inline]
    pub fn max_token_len(&self) -> usize {
        unsafe { *self.mmap.as_ptr().cast::<u32>() as usize }
    }

    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<utok> {
        #[inline(always)]
        const fn byte_index(b: u8) -> utok {
            b as utok + 3
        }

        let mut tokens = Vec::<utok>::with_capacity(text.len() + 2);
        if bos {
            tokens.push(BOS);
        }
        if !text.is_empty() {
            tokens.push(self.find_token(" ").unwrap())
        }

        text.chars().map(|c| c.to_string()).for_each(|c| {
            if let Some(index) = self.find_token(&c) {
                tokens.extend([index]);
            } else {
                tokens.extend(c.bytes().map(byte_index));
            }
        });

        loop {
            let mut best_score = std::f32::NEG_INFINITY;
            let mut replacement = None;
            for (i, pair) in tokens.windows(2).enumerate() {
                let pair = format!("{}{}", self.map_str(pair[0]), self.map_str(pair[1]));
                if let Some(index) = self.find_token(&pair) {
                    let (_, score) = file::map(&self.mmap, self.words_offset[index as usize]);
                    if score > best_score {
                        best_score = score;
                        replacement = Some((i, index));
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

    pub fn decode(&self, token: utok, next: utok) -> String {
        todo!()
    }

    #[inline]
    fn find_token(&self, token: &str) -> Option<utok> {
        self.sorted_indices
            .binary_search_by_key(&token, |&index| self.map_str(index))
            .ok()
            .map(|idx| self.sorted_indices[idx])
    }

    #[inline]
    fn map_str(&self, index: utok) -> &str {
        file::map(&self.mmap, self.words_offset[index as usize]).0
    }
}

mod file {
    //! 文件结构：
    //!
    //! ```plain_text
    //! (
    //!     max_token_len:u32,
    //!     [(
    //!         score: f32,
    //!         len  : u32,
    //!         text : [u8; len]
    //!     ); vocab_size]
    //! )
    //! ```

    use memmap2::Mmap;

    #[repr(C)]
    struct TokenHeader {
        score: f32,
        len: u32,
    }

    /// 获取 `offset` 处对象的长度。
    #[inline]
    pub fn item_len(mmap: &Mmap, offset: usize) -> usize {
        let slice = &mmap.as_ref()[offset..];
        let header = unsafe { slice.as_ptr().cast::<TokenHeader>().read_unaligned() };
        std::mem::size_of::<TokenHeader>() + header.len as usize
    }

    /// 获取 `offset` 处对象的内容。
    #[inline]
    pub fn map<'a>(mmap: &'a Mmap, offset: usize) -> (&'a str, f32) {
        let slice = &mmap.as_ref()[offset..];
        let header = unsafe { slice.as_ptr().cast::<TokenHeader>().read_unaligned() };
        let slice = &slice[std::mem::size_of::<TokenHeader>()..][..header.len as usize];
        (
            unsafe { std::str::from_utf8_unchecked(slice) },
            header.score,
        )
    }
}

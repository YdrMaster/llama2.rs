use memmap2::Mmap;
use std::{fs::File, io::Read, mem::MaybeUninit, path::Path};

pub(super) struct Tokenizer {
    mmap: Mmap,
    token_indices: Vec<TokenIndex>,
}

impl Tokenizer {
    pub fn new(tokenizer: impl AsRef<Path>, vocab_size: usize) -> Self {
        let mmap = unsafe { Mmap::map(&File::open(tokenizer).unwrap()) }.unwrap();
        fn read_from<T: Copy>(reader: &mut impl Read) -> T {
            let mut buf = MaybeUninit::<T>::uninit();
            reader
                .read_exact(unsafe {
                    std::slice::from_raw_parts_mut(
                        buf.as_mut_ptr().cast::<u8>(),
                        std::mem::size_of::<T>(),
                    )
                })
                .unwrap();
            unsafe { buf.assume_init() }
        }

        let mut token_indices = (0..vocab_size)
            .scan(std::mem::size_of::<u32>(), |next, index| {
                let ptr = mmap.as_ref()[*next..].as_ptr();
                let header = unsafe { &*ptr.cast::<TokenHeader>() };
                let offset = *next;
                *next += std::mem::size_of::<TokenHeader>() + header.len as usize;
                Some(TokenIndex { index, offset })
            })
            .collect::<Vec<_>>();
        token_indices.sort_by_key(|idx| slice_str(&mmap, idx));

        Self {
            mmap,
            token_indices,
        }
    }

    #[inline]
    pub fn max_token_len(&self) -> usize {
        unsafe { *self.mmap.as_ptr().cast::<u32>() as usize }
    }

    pub fn encode(&mut self, text: &str, bos: bool, eos: bool) -> Vec<u32> {
        const BOS: u32 = 1;
        const EOS: u32 = 2;

        let mut tokens = Vec::<u32>::with_capacity(text.len() + 2);
        if bos {
            tokens.push(BOS);
        }
        tokens
    }

    #[inline]
    fn find_token(&self, key: &str) -> Option<u32> {
        self.token_indices
            .binary_search_by_key(&key, |idx| slice_str(&self.mmap, idx))
            .ok()
            .map(|idx| idx as _)
    }
}

struct TokenIndex {
    index: usize,
    offset: usize,
}

#[repr(C)]
struct TokenHeader {
    score: f32,
    len: u32,
}

#[inline]
fn slice_str<'a>(mmap: &'a Mmap, index: &TokenIndex) -> &'a str {
    let slice = &mmap.as_ref()[index.offset..];
    let header = unsafe { &*slice.as_ptr().cast::<TokenHeader>() };
    let slice = &slice[std::mem::size_of::<TokenHeader>()..][..header.len as usize];
    unsafe { std::str::from_utf8_unchecked(slice) }
}

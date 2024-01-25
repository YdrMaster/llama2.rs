use super::{utok, Tokenizer, BOS, EOS};
use memmap2::Mmap;
use patricia_tree::PatriciaMap;
use std::{fs::File, path::Path};

pub struct LongestPrefix {
    words: Vec<String>,
    trie: PatriciaMap<utok>,
    max_piece_len: usize,
    byte_pieces: [u8; 256],
}

impl LongestPrefix {
    pub fn new(tokenizer: impl AsRef<Path>) -> Self {
        let mmap = unsafe { Mmap::map(&File::open(tokenizer).unwrap()) }.unwrap();
        let text = unsafe { std::str::from_utf8_unchecked(&mmap) };

        let mut words = Vec::new();
        let mut trie = PatriciaMap::new();
        let mut max_piece_len = 0;
        for (i, line) in text.lines().into_iter().enumerate() {
            let piece = line.strip_prefix('"').unwrap().strip_suffix('"').unwrap();
            max_piece_len = max_piece_len.max(piece.len());
            words.push(piece.to_string());
            trie.insert(piece, i as _);
        }
        let mut ans = Self {
            words,
            trie,
            max_piece_len,
            byte_pieces: [0; 256],
        };
        for i in 0..=255u8 {
            ans.byte_pieces[i as usize] = i;
        }
        ans
    }
}

impl Tokenizer for LongestPrefix {
    fn encode(&self, mut text: &str, bos: bool, eos: bool) -> Vec<utok> {
        #[inline(always)]
        const fn byte_index(b: u8) -> utok {
            b as utok + 3
        }

        let mut tokens: Vec<u32> = Vec::<utok>::new();
        if bos {
            tokens.push(BOS);
        }
        if !text.is_empty() {
            tokens.push(*self.trie.get(" ").unwrap())
        }

        while !text.is_empty() {
            let piece = if text.len() > self.max_piece_len {
                &text[..self.max_piece_len]
            } else {
                text
            };
            if let Some((pre, tok)) = self.trie.get_longest_common_prefix(piece) {
                tokens.push(*tok);
                text = &text[pre.len()..];
            } else {
                let mut chars = text.chars();
                let char = chars.next().unwrap();
                tokens.extend(char.to_string().bytes().map(byte_index));
                text = chars.as_str();
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

    fn decode(&self, token: utok, next: utok) -> &str {
        let piece = self.words[next as usize].as_str();
        if let Some(byte) = piece.strip_prefix("<0x").and_then(|s| s.strip_suffix('>')) {
            let byte = u8::from_str_radix(byte, 16).unwrap();
            let byte = &self.byte_pieces[byte as usize..][..1];
            unsafe { std::str::from_utf8_unchecked(byte) }
        } else if token == BOS && piece.starts_with(' ') {
            &piece[1..]
        } else {
            piece
        }
    }
}

use memmap2::Mmap;

#[derive(Clone, Debug)]
#[repr(C)]
pub(super) struct Config {
    dim: i32,
    hidden_dim: i32,
    n_layers: i32,
    n_heads: i32,
    n_kv_heads: i32,
    vocab_size: i32,
    seq_len: i32,
}

impl Config {
    #[inline]
    pub fn map<'a>(mmap: &'a Mmap) -> (&'a Self, &'a [u8]) {
        let (config, data) = mmap.as_ref().split_at(std::mem::size_of::<Self>());
        (unsafe { &*config.as_ptr().cast() }, data)
    }

    #[inline]
    pub const fn dim(&self) -> usize {
        self.dim as _
    }

    #[inline]
    pub const fn hidden_dim(&self) -> usize {
        self.hidden_dim as _
    }

    #[inline]
    pub const fn n_layers(&self) -> usize {
        self.n_layers as _
    }

    #[inline]
    pub const fn n_heads(&self) -> usize {
        self.n_heads as _
    }

    #[inline]
    pub const fn n_kv_heads(&self) -> usize {
        self.n_kv_heads as _
    }

    #[inline]
    pub const fn shared_weight(&self) -> bool {
        self.vocab_size > 0
    }

    #[inline]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size.abs() as _
    }

    #[inline]
    pub const fn seq_len(&self) -> usize {
        self.seq_len as _
    }

    #[inline]
    pub const fn kv_dim(&self) -> usize {
        (self.dim * self.n_kv_heads / self.n_heads) as _
    }
}

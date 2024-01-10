use super::tokenizer::utok;

pub(super) struct Sampler {
    vocab_size: usize,
    temperature: f32,
    top_p: f32,
    rng_state: u64,
}

impl Sampler {
    pub fn new(vocab_size: usize, temperature: f32, top_p: f32, rng_seed: u64) -> Self {
        Self {
            vocab_size,
            temperature,
            top_p,
            rng_state: rng_seed,
        }
    }

    pub fn sample(&mut self, logits: f32) -> utok {
        todo!()
    }
}

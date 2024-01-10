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

    pub fn sample(&mut self, logits: &[f32]) -> utok {
        if self.temperature == 0.0 {
            logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0 as _
        } else {
            todo!()
        }
    }
}

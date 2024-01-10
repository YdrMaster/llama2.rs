use super::{kernel::softmax, tokenizer::utok};

pub(super) struct Sampler {
    temperature: f32,
    top_p: f32,
    rng_state: u64,
    probindex: Vec<ProbIndex>,
}

impl Sampler {
    pub fn new(vocab_size: usize, temperature: f32, top_p: f32, rng_seed: u64) -> Self {
        Self {
            temperature,
            top_p,
            rng_state: rng_seed,
            probindex: Vec::with_capacity(vocab_size),
        }
    }

    pub fn sample(&mut self, logits: &mut [f32]) -> utok {
        if self.temperature == 0.0 {
            logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0 as _
        } else {
            for logit in logits.iter_mut() {
                *logit /= self.temperature;
            }
            softmax(logits);
            let coin = self.random_f32();
            if !(0.0..=1.0).contains(&self.top_p) {
                // FIXME: 这个方案会输出全 <unk>
                sample_mult(logits, coin)
            } else {
                sample_top_p(logits, self.top_p, &mut self.probindex, coin)
            }
        }
    }

    #[inline]
    fn random_f32(&mut self) -> f32 {
        (self.random_u32() >> 8) as f32 / 16777216.0
    }

    #[inline]
    fn random_u32(&mut self) -> u32 {
        self.rng_state ^= self.rng_state >> 12;
        self.rng_state ^= self.rng_state << 25;
        self.rng_state ^= self.rng_state >> 27;
        return ((self.rng_state * 0x2545F4914F6CDD1Du64) >> 32) as _;
    }
}

fn sample_mult(logits: &[f32], coin: f32) -> utok {
    let mut cdf = 0.;
    for (i, logit) in logits.iter().enumerate() {
        cdf += logit;
        if cdf > coin {
            return i as _;
        }
    }
    return (logits.len() - 1) as _;
}

fn sample_top_p(logits: &[f32], top_p: f32, probindex: &mut Vec<ProbIndex>, coin: f32) -> utok {
    probindex.clear();

    let cutoff = (1. - top_p) / (logits.len() - 1) as f32;
    for (i, &prob) in logits.iter().enumerate() {
        if prob >= cutoff {
            probindex.push(ProbIndex {
                prob,
                index: i as _,
            })
        }
    }
    probindex.sort_by(|a, b| a.prob.partial_cmp(&b.prob).unwrap().reverse());

    let mut cumulative_prob = 0.;
    for (i, prob) in probindex.iter().enumerate() {
        cumulative_prob += prob.prob;
        if cumulative_prob > top_p {
            probindex.truncate(i + 1);
            break;
        }
    }

    let r = coin * cumulative_prob;
    let mut cdf = 0.;
    for prob in probindex.iter() {
        cdf += prob.prob;
        if cdf > r {
            return prob.index;
        }
    }
    probindex.last().unwrap().index
}

#[derive(Clone, Default)]
struct ProbIndex {
    prob: f32,
    index: utok,
}

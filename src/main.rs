mod sampler;
mod tokenizer;
mod transformer;

use core::panic;
use sampler::Sampler;
use std::{fs::canonicalize, io::Write, path::PathBuf, time::Instant};
use tokenizer::{Tokenizer, BOS};
use transformer::{upos, Transformer};

fn main() {
    struct Args {
        check_point: PathBuf,
        tokenizer_path: PathBuf,
        temperature: f32,
        top_p: f32,
        steps: usize,
        prompt: String,
        rng_seed: u64,
    }

    let mut process_args = std::env::args();
    process_args.next().unwrap();
    let mut args = Args {
        check_point: process_args
            .next()
            .map(canonicalize)
            .expect(USAGE_HELP)
            .unwrap(),
        tokenizer_path: canonicalize("tokenizer.bin").unwrap(),
        temperature: 1.0,
        top_p: 0.9,
        steps: 256,
        prompt: String::new(),
        rng_seed: 0,
    };
    loop {
        match process_args.next() {
            Some(s) if s == "--tokenizer-path" => {
                args.tokenizer_path = process_args.next().map(PathBuf::from).expect(USAGE_HELP);
            }
            Some(s) if s == "--temperature" => {
                args.temperature = process_args.next().expect(USAGE_HELP).parse().unwrap();
            }
            Some(s) if s == "--top-p" => {
                args.top_p = process_args.next().expect(USAGE_HELP).parse().unwrap();
            }
            Some(s) if s == "--steps" => {
                args.steps = process_args.next().expect(USAGE_HELP).parse().unwrap();
            }
            Some(s) if s == "--prompt" => {
                args.prompt = process_args.next().expect(USAGE_HELP);
            }
            Some(s) if s == "--rng-seed" => {
                args.rng_seed = process_args.next().expect(USAGE_HELP).parse().unwrap();
            }
            None => break,
            _ => panic!("{USAGE_HELP}"),
        }
    }

    if args.rng_seed == 0 {
        args.rng_seed = Instant::now().elapsed().as_millis() as u64;
    }
    if args.temperature < 0.0 {
        args.temperature = 0.0;
    }
    if !(0.0..1.0).contains(&args.top_p) {
        args.top_p = 0.9;
    }

    let mut transformer = Transformer::read_checkpoint(&args.check_point);
    let tokenizer = Tokenizer::new(&args.tokenizer_path, transformer.vocab_size());
    let mut sampler = Sampler::new(
        transformer.vocab_size(),
        args.temperature,
        args.top_p,
        args.rng_seed,
    );

    generate(
        &mut transformer,
        &tokenizer,
        &mut sampler,
        args.prompt,
        args.steps,
    );
}

const USAGE_HELP: &str = "\
Usage: cargo run <checkpoint> [OPTIONS]
Options:
     --tokenizer-path <string>
     --temperature <float>
     --top-p <float>
     --steps <int>
     --prompt <string>
     --rng-seed <int>
";

fn generate(
    transformer: &mut Transformer,
    tokenizer: &Tokenizer,
    sampler: &mut Sampler,
    prompt: String,
    steps: usize,
) {
    let prompt_tokens = tokenizer.encode(&prompt, true, false);
    println!("prompt_tokens: {prompt_tokens:?}");

    let mut start = None;

    let mut token = prompt_tokens[0];

    let steps = steps as upos;
    let mut pos: upos = 0;
    while pos < steps {
        let logits = transformer.forward(token, pos);
        // println!("logits: {logits:?}");

        let next = match prompt_tokens.get(pos as usize + 1) {
            Some(&tokid) => tokid,
            None => sampler.sample(logits),
        };
        pos += 1;

        if next == BOS {
            break;
        }

        let piece = tokenizer.decode(token, next);
        print!("{piece}");
        std::io::stdout().flush().unwrap();
        token = next;

        start.get_or_insert_with(|| Instant::now());
    }
    println!();

    if let Some(start) = start {
        if pos > 1 {
            let end = Instant::now();
            println!(
                "achieved tok/s: {}",
                (pos - 1) as f64 / (end - start).as_secs() as f64
            )
        }
    }
}

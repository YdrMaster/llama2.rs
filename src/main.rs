mod kernel;
mod sampler;
mod tokenizer;
mod transformer;

use core::panic;
use sampler::Sampler;
use std::{
    fs::canonicalize,
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use tokenizer::{Tokenizer, BOS};
use transformer::Transformer;

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
        args.rng_seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
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
    let mut text = String::with_capacity(1024);
    text.push_str(&prompt);

    let prompt_tokens = tokenizer.encode(&prompt, true, false);
    let (last, tokens) = prompt_tokens.split_last().unwrap();

    let start = Instant::now();

    // 一次性输入提示词的所有 token
    transformer.update(tokens, 0);
    // 一个一个输入提示词的 token 但不计算 output
    // for (i, &t) in tokens.iter().enumerate() {
    // transformer.update(&[t], i as _);
    // }
    // 一个一个输入提示词的 token 并计算 output
    // for (i, &t) in tokens.iter().enumerate() {
    // let _ = transformer.forward(t, i as _);
    // }

    let mid = Instant::now();

    let mut pos = tokens.len();
    let mut token = *last;
    while pos < steps {
        let logits = transformer.forward(token, pos as _);
        let next = sampler.sample(logits);
        pos += 1;

        if next == BOS {
            break;
        }

        text.push_str(tokenizer.decode(token, next));
        token = next;
    }

    let end = Instant::now();
    println!("init time: {:?}", mid - start);
    println!(
        "achieved tok/s: {}",
        pos as f64 / (end - start).as_secs_f64()
    );
    println!("{text}");
}

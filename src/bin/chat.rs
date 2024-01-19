use core::panic;
use llama2_rs::{Sampler, Tokenizer, Transformer, BOS, EOS};
use std::{
    fs::canonicalize,
    io::Write,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

#[allow(unused_imports)]
use llama2_rs::FsLogger;

fn main() {
    struct Args {
        check_point: PathBuf,
        tokenizer_path: PathBuf,
        temperature: f32,
        top_p: f32,
        steps: usize,
        system: String,
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
        system: String::new(),
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
            Some(s) if s == "--system" => {
                args.system = process_args.next().expect(USAGE_HELP);
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
    if args.system.ends_with(".txt") {
        let path = PathBuf::from(&args.system);
        if path.is_file() {
            args.system = std::fs::read_to_string(path).unwrap();
        }
    }

    let mut transformer = Transformer::read_checkpoint(&args.check_point);
    let tokenizer = Tokenizer::new(&args.tokenizer_path, transformer.vocab_size());
    let mut sampler = Sampler::new(
        transformer.vocab_size(),
        args.temperature,
        args.top_p,
        args.rng_seed,
    );

    chat(
        &mut transformer,
        &tokenizer,
        &mut sampler,
        args.system,
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
     --system <string>
     --rng-seed <int>
";

fn chat(
    transformer: &mut Transformer,
    tokenizer: &Tokenizer,
    sampler: &mut Sampler,
    system: String,
    steps: usize,
) {
    let mut logger = ();
    let system = format!(
        "\
<|system|>
{}</s>
",
        system.trim()
    );

    let system_tokens = tokenizer.encode(&system, true, false);
    transformer.update(&system_tokens, 0, &mut logger);

    let mut pos = system_tokens.len();
    loop {
        let mut user = String::new();
        print!("user: ");
        std::io::stdout().flush().unwrap();
        std::io::stdin().read_line(&mut user).unwrap();
        let addition = format!(
            "\
<|user|>
{}</s>
<|assistant|>
",
            user.trim()
        );
        let addition_tokens = tokenizer.encode(&addition, false, false);
        let (last, tokens) = addition_tokens.split_last().unwrap();
        transformer.update(&addition_tokens, pos as _, &mut logger);
        pos += tokens.len();

        print!("assistant: ");
        std::io::stdout().flush().unwrap();

        let mut token = *last;
        while pos < steps {
            let logits = transformer.forward(token, pos as _, &mut logger);
            let next = sampler.sample(logits);
            pos += 1;

            if next == BOS {
                return;
            }

            print!("{}", tokenizer.decode(token, next));
            std::io::stdout().flush().unwrap();

            if next == EOS {
                break;
            }

            token = next;
        }
        println!();
    }
}

use core::panic;
use llama2_rs::{BpeTokenizer, Sampler, Tokenizer, Transformer, BOS, EOS};
use std::{
    collections::VecDeque,
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

    let mut transformer: Transformer = Transformer::read_checkpoint(&args.check_point);
    let tokenizer = BpeTokenizer::new(&args.tokenizer_path, transformer.vocab_size());
    let mut sampler = Sampler::new(
        transformer.vocab_size(),
        args.temperature,
        args.top_p,
        args.rng_seed,
    );

    chat(&mut transformer, &tokenizer, &mut sampler, args.system);
}

const USAGE_HELP: &str = "\
Usage: cargo run <checkpoint> [OPTIONS]
Options:
     --tokenizer-path <string>
     --temperature <float>
     --top-p <float>
     --system <string>
     --rng-seed <int>
";

fn chat(
    transformer: &mut Transformer,
    tokenizer: &impl Tokenizer,
    sampler: &mut Sampler,
    system: String,
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

        print!("assistant: (pos = {pos}) ");
        std::io::stdout().flush().unwrap();

        let mut token = *last;
        let mut buffer = VecDeque::<char>::new();
        loop {
            let logits = transformer.forward(token, pos as _, &mut logger);
            pos += 1;

            match sampler.sample(logits) {
                BOS => return,
                EOS => {
                    token = EOS;
                    buffer.extend("</s>".chars());
                }
                next => {
                    let piece = tokenizer.decode(token, next);
                    token = next;
                    buffer.extend(piece.chars());
                }
            }

            {
                let mut eos = false;
                let words = buffer.iter().collect::<String>();
                let words = if let Some((words, _)) = words.split_once("</s") {
                    eos = true;
                    words
                } else {
                    &words
                };
                let words = if let Some((words, _)) = words.split_once("<|") {
                    eos = true;
                    words
                } else {
                    &words
                };
                if eos {
                    print!("{words} [end]");
                    std::io::stdout().flush().unwrap();
                    break;
                } else {
                    while buffer.len() > 3 {
                        print!("{}", buffer.pop_front().unwrap());
                        std::io::stdout().flush().unwrap();
                    }
                }
            }
        }
        println!();
    }
}

use llama2_rs::SafeTensors;
use std::{fs::File, io::Write};

fn main() {
    let mut args = std::env::args();

    let _ = args.next().unwrap();
    let config = File::open(args.next().unwrap()).unwrap();
    let safetensors = File::open(args.next().unwrap()).unwrap();
    let mut out_config = File::create(args.next().unwrap()).unwrap();
    let mut out = File::create(args.next().unwrap()).unwrap();

    let config = SafeTensors::cast_f32(config, safetensors, &mut out);
    out_config.write_all(config.as_bytes()).unwrap();
}

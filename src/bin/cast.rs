use llama2_rs::SafeTensors;
use std::{fs::File, io::Write};

fn main() {
    let mut args = std::env::args();
    let _ = args.next().unwrap();
    let config = File::open(args.next().unwrap()).unwrap();
    let safetensors = File::open(args.next().unwrap()).unwrap();
    let mut out_config = File::create(args.next().unwrap()).unwrap();
    let mut out = File::create(args.next().unwrap()).unwrap();
    let import = SafeTensors::new(config, safetensors);
    let export = import.export(&mut out);
    out_config.write_all(export.as_bytes()).unwrap();
}

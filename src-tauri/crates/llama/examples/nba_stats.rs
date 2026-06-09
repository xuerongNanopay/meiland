use std::{env, path::PathBuf};

use llama::llama::{config::LlamaConfig4, engine::LlamaEngine4};

fn main() {
    let model_path: PathBuf = env::args_os().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("usage: cargo run --example nba_stats -- <model.gguf>");
        std::process::exit(2);
    });

    let model_path = model_path.to_str().unwrap();
    let llama_config = LlamaConfig4::default();

    let llama_engine = LlamaEngine4::from_file(model_path, llama_config).unwrap();

    let llama_ctx = llama_engine.init_context4().unwrap();


    let llama_batch = llama_ctx.gen_batch();

    

}

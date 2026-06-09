use std::{env, path::PathBuf};

use llama::llama::{config::LlamaConfig4, engine::LlamaEngine4};
use llama_cpp_2::model::LlamaChatMessage;

fn main() {
    let model_path: PathBuf = env::args_os().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("usage: cargo run --example nba_stats -- <model.gguf>");
        std::process::exit(2);
    });

    let model_path = model_path.to_str().unwrap();
    let mut llama_config = LlamaConfig4::default();

    llama_config.enable_backend_log = false;

    let llama_engine = LlamaEngine4::from_file(model_path, llama_config).unwrap();

    let mut llama_ctx = llama_engine.init_context4().unwrap();


    let mut llama_batch = llama_ctx.gen_batch();


    let messages = vec![
        build_chat_message("system", "You are the professor NBA stats").unwrap(),
        build_chat_message("user", "list champion of nba final conference in most recent five years").unwrap(),
    ];

    // Convert messages into ChatMl.
    let formatted_prompt = llama_ctx.apply_template(&messages).unwrap();

    println!("Prompt: \n{formatted_prompt}");

    // Prompting phase.
    let mut tokens = llama_ctx.str_to_token(&formatted_prompt).unwrap();

    let end_idx = tokens.len() - 1;
    for (idx, token) in tokens.as_slice().iter().enumerate() {
        let require_logits = if idx == end_idx {
            true
        } else {
            false
        };
        llama_batch.add_token(token, idx as i32, 0, require_logits).unwrap();
    }

    llama_ctx.decode_batch(&mut llama_batch).unwrap();

    // Generating phase.
    let mut cur = llama_batch.size()-1;
    let max_token = 1024;
    let mut generated_text = String::new();

    while cur < max_token {
        let next_token = llama_ctx.sample(llama_batch.size() - 1);

        if llama_ctx.is_eog_token(next_token) {
            break;
        }

        let next_str = llama_ctx.token_to_string(next_token, false).unwrap();
        generated_text.push_str(&next_str);
        tokens.push(next_token);

        llama_batch.clear();
        llama_batch.add_token(&next_token, (tokens.len() - 1) as i32, 0, true).unwrap();

        cur += 1;

        llama_ctx.decode_batch(&mut llama_batch).unwrap();

    }

    println!("Result: {generated_text}")

}

fn build_chat_message(role: &str, content: &str) -> Result<LlamaChatMessage, String> {
    Ok(LlamaChatMessage::new(role.to_owned(), content.to_owned())
        .map_err(|e| format!("Llama Chat Message Error: {e}"))?)
}
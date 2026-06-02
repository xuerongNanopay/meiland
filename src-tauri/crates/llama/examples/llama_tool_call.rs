use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::openai::OpenAIChatTemplateParams;
use llama_cpp_2::sampling::LlamaSampler;
use serde_json::json;
use std::env;
use std::error::Error;
use std::io::{self, Write};
use std::num::NonZeroU32;

/// Usage: `cargo run -p llama --example llama_tool_call -- {GGUF_PATH}
fn main() -> Result<(), Box<dyn Error>> {
    let model_path = env::args()
        .nth(1)
        .expect("usage: cargo run --example prompt_llama_cpp -- <model.gguf>");
    let max_tokens = 1024;

    let mut backend = LlamaBackend::init()?;
    backend.void_logs();

    let model = LlamaModel::load_from_file(&backend, model_path, &LlamaModelParams::default())?;
    let context_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(512));
    let mut context = model.new_context(&backend, context_params)?;
    let template = model.chat_template(None)?;

    let tools_json = json!([
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Fetch current weather by city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": { "type": "string" }
                    },
                    "required": ["location"]
                }
            }
        }
    ])
    .to_string();

    let messages_json = json!([
        {
            "role": "system",
            "content": "You are a tool caller."
        },
        {
            "role": "user",
            "content": "Fetch the weather in Paris."
        }
    ])
    .to_string();

    let formatted_prompt = model.apply_chat_template_oaicompat(
        &template,
        &OpenAIChatTemplateParams {
            messages_json: &messages_json,
            tools_json: Some(&tools_json),
            tool_choice: Some("auto"),
            json_schema: None,
            grammar: None,
            reasoning_format: None,
            chat_template_kwargs: Some("{}"),
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: false,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: false,
        },
    )?;

    let prompt_tokens = model.str_to_token(&formatted_prompt.prompt, AddBos::Always)?;
    let mut batch = LlamaBatch::new(prompt_tokens.len(), 1);
    batch.add_sequence(&prompt_tokens, 0, false)?;
    context.decode(&mut batch)?;

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut sampler =
        LlamaSampler::chain_simple([LlamaSampler::temp(0.8), LlamaSampler::dist(1234)]);
    let mut position = prompt_tokens.len() as i32;
    let mut result = String::new();

    for _ in 0..max_tokens {
        let token = sampler.sample(&context, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }

        let piece = model.token_to_piece(token, &mut decoder, true, None)?;
        print!("{piece}");
        result.push_str(&piece);
        io::stdout().flush()?;

        if position >= context.n_ctx() as i32 {
            break;
        }

        batch.clear();
        batch.add(token, position, &[0], true)?;
        context.decode(&mut batch)?;
        position += 1;
    }

    println!();
    println!("\nResult:\n{result}");
    Ok(())
}

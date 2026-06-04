use actix_web::post;
#[warn(dead_code)]

use actix_web::{App, HttpResponse, HttpServer, Responder, get, web};
use llama_cpp_2::{context::params::LlamaContextParams, llama_backend::LlamaBackend, llama_batch::LlamaBatch, model::{LlamaChatTemplate, LlamaModel, params::LlamaModelParams}, openai::OpenAIChatTemplateParams, sampling::LlamaSampler};
use serde_json::Value;
use llama_cpp_2::model::AddBos;
use std::{env, fmt::format, path::PathBuf};
use std::num::NonZeroU32;

const HOST_NAME: &str = "127.0.0.1";
const PORT: u16 = 4444;

struct Llama {
    backend: LlamaBackend,
    model: LlamaModel,
    template: LlamaChatTemplate,
    model_path: PathBuf,
}

// aync fn chat_complete()

fn run_llama_complete(
    Llama { 
        backend, 
        model, 
        template, 
        model_path 
    }: &Llama, body: &str
) -> Result<String, String> {
    // 1. Extract fields from input or defaul.
    let request: Value = serde_json::from_str(body).map_err(|e| format!("invalid json: {e}"))?;

    let messages = request
        .get("messages")
        .ok_or_else(|| format!("missing messages"))?;

    if !messages.is_array() {
        return Err(format!("messages must be an array"));
    }

    let messages_json = messages.to_string();

    let temperature = request
        .get("temperature")
        .and_then(Value::as_f64)
        .unwrap_or(1.0) as f32;
    if temperature < 0.0 {
        return Err(format!("temperature must be between 0 and 1"));
    }

    // 2. Initial template to convert input to tokens.
    let params = OpenAIChatTemplateParams {
        messages_json: messages_json.as_ref(),
        tools_json: None,
        tool_choice: None,
        json_schema: None,
        grammar: None,
        reasoning_format: None,
        chat_template_kwargs: None,
        add_generation_prompt: true,
        use_jinja: true,
        parallel_tool_calls: false,
        enable_thinking: true,
        add_bos: false,
        add_eos: false,
        parse_tool_calls: false,

    };

    let tpl_result = model
        .apply_chat_template_oaicompat(template, &params)
        .map_err(|e| format!("Llama Template Error: {e}"))?;

    println!("Template Result: \n{}", tpl_result.prompt);

    let tokens = model
        .str_to_token(&tpl_result.prompt, AddBos::Always)
        .map_err(|e| format!("Llama Token Error: {e}"))?;


    // 3. Setup context window and batch size.

    // Context window cannot exceed what the model can reasnably support.
    let context_window = 4096u32;
    // Higher value will have fast prompt phase, but more memory.
    let batch_size: u32 = 4096u32; // use 512 in production.(for puppost of demo to avoid extra code)

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(context_window)) // set context window
        .with_n_batch(batch_size);

    // 4. Initial LLAMA context
    let mut llama_ctx = model.new_context(backend, ctx_params)
        .map_err(|e| format!("Llama Context Error: {e}"))?;

    // 5. Add tokens into batch
    let mut batch = LlamaBatch::new(batch_size as usize, 1);
    let end_token_index = tokens.len().saturating_sub(1);

    for (i, token) in tokens.iter().copied().enumerate() {
        let is_logits = end_token_index == i;

        // add token to batch one by one
        batch
            .add(token, i as i32, &[0], is_logits)
            .map_err(|e| format!("Llama Batch Error: {e}"))?;
    }

    // 6. Inference phase 1: promopting.
    llama_ctx.decode(&mut batch)
        .map_err(|e| format!("Llama Decode Error: {e}"))?;

    // 6. Inference phase 2: generating.
    let max_tokens = 1024i32;
    let mut cur = batch.n_tokens();
    let max_cur = cur + max_tokens;
    let mut generated_text = String::new();
    let mut completion_tokens = 0i32;
    let mut decoder = encoding_rs::UTF_8.new_decoder();

    // Create sample chain.
    let temperature = 0.7;
    let top_k = 40;
    let top_p = 0.95;
    let seed = 42;

    let mut sampler = LlamaSampler::chain_simple(vec![
        LlamaSampler::temp(temperature),
        LlamaSampler::top_k(top_k),
        LlamaSampler::top_p(top_p, 1),
        LlamaSampler::dist(seed),
    ]);

    let mut finish_reason = "stop";

    // Generating next token one by one.
    while cur < max_cur {
        let next_token = sampler.sample(&llama_ctx, batch.n_tokens()-1);

        if model.is_eog_token(next_token) {
            break;
        }

        //TODO: dynamic special token.
        let text = model.token_to_piece(next_token, &mut decoder, true, None)
            .map_err(|e| format!("Llama Token2String Error: {e}"))?;

        println!("token text: {}", text);
        generated_text.push_str(&text);
        completion_tokens += 1;

        //TODO: support additional stop token.

        batch.clear();
        batch.add(next_token, cur, &[0], true)
            .map_err(|e| format!("Llama Batch Error: {e}"))?;

        cur += 1;

        llama_ctx.decode(&mut batch)
            .map_err(|e| format!("Llama Decode Error: {e}"))?;
    }

    if cur >= max_cur {
        finish_reason = "length";
    }


    Err("Todo".to_owned())
}


/*
curl --location 'localhost:4444/chat_complete' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "dummy-model",
    "messages": [
      {
        "role": "system",
        "content": "You are an helpful assistant Tess"
      },
      {
        "role": "user",
        "content": "What is your name?"
      }
    ]
  }'
*/
async fn llama_complete(ctx: web::Data<Llama>, body: String) -> impl Responder {
    match run_llama_complete(&ctx, &body) {
        Ok(body) => HttpResponse::Ok().body(body),
        Err(error_message) => HttpResponse::BadRequest().body(error_message)
    }
}

#[get("/")]
async fn server_description() -> impl Responder {
    HttpResponse::Ok().content_type("application/json").body(format!(
        r#"{{
  "name": "Simple Ollama Server",
  "status": "ok",
  "endpoints": [
    {{
        endpoint: "/chat_complete",
        usage: "
            curl http://{HOST_NAME}:{PORT}
        "
    }}
  ]
}}"#),
    )
}

fn log_model_infos(model: &LlamaModel) {
    let train_ctx = model.n_ctx_train();
    println!("model trained context: {train_ctx}");
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let model_path: PathBuf = env::args_os().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("usage: cargo run --example simple_server -- <model.gguf>");
        std::process::exit(2);
    });

    let mut backend = LlamaBackend::init().map_err(|err| std::io::Error::other(err.to_string()))?;
    backend.void_logs();
    let params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &model_path, &params)
        .map_err(|err| std::io::Error::other(err.to_string()))?;
    let template = model.chat_template(None).map_err(|err| std::io::Error::other(err.to_string()))?;

    log_model_infos(&model);

    // Initial LLM.
    let llama = Llama {
        backend,
        model,
        template,
        model_path,
    };

    let ctx = web::Data::new(llama);

    println!("Help: `curl http://{HOST_NAME}:{PORT}`");

    HttpServer::new(move || {
        App::new()
            .app_data(ctx.clone())        
            .service(server_description)
            .route("/chat_complete", web::post().to(llama_complete))
    })
    .bind((HOST_NAME, PORT))?
    .run()
    .await
}

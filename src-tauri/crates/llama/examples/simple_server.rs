use actix_web::post;
#[warn(dead_code)]

use actix_web::{App, HttpResponse, HttpServer, Responder, get, web};
use llama_cpp_2::{llama_backend::LlamaBackend, model::{LlamaChatTemplate, LlamaModel, params::LlamaModelParams}, openai::OpenAIChatTemplateParams};
use serde_json::Value;
use llama_cpp_2::model::AddBos;
use std::{env, fmt::format, path::PathBuf};

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
        .map_err(|e| format!("Template Error: {e}"))?;

    println!("Template Result: \n{}", tpl_result.prompt);

    let tokens = model
        .str_to_token(&tpl_result.prompt, AddBos::Always)
        .map_err(|e| format!("Token Error: {e}"))?;


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

use actix_web::{App, HttpResponse, HttpServer, Responder, get, web};
use llama_cpp_2::{llama_backend::LlamaBackend, model::{LlamaChatTemplate, LlamaModel, params::LlamaModelParams}};
use std::{env, path::PathBuf};

struct Llama {
    backend: LlamaBackend,
    model: LlamaModel,
    template: LlamaChatTemplate,
    model_path: PathBuf,
}

// aync fn chat_complete()

fn run_llama_complete(llama: &Llama, body: &str) -> Result<String, String> {
    Err("Todo".to_owned())
}

#[get("/chat_complete")]
async fn llama_complete(ctx: web::Data<Llama>, body: String) -> impl Responder {
    match run_llama_complete(&ctx, &body) {
        Ok(body) => HttpResponse::Ok().body(body),
        Err(error_message) => HttpResponse::BadRequest().body(error_message)
    }
}

#[get("/")]
async fn server_description() -> impl Responder {
    HttpResponse::Ok().content_type("application/json").body(
        r#"{
  "name": "Simple Ollama Server",
  "status": "ok",
  "endpoints": [
    {
        endpoint: "/chat_complete",
        usage: "
            asdfads
            adsf
        "
    }
  ]
}"#,
    )
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let model_path: PathBuf = env::args_os().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("usage: cargo run --example simple_server -- <model.gguf>");
        std::process::exit(2);
    });

    let backend = LlamaBackend::init().map_err(|err| std::io::Error::other(err.to_string()))?;
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

    let hostname = "127.0.0.1";
    let port = 8080;

    println!("Help: `curl http://{hostname}:{port}`");

    HttpServer::new(move || {
        App::new()
            .app_data(ctx.clone())        
            .service(server_description)
            .service(llama_complete)
    })
    .bind((hostname, port))?
    .run()
    .await
}

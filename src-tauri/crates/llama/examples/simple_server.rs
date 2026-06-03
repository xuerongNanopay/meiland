use actix_web::{App, HttpResponse, HttpServer, Responder, get};
use llama_cpp_2::{llama_backend::LlamaBackend, model::{LlamaChatTemplate, LlamaModel}};
use std::{env, path::PathBuf};

struct Llama {
    backend: LlamaBackend,
    model: LlamaModel,
    template: LlamaChatTemplate,
    model_path: PathBuf,
}

// aync fn chat_complete()

#[get("/chat_complete")]
async fn llama_complete() -> impl Responder {
    HttpResponse::Ok().body("TODO: llama_complete")
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

    let hostname = "127.0.0.1";
    let port = 8080;

    println!("Help: `curl http://{hostname}:{port}`");

    HttpServer::new(|| {
        App::new()
            .service(server_description)
            .service(llama_complete)
    })
    .bind((hostname, port))?
    .run()
    .await
}

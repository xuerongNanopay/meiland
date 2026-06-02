use actix_web::{App, HttpResponse, HttpServer, Responder, get};

#[get("/")]
async fn server_description() -> impl Responder {
    HttpResponse::Ok().body("Simple Ollama Server")
}

#[get("/complete")]
async fn llama_complete() -> impl Responder {
    HttpResponse::Ok().body("TODO: llama_complete")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
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

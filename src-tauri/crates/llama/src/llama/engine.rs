use std::{marker::PhantomPinned, path::Path};

use llama_cpp_2::{context::LlamaContext, llama_backend::LlamaBackend, llama_batch::LlamaBatch, model::{LlamaModel, params::LlamaModelParams}};

struct LlamaEngine {
    bankend: LlamaBackend,
    model: LlamaModel,
}

struct LlamaSession<'a> {
    context: LlamaContext<'a>,
    batch: LlamaBatch<'static>
}

impl LlamaEngine {

    fn from_file(model_path: &str) -> Result<Self, String> {
        if Path::new(model_path).is_file() {
            return Err("invalid model path".to_owned());
        }
        Err("TODO".to_owned())
    }


    fn init_session<'engine>() -> Result<LlamaSession<'engine>, String> {
        Err("TODO".to_owned())
    }
}

fn llama_model_load_from_file(model_path: &str, config: LlamaModelParams) -> Result<LlamaModel, String> {
    if Path::new(model_path).is_file() {
        return Err("invalid model path".to_owned());
    }



    Err("TODO".to_owned())
}
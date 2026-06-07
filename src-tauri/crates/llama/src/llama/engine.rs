use std::{marker::PhantomPinned, path::Path};

use llama_cpp_2::{
    context::LlamaContext,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{LlamaModel, params::LlamaModelParams},
};

use crate::llama::config::LlamaConfig;

struct LlamaEngine {
    backend: LlamaBackend,
    model: LlamaModel,
}

struct LlamaSession<'model> {
    model: &'model LlamaModel,
    context: LlamaContext<'model>,
    batch: LlamaBatch<'static>,
}

impl LlamaEngine {
    fn from_file(model_path: &str, config: LlamaConfig) -> Result<Self, String> {
        if Path::new(model_path).is_file() {
            return Err("invalid model path".to_owned());
        }

        let mut backend =
            LlamaBackend::init().map_err(|err| format!("Llama Backend Error: {err}"))?;

        if !config.enable_backend_log {
            backend.void_logs();
        }

        let params = LlamaModelParams::default();

        let model = LlamaModel::load_from_file(&backend, &model_path, &params)
            .map_err(|err| format!("Llama Model Error: {err}"))?;

        Ok(Self { backend, model })
    }

    fn init_session<'engine>() -> Result<LlamaSession<'engine>, String> {
        Err("TODO".to_owned())
    }
}

impl<'model> LlamaSession<'model> {

}
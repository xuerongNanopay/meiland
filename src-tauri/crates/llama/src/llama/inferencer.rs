use std::{default, num::NonZeroU32, path::PathBuf};

use llama_cpp_2::{
    context::{LlamaContext, params::LlamaContextParams}, llama_backend::LlamaBackend, llama_batch::LlamaBatch, model::{LlamaModel, params::LlamaModelParams},
};

enum ModelType {
    Text,
    Mtmd,
}

#[derive(Debug, Clone)]
struct InferenceConfig {
    model_path: String,
    
    enable_llama_backend_log: bool,
}

#[derive(Debug, Clone)]
struct LLMModel {
    model_name: String,
    model_path: PathBuf,
    has_mtmd: bool,
}


struct InferenceTask {}

struct LlamaInferencer {
    // model_meta: LLMModel,
    config: InferenceConfig,

    // Lora addaptor
    llama_model: LlamaModel,
    llama_backend: LlamaBackend,
    // llama_vocab: LlamaVocab
    // llama_batch: LlamaBatch<'static>,
}

impl LlamaInferencer {
    
    fn new(config: InferenceConfig) -> Result<Self, String> {

        let mut llama_backend = LlamaBackend::init()
            .map_err(|e| format!("Init Llama Backend Error: {e}"))?;

        if !config.enable_llama_backend_log  {
            llama_backend.void_logs();
        }

        let model_params = LlamaModelParams::default();

        let model_path = PathBuf::from(&config.model_path);

        let llama_model = LlamaModel::load_from_file(
                &llama_backend, 
                model_path, 
                &model_params
            )
            .map_err(|e| format!("Init Llama Model Error: {e}"))?;
    
        Ok(Self {
            config,
            llama_backend,
            llama_model,
        })
    }

    fn gen_llama_context(&self) -> Result<Self, String> {
    
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(11)) // set context window
            .with_n_batch(11);
        let llama_context = self.llama_model.new_context(&self.llama_backend, ctx_params)
            .map_err(|e| format!("Init Llama Context Error: {e}"))?;

        Err("TODO".to_owned())
    }
}
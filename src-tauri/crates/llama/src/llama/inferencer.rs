use std::path::PathBuf;

use llama_cpp_2::{
    context::LlamaContext, llama_backend::LlamaBackend, llama_batch::LlamaBatch, model::LlamaModel,
};

enum ModelType {
    Text,
    Mtmd,
}

#[derive(Debug, Clone)]
struct LLMModel {
    model_name: String,
    model_path: PathBuf,
    has_mtmd: bool,
}


struct InferenceTask {}

struct LlamaInferencer {
    model_meta: LLMModel,

    llama_model: LlamaModel,
    llama_backend: LlamaBackend,
    llama_batch: LlamaBatch<'static>,
}

impl LlamaInferencer {

}
use std::path::PathBuf;

use llama_cpp_2::{
    context::{LlamaContext, params::LlamaContextParams},
    llama_backend::LlamaBackend,
    model::LlamaModel,
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

/**
 * Inference runtime
 *
 * owns model/runtime state
 * owns request/result queues
 * owns active slots
 * runs the inferece loop
 *
 * TODO:
 *  1. load model
 *  2. start loop
 *  3. terminate
 *
 *
 */
struct LlamaInference {
    llm_model: LLMModel,
}

impl LlamaInference {}

struct LLamaInternal {
    llama_model: LlamaModel,
    llama_backend: LlamaBackend,
}

impl LLamaInternal {
    fn new_context(
        &self,
        params: LlamaContextParams,
    ) -> Result<LlamaContext<'_>, llama_cpp_2::LlamaContextLoadError> {
        self.llama_model.new_context(&self.llama_backend, params)
    }
}

struct InferenceTask {
    id: i32,
    index: i32,
}

#[cfg(test)]
mod tests {
    // use super::*;
}
pub mod llama;

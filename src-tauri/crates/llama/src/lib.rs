use std::path::PathBuf;

use llama_cpp_2::{context::LlamaContext, llama_backend::LlamaBackend, model::LlamaModel};

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

impl LlamaInference {


}

struct LLamaInternal {
    llama_model: LlamaModel,
    llama_backend: LlamaBackend,
    llama_context: LlamaContext,

}

struct InferenceTask {
    id: i32,
    index: i32,
}

#[cfg(test)]
mod tests {
    // use super::*;
}

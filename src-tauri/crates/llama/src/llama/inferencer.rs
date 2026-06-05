use llama_cpp_2::{
    context::LlamaContext, llama_backend::LlamaBackend, llama_batch::LlamaBatch, model::LlamaModel,
};

struct InferenceTask {}

struct LlamaInferencer {
    llama_model: LlamaModel,
    llama_backend: LlamaBackend,
    llama_batch: LlamaBatch<'static>,
}

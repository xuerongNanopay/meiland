use std::marker::PhantomPinned;

use llama_cpp_2::{context::LlamaContext, llama_backend::LlamaBackend, llama_batch::LlamaBatch, model::LlamaModel};

struct LlamaEngine {
    bankend: LlamaBackend,
    model: LlamaModel,
}

struct LlamaSession<'a> {
    context: LlamaContext<'a>,
    batch: LlamaBatch<'static>
}

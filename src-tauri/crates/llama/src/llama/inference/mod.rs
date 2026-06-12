pub mod config;
mod embedder;
pub mod engine;
mod inferencer;
mod metric;
mod sampler;

struct LlamaCommonParams {
    context_window: i32,
}

impl Default for LlamaCommonParams {
    fn default() -> Self {
        Self {
            context_window: 0, // 0 == context the model was trained with
        }
    }
}

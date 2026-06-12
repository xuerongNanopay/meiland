pub mod config;
mod embedder;
pub mod engine;
mod inferencer;
mod metric;
mod sampler;

pub struct LlamaCommonParams4 {
    context_window: i32,
}

impl Default for LlamaCommonParams4 {
    fn default() -> Self {
        Self {
            context_window: 0, // 0 == context the model was trained with
        }
    }
}

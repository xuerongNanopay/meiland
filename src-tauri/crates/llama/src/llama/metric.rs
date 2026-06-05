#[derive(Debug, Default, Clone, Copy)]
struct LlamaMetrics {
    t_start: i64,

    n_prompt_tokens_processed_total: u64,
    t_prompt_processing_total: u64,
    n_tokens_predicted_total: u64,
    t_tokens_generation_total: u64,

    n_tokens_max: u64,

    n_prompt_tokens_processed: u64,
    t_prompt_processing: u64,

    n_tokens_predicted: u64,
    t_tokens_generation: u64,

    n_decode_total: u64,
    n_busy_slots_total: u64,
}

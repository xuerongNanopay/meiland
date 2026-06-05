use std::time::SystemTime;

#[derive(Debug, Clone, Copy)]
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

impl Default for LlamaMetrics {
    fn default() -> Self {
        let t_start = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        Self {
            t_start,
            n_prompt_tokens_processed_total: 0,
            t_prompt_processing_total: 0,
            n_tokens_predicted_total: 0,
            t_tokens_generation_total: 0,
            n_tokens_max: 0,
            n_prompt_tokens_processed: 0,
            t_prompt_processing: 0,
            n_tokens_predicted: 0,
            t_tokens_generation: 0,
            n_decode_total: 0,
            n_busy_slots_total: 0,
        }
    }
}

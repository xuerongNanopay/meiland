pub(crate) struct LlamaConfig {
    pub(crate) enable_backend_log: bool,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            enable_backend_log: true,
        }
    }
}

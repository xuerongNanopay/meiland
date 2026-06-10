pub struct LlamaConfig4 {
    pub enable_backend_log: bool,
}

impl Default for LlamaConfig4 {
    fn default() -> Self {
        Self {
            enable_backend_log: true,
        }
    }
}

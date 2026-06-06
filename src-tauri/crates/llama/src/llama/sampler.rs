#[derive(Debug, Clone, Copy)]
struct SamplerConfig {
    temperature: f64, // <= 0.0 to sample greedily,
    top_k: i32, // <= 0 to use vocab size
    top_p: f64, // 1.0 = disabled
    min_p: f64, // 0.0 = disabled
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
           temperature: 0.8,
           top_k: 40,
           top_p: 0.95,
           min_p: 0.05,
        }
    }
}

struct SamplerBuilder {
    
}

struct Sampler {

}
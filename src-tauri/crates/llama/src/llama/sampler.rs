use llama_cpp_2::sampling::LlamaSampler;

use crate::llama::sampler::SamplerType::{Dist, Temp, TopK, TopP};

#[derive(Debug, Clone, Copy)]
struct SamplerParams {
    seed: u32,
    min_keep: usize, // 0 = disabled, otherwise samplers should return at least min_keep tokens
    temp: f32,       // <= 0.0 to sample greedily,
    top_k: i32,      // <= 0 to use vocab size
    top_p: f32,      // 1.0 = disabled
    min_p: f32,      // 0.0 = disabled

    llama_sampler_no_perf: bool, // disable performance metrics
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self {
            seed: 64,
            min_keep: 0,
            temp: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            llama_sampler_no_perf: false,
        }
    }
}

struct SamplerBuilder {
    params: SamplerParams,
    sequence: SamplerSequence,
}

impl SamplerBuilder {
    fn new(params: SamplerParams, sequence: SamplerSequence) -> Self {
        Self { params, sequence }
    }

    fn build_sampler(&self) -> Result<Sampler, String> {
        Err("TODO".to_owned())
    }
}

struct SamplerSequence(Vec<SamplerType>);

impl SamplerSequence {
    fn init_chain_sampler(&self, params: &SamplerParams) -> Result<Sampler, String> {
        let mut llama_samplers = Vec::<LlamaSampler>::new();
        for sample_type in self.0.iter() {
            match sample_type {
                SamplerType::Temp => llama_samplers.push(init_temp_sampler(params.temp)),
                SamplerType::TopK => llama_samplers.push(init_top_k_sampler(params.top_k)),
                SamplerType::TopP => {
                    llama_samplers.push(init_top_p_sampler(params.top_p, params.min_keep))
                }
                SamplerType::MinP => {
                    llama_samplers.push(init_min_p_sampler(params.min_p, params.min_keep))
                }
                SamplerType::Dist => llama_samplers.push(init_dist_sampler(params.seed)),
                // _ => {
                //     return Err("Unsupport sample type".to_owned());
                // }
            }
        }
        Ok(Sampler {
            llama_sampler: init_chain_sampler(llama_samplers, params.llama_sampler_no_perf),
        })
    }
}

enum SamplerType {
    Temp,
    TopK,
    TopP,
    MinP,
    Dist,
}

#[must_use]
fn init_temp_sampler(temperature: f32) -> LlamaSampler {
    LlamaSampler::temp(temperature)
}

#[must_use]
fn init_top_k_sampler(k: i32) -> LlamaSampler {
    LlamaSampler::top_k(k)
}

#[must_use]
fn init_top_p_sampler(p: f32, min_keep: usize) -> LlamaSampler {
    LlamaSampler::top_p(p, min_keep)
}

#[must_use]
fn init_min_p_sampler(p: f32, min_keep: usize) -> LlamaSampler {
    LlamaSampler::min_p(p, min_keep)
}

#[must_use]
fn init_dist_sampler(seed: u32) -> LlamaSampler {
    LlamaSampler::dist(seed)
}

#[must_use]
fn init_chain_sampler(llama_samplers: Vec<LlamaSampler>, no_perf: bool) -> LlamaSampler {
    LlamaSampler::chain(llama_samplers, no_perf)
}

struct Sampler {
    llama_sampler: LlamaSampler,
}

impl Sampler {
    fn with_sampler(llama_sampler: LlamaSampler) -> Self {
        Self { llama_sampler }
    }
}

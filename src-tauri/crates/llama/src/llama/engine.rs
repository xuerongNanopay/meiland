use std::{marker::PhantomPinned, num::NonZeroU32, path::Path};

use encoding_rs::Decoder;
use llama_cpp_2::{
    context::{LlamaContext, params::LlamaContextParams},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel, params::LlamaModelParams},
    sampling::LlamaSampler,
    token::LlamaToken,
};

use crate::llama::config::LlamaConfig;

struct LlamaEngine4 {
    backend: LlamaBackend,
    model: LlamaModel,
}

struct SessionMeta {}

struct LlamaContext4<'model> {
    model: &'model LlamaModel,
    context: LlamaContext<'model>,
    batch: LlamaBatch<'static>,
    template: LlamaChatTemplate,
    sampler: LlamaSampler,
    decoder: Decoder,
}

impl LlamaEngine4 {
    fn from_file(model_path: &str, config: LlamaConfig) -> Result<Self, String> {
        if Path::new(model_path).is_file() {
            return Err("invalid model path".to_owned());
        }

        let mut backend =
            LlamaBackend::init().map_err(|err| format!("Llama Backend Error: {err}"))?;

        if !config.enable_backend_log {
            backend.void_logs();
        }

        let params = LlamaModelParams::default();

        let model = LlamaModel::load_from_file(&backend, &model_path, &params)
            .map_err(|err| format!("Llama Model Error: {err}"))?;

        Ok(Self { backend, model })
    }

    fn init_context4<'model>(&'model self) -> Result<LlamaContext4<'model>, String> {
        let template = self
            .model
            .chat_template(None)
            .map_err(|err| format!("Llama Chat Template Error: {err}"))?;

        //TODO: refactor the context config.
        // Context window cannot exceed what the model can reasnably support.
        let context_window = 4096u32;
        // Higher value will have fast prompt phase, but more memory.
        let batch_size: u32 = 4096u32; // use 512 in production.(for puppost of demo to avoid extra code)

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(context_window)) // set context window
            .with_n_batch(batch_size);

        let context = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| format!("Llama Context Error: {e}"))?;

        let batch = LlamaBatch::new(batch_size as usize, 1);

        let mut sampler = LlamaSampler::chain_simple(vec![LlamaSampler::temp(0.7)]);

        Ok(LlamaContext4 {
            model: &self.model,
            context,
            batch,
            template,
            sampler,
            decoder: encoding_rs::UTF_8.new_decoder(),
        })
    }
}

// while server running:
//     drain queued HTTP tasks
//     assign tasks to slots
//     update_slots()
//         build one shared batch from active slots
//         llama_decode(ctx, batch)
//         sample next token per slot
//         stream/finalize responses
//     wait for more tasks

/**
 * TODO:
 *  1. update_batch
 *  2.
 */
impl<'model> LlamaContext4<'model> {
    // fn chat(&self) -> Result<String, String> {
    //     let messages = vec![
    //         build_chat_message("system", "You are a helpful assistant.")?,
    //         build_chat_message("user", "What is python.")?,
    //     ];

    //     let prompt = self
    //         .model
    //         .apply_chat_template(&self.template, &messages, true)
    //         .map_err(|e| format!("Llama Template Error: {e}"))?;

    //     println!("Prompt: \n{}", prompt);

    //     let tokens = self
    //         .model
    //         .str_to_token(&prompt, AddBos::Always)
    //         .map_err(|e| format!("Llama Token Error: {e}"))?;

    //     // self.context.clear_kv_cache_seq(src, p0, p1)
    //     Ok(prompt)
    // }

    // fn clean_token(&mut self) {
    //     self.batch.clear();
    // }

    fn decode_batch(&mut self, batch: &mut LlamaBatch4) -> Result<(), String> {
        self.context
            .decode(&mut batch.inner)
            .map_err(|e| format!("Llama Decode Error: {e}"))?;

        Ok(())
    }

    fn sample(&mut self, batch_logit_idx: i32) -> LlamaToken {
        self.sampler.sample(&self.context, batch_logit_idx)
    }

    fn token_to_string(
        &mut self,
        llama_token: LlamaToken,
        special: bool,
    ) -> Result<String, String> {
        Ok(self
            .model
            .token_to_piece(llama_token, &mut self.decoder, special, None)
            .map_err(|e| format!("Llama Token2String Error: {e}"))?)
    }

    fn apply_template(&self, messages: &[LlamaChatMessage]) -> Result<String, String> {
        Ok(self
            .model
            .apply_chat_template(&self.template, &messages, true)
            .map_err(|e| format!("Llama Template Error: {e}"))?)
    }

    fn str_to_token(&self, prompt: &str) -> Result<Vec<LlamaToken>, String> {
        Ok(self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| format!("Llama Token Error: {e}"))?)
    }

    fn clear_seq_kv(&mut self, seq_id: u32, start: Option<u32>, end: Option<u32>) -> Result<(), String> {
        self.context.clear_kv_cache_seq(Some(seq_id), start, end)
            .map_err(|e| format!("Llama KV Clear Error: {e}"))?;
        Ok(())
    }

    fn clear_kv(&mut self) {
        self.context.clear_kv_cache();
    }

    fn gen_batch(&self) -> LlamaBatch4 {
        LlamaBatch4::new(4096, 16)
    }
}

fn build_chat_message(role: &str, content: &str) -> Result<LlamaChatMessage, String> {
    Ok(LlamaChatMessage::new(role.to_owned(), content.to_owned())
        .map_err(|e| format!("Llama Chat Message Error: {e}"))?)
}

struct LlamaBatch4 {
    capacity: i32,
    inner: LlamaBatch<'static>,
}

impl LlamaBatch4 {
    fn new(capacity: i32, no_seq_max: i32) -> Self {
        Self {
            capacity,
            inner: LlamaBatch::new(capacity as usize, no_seq_max),
        }
    }

    fn size(&self) -> i32 {
        self.inner.n_tokens()
    }

    fn capacity(&self) -> i32 {
        self.capacity
    }

    fn add_token(
        &mut self,
        token: LlamaToken,
        seq_pos: i32,
        seq_ids: &[i32],
        require_logits: bool,
    ) -> Result<i32, String> {
        let idx = self.inner.n_tokens();

        self.inner
            .add(token, seq_pos, seq_ids, require_logits)
            .map_err(|e| format!("Llama Batch Error: {e}"))?;

        Ok(idx)
    }

    fn clear(&mut self) {
        self.clear();
    }
}

use std::{num::NonZeroU32, path::Path};

use encoding_rs::Decoder;
use llama_cpp_2::{
    context::{LlamaContext, params::LlamaContextParams},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel, params::LlamaModelParams},
    mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams},
    sampling::LlamaSampler,
    token::{self, LlamaToken},
};

use crate::llama::inference::LlamaCommonParams4;

use super::config::LlamaConfig4;

pub struct LlamaEngine4 {
    backend: LlamaBackend,
    model: LlamaModel,
}

pub struct LlamaContextParams4 {
    context_size: i32,
    batch_size: i32,
    num_sequence: i32,
}

impl Default for LlamaContextParams4 {
    fn default() -> Self {
        Self {
            context_size: 0, // 0: use model context size.
            batch_size: 2048,
            num_sequence: 1,
        }
    }
}

pub struct LlamaContext4<'engine> {
    params: LlamaContextParams4,
    engine: &'engine LlamaEngine4,
    context: LlamaContext<'engine>,
    template: LlamaChatTemplate,
    sampler: LlamaSampler,
    mtmd_context: Option<MtmdContext>,
}

impl LlamaEngine4 {
    pub fn from_file(model_path: &str, config: LlamaConfig4) -> Result<Self, String> {
        if !Path::new(model_path).is_file() {
            return Err(format!("invalid model path: {model_path}"));
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

    pub fn init_context4<'engine>(
        &'engine self,
        params: LlamaContextParams4,
        mmproj_path: Option<&str>,
    ) -> Result<LlamaContext4<'engine>, String> {
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

        let temperature = 0.0;
        let seed = 42;

        let sampler = LlamaSampler::chain_simple(vec![
            LlamaSampler::temp(temperature),
            LlamaSampler::dist(seed),
        ]);

        let mtmd_context = if let Some(path) = mmproj_path {
            let mtmd_params = MtmdContextParams::default();
            Some(
                MtmdContext::init_from_file(&path, &self.model, &mtmd_params)
                    .map_err(|e| format!("Llama Mtmd Initial Error: {:#?}", e))?,
            )
        } else {
            None
        };

        Ok(LlamaContext4 {
            params,
            engine: self,
            context,
            template,
            sampler,
            mtmd_context,
        })
    }
}

impl<'engine> LlamaContext4<'engine> {
    pub fn decode_text(
        &mut self,
        messages: &[LlamaChatMessage],
        seq_id: i32,
        seq_pos: i32,
        last_logit: bool,
    ) -> Result<i32, String> {
        let prompt = self.apply_template(messages)?;

        let tokens = self.str_to_token(&prompt)?;
        let end_pos = seq_pos + tokens.len() as i32 - 1;

        let mut batch = self.gen_batch();

        let mut seq_pos = seq_pos;

        for chunk in tokens.chunks(self.params.batch_size as usize) {
            for token in chunk.iter() {
                let require_logit = if last_logit && seq_pos == end_pos {
                    true
                } else {
                    false
                };
                batch.add_token(token, seq_pos, seq_id, require_logit)?;
                seq_pos += 1;
            }

            self.decode_batch(&mut batch)?;
            batch.clear();
        }
        Ok(tokens.len() as i32)
    }

    pub fn decode_mtmd(&mut self, messages: &[LlamaChatMessage], bitmaps: &[&MtmdBitmap], seq_id: i32, seq_pos:i32, last_logit:bool) {

    }

    pub fn decode_batch(&mut self, batch: &mut LlamaBatch4) -> Result<(), String> {
        self.context
            .decode(&mut batch.inner)
            .map_err(|e| format!("Llama Decode Error: {e}"))?;

        Ok(())
    }

    pub fn sample(&mut self, batch_logit_idx: i32) -> LlamaToken {
        self.sampler.sample(&self.context, batch_logit_idx)
    }

    pub fn is_eog_token(&self, token: LlamaToken) -> bool {
        self.engine.model.is_eog_token(token)
    }

    pub fn token_to_string(
        &mut self,
        llama_token: LlamaToken,
        special: bool,
    ) -> Result<String, String> {
        Ok(self
            .engine
            .model
            .token_to_piece(
                llama_token,
                &mut encoding_rs::UTF_8.new_decoder(),
                special,
                None,
            )
            .map_err(|e| format!("Llama Token2String Error: {e}"))?)
    }

    pub fn apply_template(&self, messages: &[LlamaChatMessage]) -> Result<String, String> {
        Ok(self
            .engine
            .model
            .apply_chat_template_with_tools_oaicompat(&self.template, &messages, None, None, true)
            .map_err(|e| format!("Llama Template Error: {:#?}", e))?
            .prompt)
    }

    pub fn str_to_token(&self, prompt: &str) -> Result<Vec<LlamaToken>, String> {
        Ok(self
            .engine
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| format!("Llama Token Error: {e}"))?)
    }

    pub fn clear_seq_kv(
        &mut self,
        seq_id: u32,
        start: Option<u32>,
        end: Option<u32>,
    ) -> Result<(), String> {
        self.context
            .clear_kv_cache_seq(Some(seq_id), start, end)
            .map_err(|e| format!("Llama KV Clear Error: {e}"))?;
        Ok(())
    }

    pub fn clear_kv(&mut self) {
        self.context.clear_kv_cache();
    }

    pub fn gen_batch(&self) -> LlamaBatch4 {
        LlamaBatch4::new(self.params.batch_size, self.params.num_sequence)
    }

    pub fn support_mtmd(&self) -> bool {
        self.mtmd_context.is_some()
    }

    pub fn support_vision(&self) -> bool {
        if let Some(ctx) = self.mtmd_context.as_ref() {
            ctx.support_vision()
        } else {
            false
        }
    }

    pub fn support_audio(&self) -> bool {
        if let Some(ctx) = self.mtmd_context.as_ref() {
            ctx.support_audio()
        } else {
            false
        }
    }

    pub fn sequence_next_pos(&self, seq_id: i32) -> i32 {
        self.context.kv_cache_seq_pos_max(seq_id)
    }
}

pub struct LlamaBatch4 {
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

    pub fn size(&self) -> i32 {
        self.inner.n_tokens()
    }

    pub fn capacity(&self) -> i32 {
        self.capacity
    }

    pub fn add_token(
        &mut self,
        token: &LlamaToken,
        seq_pos: i32,
        seq_id: i32,
        require_logits: bool,
    ) -> Result<i32, String> {
        self.inner
            .add(token.clone(), seq_pos, &[seq_id], require_logits)
            .map_err(|e| format!("Llama Batch Error: {e}"))?;

        Ok(self.inner.n_tokens() - 1)
    }

    pub fn add_tokens(
        &mut self,
        tokens: &[LlamaToken],
        seq_offset: i32,
        seq_id: i32,
        end_with_logits: bool,
    ) -> Result<i32, String> {
        let batch_offset = self.inner.n_tokens();

        let end_idx = tokens.len() - 1;

        let mut ret: Option<i32> = None;

        for (idx, token) in tokens.iter().enumerate() {
            let require_logits = if end_with_logits == true && idx == end_idx {
                true
            } else {
                false
            };

            ret = Some(self.add_token(token, seq_offset + idx as i32, seq_id, require_logits)?);
        }

        Ok(ret.unwrap())
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaBatch4;
    use llama_cpp_2::token::LlamaToken;

    #[test]
    fn text_llama_batch4_size() {
        let seq_0_token_0 = LlamaToken::new(8964);
        let seq_0_token_1 = LlamaToken::new(8964);
        let seq_1_token_0 = LlamaToken::new(8964);
        let seq_1_token_1 = LlamaToken::new(8964);
        let seq_1_token_2 = LlamaToken::new(8964);

        let mut batch = LlamaBatch4::new(4, 1);
        assert_eq!(batch.size(), 0);

        let idx = batch.add_token(&seq_0_token_0, 0, 0, false).unwrap();
        assert_eq!(batch.size(), 1);
        assert_eq!(idx, 0);

        let idx = batch.add_token(&seq_0_token_1, 1, 0, false).unwrap();
        assert_eq!(batch.size(), 2);
        assert_eq!(idx, 1);

        batch.clear();
        assert_eq!(batch.size(), 0);

        let idx = batch.add_token(&seq_1_token_0, 0, 1, false).unwrap();
        assert_eq!(batch.size(), 1);
        assert_eq!(idx, 0);

        let idx = batch.add_token(&seq_1_token_1, 1, 1, false).unwrap();
        let idx = batch.add_token(&seq_1_token_2, 2, 1, false).unwrap();

        assert_eq!(batch.size(), 3);
        assert_eq!(idx, 2);

        batch.clear();
        assert_eq!(batch.size(), 0);
    }
}

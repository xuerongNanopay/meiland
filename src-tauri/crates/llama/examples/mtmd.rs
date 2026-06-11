use std::{env, io, num::NonZeroU32, path::PathBuf};

use encoding_rs::UTF_8;
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{LlamaChatMessage, LlamaModel, params::LlamaModelParams},
    mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText},
    sampling::LlamaSampler,
};

fn main() {
    let mut args = env::args_os().skip(1);

    let (Some(model_path), Some(pproj_path), Some(image_path)) =
        (args.next(), args.next(), args.next())
    else {
        eprintln!("usage: cargo run --example mtmd -- <model.gguf> <pproj.gguf> <image>");
        std::process::exit(2);
    };

    //Initial Llama model
    let mut backend = LlamaBackend::init().unwrap();
    backend.void_logs();

    let params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &model_path, &params).unwrap();

    let context_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(4096))
        .with_n_batch(512)
        .with_n_threads(4);

    let mut context = model.new_context(&backend, context_params).unwrap();
    println!("Model loaded successfully");

    // Initial Mtmd context.
    let mtmd_params = MtmdContextParams::default();
    let mtmd_ctx =
        MtmdContext::init_from_file(pproj_path.to_str().unwrap(), &model, &mtmd_params).unwrap();

    println!("**** Vision Support {}", mtmd_ctx.support_vision());
    println!("**** Vision audio {}", mtmd_ctx.support_audio());

    let model_path = PathBuf::from(model_path);
    let pproj_path = PathBuf::from(pproj_path);
    let image_path = PathBuf::from(image_path);

    println!("Model path: {}", model_path.display());
    println!("Projector path: {}", pproj_path.display());
    println!("Image path: {}", image_path.display());

    let marker = llama_cpp_2::mtmd::mtmd_default_marker();

    let prompt = format!("Describe the image in detail. Identify any visible text.\n{marker}");

    let message = LlamaChatMessage::new("user".to_string(), prompt).unwrap();

    let template = model.chat_template(None).unwrap();

    let template_result = model
        .apply_chat_template_with_tools_oaicompat(&template, &[message], None, None, true)
        .unwrap();

    println!("Template result: {}", template_result.prompt);

    // Load image.
    let image = MtmdBitmap::from_file(&mtmd_ctx, image_path.to_str().unwrap()).unwrap();

    let input = MtmdInputText {
        text: template_result.prompt,
        add_special: true,
        parse_special: true,
    };

    // tokenize content.
    let chunks = mtmd_ctx.tokenize(input, &[&image]).unwrap();

    println!("Chunk len: {}", chunks.len());
    println!("Chunk total position: {}", chunks.total_positions());
    println!("Chunk total tokens: {}", chunks.total_tokens());

    let mut n_past = chunks
        .eval_chunks(&mtmd_ctx, &context, 0, 0, 512i32, true)
        .unwrap();

    println!("n_path: {}", n_past);

    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);

    let mut batch = LlamaBatch::new(1, 1);
    let mut decoder = UTF_8.new_decoder();

    // 9. Generation loop.
    for _ in 0..256 {
        let token = sampler.sample(&context, -1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        let text = model
            .token_to_piece(token, &mut decoder, true, None)
            .unwrap();

        print!("{text}");

        // Decode the generated token to obtain the next logits.
        batch.clear();
        batch.add(token, n_past, &[0], true).unwrap();
        n_past += 1;

        context.decode(&mut batch).unwrap();
    }
}

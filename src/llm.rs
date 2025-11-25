use anyhow::{Error as E, Result};
use candle_core::quantized::gguf_file;
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::ModelWeights;
use hf_hub::api::tokio::Api;
use hf_hub::{Repo, RepoType};
use std::path::PathBuf;
use tokenizers::Tokenizer;

// --- 1. Model and Tokenizer Loading ---

pub struct ModelPaths {
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
}

pub async fn download_model(repo_id: &str, model_file: &str) -> Result<ModelPaths> {
    println!("Locating model: {} ({})", repo_id, model_file);
    let api = Api::new()?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

    // Get the model file (GGUF)
    let model_path = repo.get(model_file).await?;

    // Get the tokenizer, with fallback logic
    let tokenizer_path = match repo.get("tokenizer.json").await {
        Ok(path) => path,
        Err(_) => {
            println!("Tokenizer not found in model repo, fetching from a base repo...");
            let base_api = Api::new()?;
            let base_repo_id = if repo_id.to_lowercase().contains("mistral") {
                "mistralai/Mistral-7B-Instruct-v0.2"
            } else {
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            };
            println!("Using fallback tokenizer from: {}", base_repo_id);
            let base_repo = base_api.repo(Repo::new(base_repo_id.to_string(), RepoType::Model));
            base_repo.get("tokenizer.json").await?
        }
    };

    Ok(ModelPaths {
        model_path,
        tokenizer_path,
    })
}

// --- 2. Device Selection ---

pub fn select_device() -> Device {
    let device = if cuda_is_available() {
        match Device::new_cuda(0) {
            Ok(device) => device,
            Err(_) => {
                println!("Warning: CUDA device not found, falling back to CPU.");
                Device::Cpu
            }
        }
    } else if metal_is_available() {
        match Device::new_metal(0) {
            Ok(device) => device,
            Err(_) => {
                println!("Warning: Metal device not found, falling back to CPU.");
                Device::Cpu
            }
        }
    } else {
        Device::Cpu
    };
    println!("Using device: {:?}", device);
    device
}

// --- 3. The main Inference Pipeline ---

pub struct LlmPipeline {
    model: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
}

impl LlmPipeline {
    pub fn new(model_paths: ModelPaths, device: Device) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(model_paths.tokenizer_path).map_err(E::msg)?;
        
        let mut file = std::fs::File::open(&model_paths.model_path)?;
        let model_content = gguf_file::Content::read(&mut file)
            .map_err(|e| E::msg(format!("Failed to read GGUF: {}", e)))?;
        let model = ModelWeights::from_gguf(model_content, &mut file, &device)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn run<F: FnMut(String) -> Result<()>>(
        &mut self,
        prompt: &str,
        mut callback: F,
    ) -> Result<()> {
        let tokens = self.tokenizer.encode(prompt, true).map_err(E::msg)?;
        let pre_prompt_tokens = tokens.get_ids();

        const MAX_CONTEXT: usize = 4096;
        const GEN_RESERVE: usize = 512;
        const MAX_INPUT_TOKENS: usize = MAX_CONTEXT - GEN_RESERVE;
        const SYSTEM_PRESERVE: usize = 150;

        let mut all_tokens = if pre_prompt_tokens.len() > MAX_INPUT_TOKENS {
            // Truncate the middle
            let keep_tail = MAX_INPUT_TOKENS - SYSTEM_PRESERVE;
            let start = &pre_prompt_tokens[0..SYSTEM_PRESERVE];
            let end = &pre_prompt_tokens[pre_prompt_tokens.len() - keep_tail..];

            println!(
                "Warning: Input too long ({} tokens). Truncating to safe limit ({} tokens).",
                pre_prompt_tokens.len(),
                MAX_INPUT_TOKENS
            );

            [start, end].concat()
        } else {
            pre_prompt_tokens.to_vec()
        };

        let mut logits_processor = LogitsProcessor::new(299792458, Some(0.7), Some(0.9));
        let eos_token_id = self.tokenizer.token_to_id("</s>").unwrap_or(2);

        for index in 0..GEN_RESERVE {
            let context_size = if index > 0 { 1 } else { all_tokens.len() };
            let start_pos = all_tokens.len() - context_size;
            let input = Tensor::new(&all_tokens[start_pos..], &self.device)?.unsqueeze(0)?;

            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?;

            let logits = if logits.rank() == 2 {
                logits.get(logits.dim(0)? - 1)?
            } else {
                logits
            };

            let next_token = logits_processor.sample(&logits)?;

            if next_token == eos_token_id {
                break;
            }

            if let Some(t) = self.tokenizer.id_to_token(next_token) {
                if t.contains("</s>") || t.contains("<|user|>") || t.contains("<|system|>") {
                    break;
                }
                callback(t.to_string())?;
            }

            all_tokens.push(next_token);
        }

        Ok(())
    }
}

// --- 4. Prompt Building ---
pub fn build_prompt(log_text: &str, prompt_template: Option<String>) -> String {
    if let Some(template) = prompt_template {
        template.replace("{{LOG_TEXT}}", log_text)
    } else {
        format!(
            "<|system|>\n\
            You are a CLI log analysis expert. Your job is to explain errors concisely. \n\
            Analyze the following log output. Provide a summary of the error and a suggested fix.\n\
            Do NOT repeat the full log. Be brief. Use Markdown.</s>\n\
            <|user|>\n\
            {}\n\
            </s>\n\
            <|assistant|>\n",
            log_text
        )
    }
}

use anyhow::{Error as E, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::ModelWeights;
use hf_hub::api::tokio::Api;
use hf_hub::{Repo, RepoType};
use tokenizers::Tokenizer;

pub struct Inferencer {
    model: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
}

impl Inferencer {
    pub async fn load(repo_id: &str) -> Result<Self> {
        // Use TheBloke's TinyLlama as the safe default
        let actual_repo = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF";
        
        println!("Locating model: {} ...", actual_repo);
        let api = Api::new()?;
        let repo = api.repo(Repo::new(actual_repo.to_string(), RepoType::Model));

        // 1. Get the model file (GGUF)
        let model_path = repo.get("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf").await?;
        
        // 2. Get the tokenizer
        // TheBloke usually has tokenizer.json
        let tokenizer_path = match repo.get("tokenizer.json").await {
            Ok(path) => path,
            Err(_) => {
                println!("Tokenizer not found in GGUF repo, fetching from base repo...");
                let base_api = Api::new()?;
                let base_repo = base_api.repo(Repo::new("TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(), RepoType::Model));
                base_repo.get("tokenizer.json").await?
            }
        };
        
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;
        
        // 3. Load Model
        let device = Device::Cpu; 
        let mut file = std::fs::File::open(&model_path)?;
        
        let model = gguf_file::Content::read(&mut file)
            .map_err(|e| E::msg(format!("Failed to read GGUF: {}", e)))?;
            
        let model = ModelWeights::from_gguf(model, &mut file, &device)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn explain(&mut self, log_text: &str) -> Result<String> {
        // TinyLlama Chat Template
        let prompt = format!(
            "<|system|>\n\
            You are a CLI log analysis expert. Your job is to explain errors concisely. \n\
            Analyze the following log output. Provide a summary of the error and a suggested fix.\n\
            Do NOT repeat the full log. Be brief. Use Markdown.</s>\n\
            <|user|>\n\
            {}\n\
            </s>\n\
            <|assistant|>\n",
            log_text
        );

        let tokens = self.tokenizer.encode(prompt, true).map_err(E::msg)?;
        let pre_prompt_tokens = tokens.get_ids();

        let mut logits_processor = LogitsProcessor::new(299792458, Some(0.7), Some(0.9));
        let mut output_tokens = Vec::new();
        let mut all_tokens = pre_prompt_tokens.to_vec();
        
        // Detect EOS token ID usually 2 for Llama/TinyLlama
        let eos_token_id = self.tokenizer.token_to_id("</s>").unwrap_or(2);

        for index in 0..500 {
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
            all_tokens.push(next_token);
            output_tokens.push(next_token);

            // Strict stopping conditions
            if next_token == eos_token_id {
                break;
            }
            if let Some(t) = self.tokenizer.id_to_token(next_token) {
                // Check for user prompt start which indicates hallucinated continuation
                if t.contains("</s>") || t.contains("<|user|>") || t.contains("<|system|>") {
                    break;
                }
            }
        }

        let final_text = self.tokenizer.decode(&output_tokens, true).map_err(E::msg)?;
        // Clean up artifacts
        let final_text = final_text
            .replace("</s>", "")
            .replace("<|user|>", "")
            .replace("<|system|>", "")
            .trim()
            .to_string();
            
        Ok(final_text)
    }
}

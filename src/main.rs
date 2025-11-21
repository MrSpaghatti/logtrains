mod llm;

use anyhow::{Context, Result};
use clap::Parser;
use colored::Colorize;
use dialoguer::Confirm;
use std::io::{self, Read};
use std::path::PathBuf;

/// LogTrains: specialized AI log interpreter
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The log file to read (reads from stdin if not provided)
    #[arg(name = "FILE")]
    file: Option<PathBuf>,

    /// Force a redownload/check of the model
    #[arg(long)]
    update_model: bool,

    /// Use a specific model repo
    #[arg(long, default_value = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")]
    model_repo: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // 1. Input Handling
    let input_text = get_input(args.file.as_ref())?;
    if input_text.trim().is_empty() {
        eprintln!("{}", "Error: No input provided. Pipe logs or provide a filename.".red());
        std::process::exit(1);
    }

    // 2. Model Confirmation
    // We don't check if file exists manually because hf-hub handles it, 
    // but we can ask permission if we think it might be a fresh download. 
    // For a CLI, it's better to just warn "This might download 1GB".
    println!("{}", "LogTrains: Initializing... (First run requires ~1GB download)".yellow());

    // 3. Load Model & Run Inference
    // We move the loading into a spinner or just print status
    let mut engine = match llm::Inferencer::load(&args.model_repo).await {
        Ok(e) => e,
        Err(e) => {
            eprintln!("{} {}", "Failed to load model:".red(), e);
            eprintln!("Check your internet connection or model name.");
            std::process::exit(1);
        }
    };

    println!("{}", "LogTrains: Analyzing input...".cyan().bold());
    
    match engine.explain(&input_text) {
        Ok(explanation) => {
            println!("\n{}", "=== Explanation ===".green().bold());
            println!("{}", explanation);
            println!("{}", "===================".green().bold());
        },
        Err(e) => {
            eprintln!("{} {}", "Inference failed:".red(), e);
        }
    }

    Ok(())
}

fn get_input(file_path: Option<&PathBuf>) -> Result<String> {
    let mut buffer = String::new();
    if let Some(path) = file_path {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read file: {:?}", path))?;
        buffer = content;
    } else {
        // check if stdin is a tty, if it is, we might be waiting forever for user input which is confusing
        // but for now, we assume the user knows to pipe or type
        if atty::is(atty::Stream::Stdin) {
             println!("{}", "Listening on stdin... (Ctrl+D to finish)".yellow());
        }
        io::stdin().read_to_string(&mut buffer).context("Failed to read from stdin")?;
    }
    Ok(buffer)
}
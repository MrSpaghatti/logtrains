mod llm;

use anyhow::{Context, Result};
use colored::Colorize;
use serde::Deserialize;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// A specialized AI log interpreter for your terminal.
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about,
    long_about = "LogTrains is a command-line tool that uses a local large language model
to analyze and explain log files or the output of other commands.

It can be used in three ways:
1. By passing a file path: `logtrains /path/to/your.log`
2. By piping from stdin: `cargo build | logtrains`
3. By executing a command directly: `logtrains --run \"npm install\"`
4. By analyzing the last command's output: `logtrains --last` (requires setup)

To enable the '--last' feature, run `logtrains setup` and follow the instructions.

The tool can be configured via a file at `~/.config/logtrains/config.toml`.
Example config.toml:
    model_repo = \"TheBloke/CodeLlama-7B-Instruct-GGUF\"
    model_file = \"codellama-7b-instruct.Q4_K_M.gguf\"
    prompt_file = \"/path/to/my/prompt.txt\"
"
)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Analyze a log file, piped input, or command output.
    Analyze(AnalyzeArgs),
    /// Print the shell script to enable the '--last' feature.
    Setup,
}

#[derive(Parser, Debug)]
struct AnalyzeArgs {
    /// The log file to read. If not provided, reads from stdin.
    #[arg(name = "log_file", conflicts_with_all = &["run", "last"])]
    file: Option<PathBuf>,

    /// Execute a command, stream its output, and analyze the result.
    #[arg(long, conflicts_with_all = &["log_file", "last"])]
    run: Option<String>,

    /// Analyze the output of the last command (requires setup).
    #[arg(long, conflicts_with_all = &["log_file", "run"])]
    last: bool,

    /// Force a redownload/check of the model weights.
    #[arg(long)]
    update_model: bool,

    /// The HuggingFace repository ID for the model.
    #[arg(long)]
    model_repo: Option<String>,

    /// The specific model file (GGUF) to use from the repository.
    #[arg(long)]
    model_file: Option<String>,

    /// Path to a custom prompt template file.
    #[arg(long)]
    prompt_file: Option<PathBuf>,
}

#[derive(Deserialize, Debug, Default)]
struct Config {
    model_repo: Option<String>,
    model_file: Option<String>,
    prompt_file: Option<PathBuf>,
}

impl Config {
    fn load() -> Result<Self> {
        if let Some(config_dir) = dirs::config_dir() {
            let config_path = config_dir.join("logtrains/config.toml");
            if config_path.exists() {
                let config_str = std::fs::read_to_string(config_path)?;
                let config: Config = toml::from_str(&config_str)?;
                return Ok(config);
            }
        }
        Ok(Config::default())
    }
}

const MAX_INPUT_CHARS: usize = 12_000;

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Commands::Analyze(analyze_args) => {
            let config = Config::load()?;

            // Layer the configuration: CLI args > config file > defaults
            let model_repo = analyze_args.model_repo.or(config.model_repo).unwrap_or_else(|| "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".to_string());
            let model_file = analyze_args.model_file.or(config.model_file).unwrap_or_else(|| "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf".to_string());
            let prompt_file = analyze_args.prompt_file.or(config.prompt_file);

            // 1. Input Handling
            let mut input_text = if analyze_args.last {
                let log_file = if let Some(cache_dir) = dirs::cache_dir() {
                    cache_dir.join("logtrains").join("last.log")
                } else {
                    return Err(anyhow::anyhow!("Could not determine cache directory."));
                };
                std::fs::read_to_string(log_file)?
            } else if let Some(command) = analyze_args.run {
                println!("Running command: {}", command.cyan());

                let reader = duct::cmd("sh", ["-c", &command])
                    .stderr_to_stdout()
                    .reader()?;

                let mut output = String::new();
                let mut line = String::new();
                let mut reader = BufReader::new(reader);

                while let Ok(bytes_read) = reader.read_line(&mut line) {
                    if bytes_read == 0 {
                        break;
                    }
                    print!("{}", line);
                    output.push_str(&line);
                    line.clear();
                }

                output
            } else {
                get_input(analyze_args.file.as_ref())?
            };

            if input_text.trim().is_empty() {
                eprintln!("{}", "Error: No input provided. Pipe logs, provide a filename, or use --run.".red());
                std::process::exit(1);
            }

            if input_text.len() > MAX_INPUT_CHARS {
                eprintln!(
                    "{}",
                    format!(
                        "Warning: Input truncated to last {} characters.",
                        MAX_INPUT_CHARS
                    )
                    .yellow()
                );
                let start = input_text.len() - MAX_INPUT_CHARS;
                input_text = input_text[start..].to_string();
            }

            // 2. Model Confirmation
            println!("{}", "LogTrains: Initializing... (First run requires ~1GB download)".yellow());

            // 3. Load Model & Run Inference
            let mut engine = match llm::Inferencer::load(&model_repo, &model_file).await {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("{} {}", "Failed to load model:".red(), e);
                    eprintln!("Check your internet connection or model name.");
                    std::process::exit(1);
                }
            };

            let prompt_template = if let Some(path) = prompt_file {
                Some(std::fs::read_to_string(path)?)
            } else {
                None
            };

            println!("{}", "LogTrains: Analyzing input...".cyan().bold());
            println!("\n{}", "=== Explanation ===".green().bold());

            let res = engine.explain(&input_text, prompt_template, |token| {
                print!("{}", token);
                io::stdout().flush()?;
                Ok(())
            });

            println!("\n{}", "===================".green().bold());

            if let Err(e) = res {
                eprintln!("{} {}", "Inference failed:".red(), e);
            }
        }
        Commands::Setup => {
            let shell = std::env::var("SHELL").unwrap_or_else(|_| "bash".to_string());
            let shell_name = std::path::Path::new(&shell)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("bash");

            let script = match shell_name {
                "bash" | "zsh" => {
                    let log_dir = if let Some(cache_dir) = dirs::cache_dir() {
                        cache_dir.join("logtrains")
                    } else {
                        return Err(anyhow::anyhow!("Could not determine cache directory."));
                    };
                    std::fs::create_dir_all(&log_dir)?;
                    let log_file = log_dir.join("last.log");

                    format!(
                        r#"
# LogTrains Setup Script for {shell}
# Add the following function to your ~/.{shell}rc file:

logtrains-run() {{
    script -q -c "$@" "{log_file}"
}}

# Now you can run a command and analyze it like this:
# logtrains-run npm install
# logtrains analyze --last
"#,
                        shell = shell_name,
                        log_file = log_file.display()
                    )
                }
                _ => {
                    format!("Unsupported shell: {}. Please open an issue on GitHub to request support.", shell_name)
                }
            };

            println!("{}", script);
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
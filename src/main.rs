mod llm;

use anyhow::{Context, Result};
use colored::Colorize;
use serde::Deserialize;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::time::UNIX_EPOCH;

use clap::{Parser, Subcommand};

/// A specialized AI log interpreter for your terminal.
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about,
    long_about = r#"LogTrains is a command-line tool that uses a local large language model
to analyze and explain log files or the output of other commands.

It can be used in three ways:
1. By passing a file path: `logtrains /path/to/your.log`
2. By piping from stdin: `cargo build | logtrains`
3. By executing a command directly: `logtrains --run "npm install"`
4. By analyzing previous commands' output: `logtrains analyze --last [N]` (requires setup)

To enable history, run `logtrains setup` and follow the instructions.

The tool can be configured via a file at `~/.config/logtrains/config.toml`.
Example config.toml:
    model_repo = "TheBloke/CodeLlama-7B-Instruct-GGUF"
    model_file = "codellama-7b-instruct.Q4_K_M.gguf"
    prompt = """
You are a {{ROLE}}.
Your task is to analyze the following log output:
{{LOG_TEXT}}
"""
"#
)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Analyze a log file, piped input, or command output.
    Analyze(AnalyzeArgs),
    /// Print the shell script to enable command history.
    Setup,
    /// List the history of recorded commands.
    History,
}

#[derive(Parser, Debug)]
struct AnalyzeArgs {
    /// The log file to read. If not provided, reads from stdin.
    #[arg(name = "log_file", conflicts_with_all = &["run", "last"])]
    file: Option<PathBuf>,

    /// Execute a command, stream its output, and analyze the result.
    #[arg(long, conflicts_with_all = &["log_file", "last"])]
    run: Option<String>,

    /// Analyze the output of the last N commands (requires setup).
    /// If N is not given, it defaults to 1.
    #[arg(long, conflicts_with_all = &["log_file", "run"], num_args=0..=1, default_missing_value="1")]
    last: Option<usize>,

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

    /// Model size preset to use (overridden by --model-repo).
    #[arg(long, value_enum, default_value = "medium")]
    preset: Preset,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum Preset {
    /// TinyLlama 1.1B (~600MB) - Fast, lower quality
    Tiny,
    /// Mistral 7B (~4.1GB) - Balanced, high quality
    Medium,
}

#[derive(Deserialize, Debug, Default)]
struct Config {
    model_repo: Option<String>,
    model_file: Option<String>,
    prompt_file: Option<PathBuf>,
    prompt: Option<String>,
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

            // Determine model based on preset or overrides
            let (default_repo, default_file) = match analyze_args.preset {
                Preset::Tiny => (
                    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                    "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                ),
                Preset::Medium => (
                    "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                    "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                ),
            };

            // Layer the configuration: CLI args > config file > defaults (from preset)
            let model_repo = analyze_args
                .model_repo
                .or(config.model_repo)
                .unwrap_or_else(|| default_repo.to_string());
            let model_file = analyze_args
                .model_file
                .or(config.model_file)
                .unwrap_or_else(|| default_file.to_string());
            let prompt_file = analyze_args.prompt_file.or(config.prompt_file);
            let prompt_template = config.prompt;

            // 1. Input Handling
            let mut input_text = if let Some(n) = analyze_args.last {
                let log_dir = if let Some(cache_dir) = dirs::cache_dir() {
                    cache_dir.join("logtrains")
                } else {
                    return Err(anyhow::anyhow!("Could not determine cache directory."));
                };

                let files = get_sorted_log_files(&log_dir)?;
                if files.is_empty() {
                    return Err(anyhow::anyhow!("No recorded logs found. Run 'logtrains setup' to enable recording."));
                }

                if n == 0 || n > files.len() {
                    return Err(anyhow::anyhow!("Invalid history count. Available logs: {}", files.len()));
                }

                // Take the last n files (which are the first n in the sorted list)
                // Since files are sorted newest first, we take the range 0..n
                // However, we want to present them in chronological order to the LLM
                let mut selected_files = files[0..n].to_vec();
                selected_files.reverse(); // Now oldest to newest

                let mut combined_input = String::new();
                for log_file in selected_files {
                    let filename = log_file.file_name().unwrap().to_string_lossy();
                    // Parse command slug from filename: log_{timestamp}_{slug}.log
                    let cmd_slug = filename.split('_').skip(2).collect::<Vec<_>>().join("_").replace(".log", "");

                    println!("Reading log file: {}", filename.cyan());
                    combined_input.push_str(&format!("\n=== Command: {} ===\n", cmd_slug));
                    combined_input.push_str(&std::fs::read_to_string(log_file)?);
                    combined_input.push('\n');
                }
                combined_input
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

            input_text = truncate_input(input_text, MAX_INPUT_CHARS);

            // 2. Model Loading
            println!(
                "{}",
                format!(
                    "LogTrains: Initializing... (Model: {}). First run may require a large download.",
                    model_file
                )
                .yellow()
            );

            // Using the new Builder from the refactored llm.rs (HEAD)
            let mut engine = match llm::ModelLoaderBuilder::new(&model_repo, &model_file).load().await {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("{} {}", "Failed to load model:".red(), e);
                    eprintln!("Check your internet connection or model name.");
                    std::process::exit(1);
                }
            };

            // 3. Prompt Construction & Inference
            let final_prompt_template = if let Some(path) = prompt_file {
                Some(std::fs::read_to_string(path)?)
            } else {
                prompt_template
            };
            
            println!("{}", "LogTrains: Analyzing input...".cyan().bold());
            println!("\n{}", "=== Explanation ===".green().bold());

            let res = engine.explain(&input_text, final_prompt_template, |token| {
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

            match shell_name {
                "bash" | "zsh" => {
                    let log_dir = if let Some(cache_dir) = dirs::cache_dir() {
                        cache_dir.join("logtrains")
                    } else {
                        return Err(anyhow::anyhow!("Could not determine cache directory."));
                    };
                    std::fs::create_dir_all(&log_dir)?;

                    let script_cmd = match std::env::consts::OS {
                        "macos" => r###"script -q "$logfile" "$@""###,
                        "linux" => r###"script -q -c "$@" "$logfile""###,
                        _ => "echo 'Unsupported OS'",
                    };

                    println!(
                        r#"# LogTrains Setup Script for {shell}
# Add the following function to your ~/.{shell}rc or ~/.zshrc file:

logtrains-run() {{ 
    # Configuration
    # You can override these in your environment
    local max_files=${{LOGTRAINS_MAX_FILES:-50}}
    local exclude_cmds="${{LOGTRAINS_EXCLUDE:-cd ls pwd clear exit history}}"
    local log_dir="{log_dir}"

    # Check exclusion
    local cmd="$1"
    # Check if cmd is in the space-separated list
    if [[ " $exclude_cmds " == *" $cmd "* ]] || [[ -z "$cmd" ]]; then
        "$@"
        return $?
    fi

    # Create directory if it doesn't exist
    mkdir -p "$log_dir"

    local timestamp=$(date +%s)
    # Sanitize command for filename: replace non-alphanumeric with _, truncate to 30 chars
    local cmd_slug=$(echo "$@" | sed 's/[^a-zA-Z0-9]/_/g' | cut -c 1-30)
    # If cmd_slug is empty, use 'unknown'
    [ -z "$cmd_slug" ] && cmd_slug="unknown"

    local logfile="$log_dir/log_${{timestamp}}_${{cmd_slug}}.log"

    # Execute and record
    {script_cmd}
    local ret=$?

    # Cleanup: Delete excess files
    # List files sorted by name (oldest first because of timestamp prefix), count them
    local files=$(ls -1 "$log_dir"/log_*.log 2>/dev/null)
    local count=$(echo "$files" | grep -c "log_")

    if [ "$count" -gt "$max_files" ]; then
        local num_delete=$((count - max_files))
        # Delete the oldest $num_delete files
        echo "$files" | head -n "$num_delete" | xargs rm -f
    fi

    return $ret
}}

# Usage:
# logtrains-run npm install
# logtrains analyze --last      # Analyze the most recent command
# logtrains analyze --last 2    # Analyze the 2nd most recent command
# logtrains history             # See list of recorded commands
"#,
                        shell = shell_name,
                        log_dir = log_dir.display(),
                        script_cmd = script_cmd
                    );
                }
                _ => {
                    eprintln!("Unsupported shell: {}. Please open an issue on GitHub to request support.", shell_name);
                }
            };
        }
        Commands::History => {
            let log_dir = if let Some(cache_dir) = dirs::cache_dir() {
                cache_dir.join("logtrains")
            } else {
                return Err(anyhow::anyhow!("Could not determine cache directory."));
            };

            let files = get_sorted_log_files(&log_dir)?;
            if files.is_empty() {
                println!("No command history found.");
                return Ok(());
            }

            println!("{:<5} | {:<20} | {}", "Index", "Time", "File/Command");
            println!("{}", "-".repeat(60));

            for (i, file) in files.iter().enumerate() {
                let filename = file.file_name().unwrap().to_string_lossy();

                // Try to parse timestamp from filename: log_{timestamp}_{slug}.log
                let timestamp_str = filename.split('_').nth(1).unwrap_or("0");
                let time_display = if let Ok(ts) = timestamp_str.parse::<u64>() {
                     let d = UNIX_EPOCH + std::time::Duration::from_secs(ts);
                     let datetime: chrono::DateTime<chrono::Local> = d.into();
                     datetime.format("%Y-%m-%d %H:%M:%S").to_string()
                } else {
                    "Unknown Time".to_string()
                };

                println!("{:<5} | {:<20} | {}", i + 1, time_display, filename);
            }
        }
    }

    Ok(())
}

fn get_sorted_log_files(log_dir: &std::path::Path) -> Result<Vec<PathBuf>> {
    if !log_dir.exists() {
        return Ok(vec![]);
    }

    let mut files: Vec<PathBuf> = std::fs::read_dir(log_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .collect();

    // Sort by filename (which includes timestamp), newest first (descending)
    files.sort_by(|a, b| {
        let name_a = a.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let name_b = b.file_name().and_then(|s| s.to_str()).unwrap_or("");
        name_b.cmp(name_a)
    });

    Ok(files)
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

fn truncate_input(input: String, max_chars: usize) -> String {
    if input.len() > max_chars {
        eprintln!(
            "{}",
            format!(
                "Warning: Input truncated to last {} characters.",
                max_chars
            )
            .yellow()
        );
        let start = input.len() - max_chars;
        input[start..].to_string()
    } else {
        input
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_input_no_truncation() {
        let input = "hello world".to_string();
        let truncated = truncate_input(input.clone(), 20);
        assert_eq!(truncated, input);
    }

    #[test]
    fn test_truncate_input_with_truncation() {
        let input = "hello world".to_string();
        let truncated = truncate_input(input.clone(), 5);
        assert_eq!(truncated, "world");
    }

    #[test]
    fn test_truncate_input_zero_max_chars() {
        let input = "hello world".to_string();
        let truncated = truncate_input(input.clone(), 0);
        assert_eq!(truncated, "");
    }
}

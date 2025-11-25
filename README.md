# LogTrains 🚂

LogTrains is a specialized CLI tool that uses local Large Language Models (LLMs) to analyze log files, explain errors, and suggest fixes. It runs entirely on your machine using the `candle` ML framework.

## Features

- **Local Inference:** Uses local models (defaulting to `Mistral-7B-Instruct`) running on your CPU or GPU. No API keys or data egress required.
- **Flexible Input:** Accepts logs via standard input (stdin) or file path.
- **Command History:** Can record the output of commands and analyze them later.
- **Concise Analysis:** specialized system prompt to avoid verbose LLM chatter and focus on the error.

## Installation

Ensure you have Rust installed.

```bash
cargo install --path .
```

## Usage

### Basic Usage

Read from a file:
```bash
logtrains analyze /path/to/your.log
```

Pipe from another command:
```bash
cat /var/log/syslog | logtrains analyze
```

### Command History and Analysis

LogTrains can record the output of your commands so you can analyze them later.

**1. Setup**

First, set up the `logtrains-run` shell function. Run the following command and add the output to your `.bashrc` or `.zshrc` file.

```bash
logtrains setup
```

**2. Record a command**

Now, you can run commands with `logtrains-run` to record their output.

```bash
logtrains-run npm install
```

**3. List History**

Use the `history` subcommand to see a list of recorded commands.

```bash
logtrains history
```

**4. Analyze Previous Commands**

Use `analyze --last [N]` to analyze the output of the last `N` commands. If `N` is not provided, it defaults to 1.

```bash
# Analyze the last command
logtrains analyze --last

# Analyze the last 3 commands
logtrains analyze --last 3
```

### First Run

On the first run, LogTrains will download the model weights (default is ~4.1GB) from Hugging Face. These are cached locally in your huggingface cache directory.

## Technical Details

- **Model Presets:**
    - `tiny`: [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) (~600MB)
    - `medium` (default): [TheBloke/Mistral-7B-Instruct-v0.2-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) (~4.1GB)
- **Engine:** [Candle](https://github.com/huggingface/candle) (Rust-native ML framework)
- **Quantization:** 4-bit (Q4_K_M) for efficient CPU inference.

## Development

```bash
# Run directly
cargo run -- analyze src/main.rs
```

## License

[MIT](LICENSE)

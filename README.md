# LogTrains 🚂

LogTrains is a specialized CLI tool that uses local Large Language Models (LLMs) to analyze log files, explain errors, and suggest fixes. It runs entirely on your machine using the `candle` ML framework.

## Features

- **Local Inference:** Uses `TinyLlama-1.1B` (via GGUF) running locally on your CPU. No API keys or data egress required.
- **Flexible Input:** Accepts logs via standard input (stdin) or file path.
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
logtrains target/input.log
```

Pipe from another command:
```bash
cat /var/log/syslog | logtrains
```

### First Run

On the first run, LogTrains will download the model weights (~600MB - 1GB) from Hugging Face. These are cached locally in your huggingface cache directory.

## Technical Details

- **Model:** [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)
- **Engine:** [Candle](https://github.com/huggingface/candle) (Rust-native ML framework)
- **Quantization:** 4-bit (Q4_K_M) for efficient CPU inference.

## Development

```bash
# Run directly
cargo run -- src/main.rs
```

## License

[MIT](LICENSE)

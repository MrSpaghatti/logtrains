# Development Plan

## 1. Status Update
**Date:** November 25, 2025
**Action:** Merged `feature-setup-and-llm-refactor` and `feature-enhance-log-analysis` into `main`.

Significant progress has been made. The application now supports cross-platform setup (Linux/macOS), has a cleaner internal architecture for LLM handling, and includes advanced features like model presets and log history management.

## 2. Completed Features

| Feature | Description | Status |
| :--- | :--- | :--- |
| **Cross-Platform Setup** | `logtrains setup` now detects the OS and generates correct `script` syntax for Linux and macOS. | ✅ Completed |
| **LLM Refactoring** | `src/llm.rs` has been refactored into `ModelLoaderBuilder` and `Inferencer`, improving separation of concerns. | ✅ Completed |
| **Model Presets** | Users can choose between `tiny` (TinyLlama) and `medium` (Mistral) models via the `--preset` flag. | ✅ Completed |
| **Smart History** | Setup script now excludes common commands (cd, ls) and auto-cleans old logs to save space. | ✅ Completed |

## 3. Current Architecture & Known Issues

### A. Input Handling (Limitations)
- `MAX_INPUT_CHARS` is hardcoded to 12,000.
- **Issue:** Large logs are blindly truncated at the start (keeping the tail).
- **Plan:** Implement smarter windowing (e.g., keep head + tail, or searchable windows).

### B. User Experience / Ambiguity
- The `--last N` flag currently **concatenates** the last N logs into a single prompt.
- **Risk:** Users might expect `--last 2` to analyze the *2nd to last* command, not the *last 2 commands combined*.
- **Plan:** Introduce interactive selection or clearer flags.

### C. Testing
- While `truncate_input` has tests, complex logic like `get_sorted_log_files` and the new `ModelLoader` lacking comprehensive unit/integration tests.

## 4. Next Steps (Prioritized)

1.  **Add Unit Tests**: Focus on the file system logic (`get_sorted_log_files`) and the new configuration merging logic in `main.rs`.
2.  **Smart Truncation**: Replace the hardcoded character limit with a token-aware or "head+tail" truncation strategy to preserve context.
3.  **Interactive Selection (TUI)**: Implement a feature to interactively select a log from history using `ratatui` or `dialoguer`, rather than relying on index guessing with `--last N`.

## 5. Brainstorming: Future Additions

-   **Daemon Mode**: A background watcher that automatically logs specific commands or errors without manual wrapping.
-   **Conversation Persistence**: Allow the LLM to remember the context of the *previous* explanation for follow-up questions.
-   **Syntax Highlighting**: Use `syntect` to pretty-print the code blocks in the LLM's response.
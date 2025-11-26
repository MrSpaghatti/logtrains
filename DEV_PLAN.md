# Development Plan

## 1. Status Update
**Date:** November 25, 2025
**Action:** Merged `origin/enhance-rewind-feature` into `main`.

The codebase has been unified. We now have a robust feature set including cross-platform setup, advanced LLM handling (presets, smart truncation), and history management.

## 2. Completed Features

| Feature | Description | Status |
| :--- | :--- | :--- |
| **Cross-Platform Setup** | `logtrains setup` now detects the OS and generates correct `script` syntax for Linux and macOS. | ✅ Completed |
| **LLM Refactoring** | `src/llm.rs` has been refactored into `ModelLoaderBuilder` and `Inferencer`. | ✅ Completed |
| **Model Presets** | Users can choose between `tiny` (TinyLlama) and `medium` (Mistral) models via the `--preset` flag. | ✅ Completed |
| **Smart History** | Setup script excludes common commands and auto-cleans old logs. | ✅ Completed |
| **Smart Truncation** | Input is now truncated based on token count (4096 limit), preserving the system prompt (head) and the most recent logs (tail). | ✅ Completed |

## 3. Current Architecture & Known Issues

### A. User Experience / Ambiguity
- The `--last N` flag currently **concatenates** the last N logs into a single prompt.
- **Risk:** Users might expect `--last 2` to analyze the *2nd to last* command, not the *last 2 commands combined*.
- **Plan:** Introduce interactive selection or clearer flags.

### B. Testing
- `truncate_input` has tests.
- `src/llm.rs` token truncation logic needs tests.
- `get_sorted_log_files` and config merging need tests.

## 4. Next Steps (Prioritized)

1.  **Add Unit Tests**: 
    - Test `src/llm.rs` truncation logic (mocking the tokenizer if possible, or separating the logic).
    - Test `get_sorted_log_files`.
2.  **Interactive Selection (TUI)**: Implement a feature to interactively select a log from history using `ratatui` or `dialoguer`.
3.  **Refine --last N behavior**: Consider changing default to "select Nth" or "list last N", or keep as is but add TUI.

## 5. Brainstorming: Future Additions

-   **Daemon Mode**: A background watcher that automatically logs specific commands or errors without manual wrapping.
-   **Conversation Persistence**: Allow the LLM to remember the context of the *previous* explanation for follow-up questions.
-   **Syntax Highlighting**: Use `syntect` to pretty-print the code blocks in the LLM's response.

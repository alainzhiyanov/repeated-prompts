"""Shared constants and helpers for the repeated-prompts project."""

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "checkpoints/qwen-1.5b-double-prompt-multi/final"


def make_messages(
    system_prompt: str, content: str, answer: str | None = None
) -> list[dict]:
    """Build a chat-message list (system + user + optional assistant)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages

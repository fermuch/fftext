"""Shared LLM infrastructure: model load, streaming, input clipping.

Everything here is used by more than one task module (summarize, explain,
check, translate). Per-task prompts and pipelines live in their own files.
"""

import os
import re
import sys
import time
import ctypes
from typing import Iterable

import llama_cpp
from llama_cpp import Llama


# Silence llama.cpp's C-level log output (the stuff that bypasses Python's
# `verbose=False` and goes straight to stderr via printf). This kills the
# "n_ctx_seq (4096) < n_ctx_train (262144)" warning and similar nags.
# Trade-off: we lose llama.cpp's own error messages too, but Python-level
# exceptions still propagate fine.
@ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
def _silence_llama_log(level, text, user_data):  # noqa: ARG001
    pass


llama_cpp.llama_log_set(_silence_llama_log, ctypes.c_void_p(0))


# --- model + threading ------------------------------------------------------

# os.cpu_count() returns logical cores; halving is a decent x86 heuristic.
# Override with QWEN_THREADS if you know your physical core count.
_LOGICAL = os.cpu_count() or 4
N_THREADS = int(os.environ.get("QWEN_THREADS", max(1, _LOGICAL // 2)))

# Context budget. n_ctx is fixed at load time; we keep input under
# INPUT_CHAR_BUDGET so prompt + generation + chat template fit comfortably.
# Rough rule: ~4 chars per token. 10000 chars -> ~2500 input tokens; leave
# ~500 for output + a couple hundred for template/system -> 4096 ctx fits.
# Note: per-token generation cost scales with *filled* context, not n_ctx,
# so bumping n_ctx alone is nearly free (a bit more KV-cache RAM); what
# costs you is actually filling it via bigger inputs.
N_CTX = 4096
INPUT_CHAR_BUDGET = 10000


def load_model() -> Llama:
    return Llama.from_pretrained(
        repo_id="unsloth/Qwen3.5-0.8B-GGUF",
        filename="Qwen3.5-0.8B-Q4_K_M.gguf",
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_batch=512,
        use_mmap=True,
        use_mlock=False,
        verbose=False,
    )


# --- input clipping ---------------------------------------------------------

def clip(text: str, budget: int = INPUT_CHAR_BUDGET) -> tuple[str, bool]:
    """Head+tail clip so we keep document structure (intro + conclusion)
    when something is too long. Cheaper than running the tokenizer twice."""
    if len(text) <= budget:
        return text, False
    head = budget * 2 // 3
    tail = budget - head - 32  # 32 chars for the marker
    return f"{text[:head]}\n...[truncated]...\n{text[-tail:]}", True


# --- generation -------------------------------------------------------------

# Stop sequences guard against the model trying to continue the "conversation"
# with a fake user turn, which small models do constantly.
STOP = ["\n\nUser:", "\n\nQ:", "<|im_end|>", "<|endoftext|>"]


def stream_chat(
    llm: Llama,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.3,
    verbose: bool = False,
) -> str:
    """Stream a chat completion to stdout, return the full text.
    Streaming matters on CPU because perceived latency >> total latency."""
    t0 = time.perf_counter()
    chunks: Iterable = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=STOP,
        stream=True,
    )

    pieces: list[str] = []
    n_tokens = 0  # approximate: counts chunks, not true tokens
    for chunk in chunks:
        delta = chunk["choices"][0].get("delta", {})
        piece = delta.get("content")
        if not piece:
            continue
        pieces.append(piece)
        n_tokens += 1
        print(piece, end="", flush=True)
    print()

    if verbose:
        elapsed = time.perf_counter() - t0
        rate = n_tokens / elapsed if elapsed > 0 else 0.0
        # n_tokens counts chunks; usually ~1 token/chunk but treat as approx.
        print(
            f"[~{n_tokens} chunks in {elapsed:.2f}s = ~{rate:.1f} chunks/s]",
            file=sys.stderr,
        )

    return "".join(pieces)


# --- thinking-mode helper ---------------------------------------------------

# Qwen3.5 family models can emit <think>...</think> blocks containing chain-
# of-thought before the actual answer. Used by check.py on stages that
# benefit from reasoning (ranking, synthesis, evaluation); other tasks
# typically don't enable thinking but the stripping logic is shared.

_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def strip_thinking(text: str) -> str:
    """Remove any <think>...</think> blocks from a model response.

    Also handles unclosed/malformed thinking tags by stripping any text
    before a closing </think> if a stray closing tag appears without an
    opener (small models occasionally emit truncated thinking)."""
    text = _THINK_RE.sub("", text)
    # If a stray </think> remains (no opening tag), drop everything before it.
    idx = text.lower().find("</think>")
    if idx != -1:
        text = text[idx + len("</think>"):]
    return text.strip()


# --- legacy demo modes (kept here because they use stream_chat directly) ---

def demo_oneshot(llm: Llama, prompt: str, verbose: bool) -> None:
    stream_chat(
        llm,
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user",   "content": prompt}],
        max_tokens=512,
        temperature=0.7,
        verbose=verbose,
    )


def demo_interactive(llm: Llama, verbose: bool) -> None:
    history = [{"role": "system", "content": "You are a helpful assistant."}]
    print("Chat with Qwen3.5-0.8B. Ctrl-C or empty line to quit.\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            break
        history.append({"role": "user", "content": user})
        print("qwen> ", end="", flush=True)
        reply = stream_chat(llm, history, max_tokens=512,
                            temperature=0.7, verbose=verbose)
        history.append({"role": "assistant", "content": reply})
        print()

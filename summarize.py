"""Summarize task: produce a tight 3-bullet summary of the input."""

import sys

from llama_cpp import Llama

from llm import INPUT_CHAR_BUDGET, clip, stream_chat


# Keep system prompts short. A 0.8B model gets confused by long instructions
# and burns tokens copying them. Brevity is also faster prompt eval.
SUMMARIZE_SYS = (
    "You summarize text. Output exactly 3 short bullet points. "
    "No preamble, no closing remarks. Be concrete and specific."
)


def task_summarize(llm: Llama, text: str, source: str, verbose: bool) -> None:
    clipped, was_clipped = clip(text)
    if was_clipped:
        print(f"[note: input clipped to ~{INPUT_CHAR_BUDGET} chars to fit context]",
              file=sys.stderr)
    user = f"Summarize the following ({source}):\n\n{clipped}"
    stream_chat(
        llm,
        [{"role": "system", "content": SUMMARIZE_SYS},
         {"role": "user",   "content": user}],
        max_tokens=200,
        verbose=verbose,
    )

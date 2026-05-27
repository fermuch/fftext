"""Explain task: ELI5-style plain-language explanation."""

import sys

from llama_cpp import Llama

from llm import INPUT_CHAR_BUDGET, clip, stream_chat


EXPLAIN_SYS = (
    "You explain things simply, like to a curious 10-year-old. "
    "Use short sentences and everyday words. No jargon. "
    "Aim for 4-6 sentences. No preamble."
)


def task_explain(llm: Llama, text: str, source: str, verbose: bool) -> None:
    clipped, was_clipped = clip(text)
    if was_clipped:
        print(f"[note: input clipped to ~{INPUT_CHAR_BUDGET} chars to fit context]",
              file=sys.stderr)
    user = f"Explain this so a kid would get it ({source}):\n\n{clipped}"
    stream_chat(
        llm,
        [{"role": "system", "content": EXPLAIN_SYS},
         {"role": "user",   "content": user}],
        max_tokens=180,
        verbose=verbose,
    )

"""Translate task: render the input text into a target language.

The target language description is passed in verbatim from the CLI flag
(e.g. --lang "casual Spanish", --lang "Japanese, polite register").
That string is dropped straight into the system prompt, which gives the
user fine control over register/dialect without expanding the CLI.
"""

import sys

from llama_cpp import Llama

from llm import INPUT_CHAR_BUDGET, clip, stream_chat


# The {lang} slot is filled by the CLI's --lang argument. We don't sanitize
# it — the user is also the operator here, so prompt-injection isn't a
# threat model. Long descriptions are fine; the model handles them.
TRANSLATE_SYS_TMPL = (
    "You translate text into {lang}. "
    "Output ONLY the translation. No preamble, no notes, no original text, "
    "no transliteration unless the target language requests it. "
    "Preserve paragraph breaks and any markdown/code formatting from the "
    "source. If a proper noun has no standard translation, keep it as-is."
)


def task_translate(llm: Llama, text: str, source: str, lang: str,
                   verbose: bool) -> None:
    # Default to English when --lang isn't supplied. The CLI passes "" in
    # that case; normalize here so callers don't need to know the default.
    if not lang.strip():
        lang = "English"

    clipped, was_clipped = clip(text)
    if was_clipped:
        print(f"[note: input clipped to ~{INPUT_CHAR_BUDGET} chars to fit context]",
              file=sys.stderr)

    system = TRANSLATE_SYS_TMPL.format(lang=lang.strip())
    user = f"Translate the following ({source}) into {lang.strip()}:\n\n{clipped}"
    # Translation wants more output tokens than summarization — output is
    # roughly the same length as input, not a compression. Cap so we don't
    # run away on a long input that already used most of the context.
    stream_chat(
        llm,
        [{"role": "system", "content": system},
         {"role": "user",   "content": user}],
        max_tokens=1024,
        # Slightly lower than chat defaults: translation should be faithful,
        # not creative. But not zero — small models loop at temp 0.
        temperature=0.3,
        verbose=verbose,
    )

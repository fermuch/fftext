"""Fact-check task: extract verifiable claims, web-search each, judge them.

Pipeline per run:
  1. extract_claims      — LLM emits a JSON array of claims
  2. rank_claims         — LLM picks top-N most fact-checkable (cuts cost)
  3. rewrite_query       — LLM turns each claim into a keyword query
  4. search_with_fallback — Mojeek/Startpage with rotation + fallback
  5. summarize_evidence  — LLM compresses snippets, drops irrelevant ones
  6. synthesize_evidence — LLM lays out support/contradict/gaps
  7. evaluate_verdict    — deterministic shortcut or LLM picks final label

Per-claim cost is ~4 LLM calls + 1 search round-trip, hence the ranker.
"""

import re
import sys
import time
import json
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup

from llama_cpp import Llama

from llm import INPUT_CHAR_BUDGET, clip, strip_thinking


# --- search-side knobs ------------------------------------------------------

# Scraping is fragile by nature: HTML class names move, captchas appear, and
# both providers ban "obvious bot" UAs. We pick a generic desktop UA, keep
# timeouts tight so a single slow query can't stall the whole run, and treat
# any unexpected page shape as an empty result rather than crashing.

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_HEADERS = {"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"}
_SEARCH_TIMEOUT = 8  # seconds per request
_MAX_CLAIMS = 12
_RANK_TO = 3  # ranker keeps this many claims; rest discarded before search
_TOP_K = 3  # snippets per claim into the judge

# Path used by --debug to dump the first empty SERP from each backend so we
# can tell whether we're getting captchas, rate-limited, or just hitting a
# selector regression.
_DEBUG_DUMP_DIR = Path("/tmp/fftext-debug")
_dumped: set[str] = set()  # track which backends have already dumped


def _dump_html(backend: str, html: str, query: str) -> None:
    """One dump per backend per process; called on the first empty result."""
    if backend in _dumped:
        return
    _dumped.add(backend)
    try:
        _DEBUG_DUMP_DIR.mkdir(parents=True, exist_ok=True)
        p = _DEBUG_DUMP_DIR / f"{backend}-empty.html"
        p.write_text(f"<!-- query: {query} -->\n{html}", encoding="utf-8")
        print(f"[debug: dumped empty {backend} SERP to {p}]", file=sys.stderr)
    except OSError:
        pass


# --- search backends -------------------------------------------------------
#
# DDG and Brave were dropped after both started returning captchas / 429s
# from the test IP. SearXNG was dropped after two runs in a row showed
# most public instances either DNS-dead or returning empty pages.
# Current stack:
#   - Mojeek: independent UK index, not derived from Google/Bing
#   - Startpage: proxies Google with a different bot-detection surface
#
# Rotation: alternate primary by claim index, fall through to the other
# on empty. Jitter sleep per call avoids burst-looking scrape attacks.
# Selectors are permissive — sites change DOM regularly and we'd rather
# get fuzzy hits than zero.


def _looks_like_external_url(href: str) -> bool:
    """Filter out same-host nav links and javascript: stubs."""
    return (href.startswith("http://") or href.startswith("https://")) \
        and "/about" not in href and "/preferences" not in href


def search_mojeek(query: str, k: int = _TOP_K,
                  debug: bool = False) -> list[tuple[str, str, str]]:
    """Mojeek HTML SERP — independent UK index."""
    try:
        r = requests.get(
            "https://www.mojeek.com/search",
            params={"q": query},
            headers=_HEADERS,
            timeout=_SEARCH_TIMEOUT,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        if debug:
            print(f"[debug: mojeek failed: {e}]", file=sys.stderr)
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    out: list[tuple[str, str, str]] = []
    # Mojeek wraps each result in <li> under <ul class="results-standard">.
    # Permissive: any list item with an outbound anchor.
    for res in (soup.select("ul.results-standard > li")
                or soup.select("ol.results > li")
                or soup.select("li.result")):
        a = res.find("a", href=True, class_=re.compile("^(?!.*ob-)"))
        if not a:
            a = res.find("a", href=True)
        if not a:
            continue
        url = a["href"]
        if not _looks_like_external_url(url):
            continue
        # Title is usually <h2><a>...</a></h2>
        title_el = res.find(["h2", "h3"])
        title = (title_el.get_text(" ", strip=True) if title_el
                 else a.get_text(" ", strip=True))
        snip_el = res.find("p", class_=re.compile("(s|desc|snippet)", re.I)) \
            or res.find("p")
        snippet = snip_el.get_text(" ", strip=True) if snip_el else ""
        if title and url:
            out.append((title, url, snippet))
        if len(out) >= k:
            break
    if not out and debug:
        _dump_html("mojeek", r.text, query)
    return out


def search_startpage(query: str, k: int = _TOP_K,
                     debug: bool = False) -> list[tuple[str, str, str]]:
    """Startpage HTML SERP — proxies Google but with different bot surface."""
    try:
        r = requests.get(
            "https://www.startpage.com/sp/search",
            params={"query": query, "cat": "web"},
            headers=_HEADERS,
            timeout=_SEARCH_TIMEOUT,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        if debug:
            print(f"[debug: startpage failed: {e}]", file=sys.stderr)
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    out: list[tuple[str, str, str]] = []
    # Startpage wraps results in <section class="w-gl__result"> or div.result
    for res in (soup.select("section.w-gl__result")
                or soup.select("div.w-gl__result")
                or soup.select("div.result")):
        a = res.find("a", class_=re.compile("result-link|w-gl__result", re.I),
                     href=True)
        if not a:
            a = res.find("a", href=True)
        if not a:
            continue
        url = a["href"]
        if not _looks_like_external_url(url):
            continue
        title_el = res.find(class_=re.compile("title", re.I)) \
            or res.find(["h2", "h3"])
        title = (title_el.get_text(" ", strip=True) if title_el
                 else a.get_text(" ", strip=True))
        snip_el = res.find(class_=re.compile("description|snippet|desc", re.I))
        snippet = snip_el.get_text(" ", strip=True) if snip_el else ""
        if title and url:
            out.append((title, url, snippet))
        if len(out) >= k:
            break
    if not out and debug:
        _dump_html("startpage", r.text, query)
    return out


# Rotation order — claim N starts at _BACKENDS[N % 2], falls through on empty.
_BACKENDS = [
    ("mojeek", search_mojeek),
    ("startpage", search_startpage),
]


def _sanitize_query(query: str) -> str:
    """Strip characters that trigger WAFs on search providers.

    Mojeek returned 403 on queries containing '$' (URL-encoded as %24).
    Other punctuation that occasionally trips bot filters: backticks,
    pipes, raw quotes. Spaces and alphanumerics are obviously fine;
    quoted phrases (\"foo bar\") are kept since they help search recall.
    Result is normalized to single spaces and capped at 180 chars.
    """
    # Drop characters that providers' WAFs treat as injection signals.
    q = re.sub(r"[\$`|\\<>{}]", " ", query)
    # Collapse whitespace.
    q = re.sub(r"\s+", " ", q).strip()
    if len(q) > 180:
        q = q[:180].rsplit(" ", 1)[0]
    return q


def search_with_fallback(
    query: str, primary_idx: int, k: int = _TOP_K,
    debug: bool = False,
) -> tuple[list[tuple[str, str, str]], str]:
    """Try primary, fall through to others on empty. Tiny jitter sleep
    avoids thread-pool bursts looking like scrape attacks. Returns
    (results, name_of_backend_that_answered)."""
    query = _sanitize_query(query)
    n = len(_BACKENDS)
    order = [(primary_idx + i) % n for i in range(n)]
    names: list[str] = []
    for i in order:
        name, fn = _BACKENDS[i]
        names.append(name)
        time.sleep(random.uniform(0.2, 0.7))
        results = fn(query, k, debug=debug)
        if results:
            if debug and len(names) > 1:
                print(f"[debug: fell back to {name} after "
                      f"{'+'.join(names[:-1])} empty for {query!r}]",
                      file=sys.stderr)
            return results, name
    return [], "+".join(names)


def _heuristic_keywords(claim: str, max_chars: int = 120) -> str:
    """Last-resort fallback when the LLM keyword rewrite fails.

    Strip stopwords and obvious filler, keep proper-noun-shaped tokens and
    numbers. Real search engines weight rare tokens; sending a whole
    sentence dilutes them with 'the', 'is', 'a', etc.
    """
    _STOP = {
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
        "for", "from", "has", "have", "he", "her", "him", "his", "i", "in",
        "is", "it", "its", "of", "on", "or", "she", "that", "the", "their",
        "them", "they", "this", "to", "was", "were", "will", "with", "who",
        "which", "what", "where", "when", "how", "why", "but", "not", "no",
        "do", "does", "did", "would", "could", "should", "can", "may",
        "might", "you", "your", "we", "us", "our", "i", "me", "my", "such",
        "into", "out", "over", "than", "then", "there", "these", "those",
        "some", "any", "all", "more", "most", "other", "another", "also",
        "according",
    }
    cleaned = re.sub(r"[^\w\s'\-]", " ", claim)
    words = [w for w in cleaned.split() if w]
    keep: list[str] = []
    for w in words:
        wl = w.lower()
        if wl in _STOP:
            continue
        # Keep capitalized words (likely proper nouns), numbers, and any
        # word >= 4 chars (drops most remaining filler verbs/articles).
        if w[0].isupper() or any(c.isdigit() for c in w) or len(w) >= 4:
            keep.append(w)
    q = " ".join(keep[:8])  # cap at 8 tokens; long queries are noisy
    return q[:max_chars] if q else claim[:max_chars]


REWRITE_SYS = (
    "You convert claims into short web search queries. "
    "Output ONLY a single line: 3-6 keywords or short phrases, "
    "names in quotes if multi-word. No prose, no preamble, no explanation. "
    "Focus on proper nouns, numbers, dates, and the specific assertion."
)


def rewrite_query(llm: Llama, claim: str, debug: bool = False) -> str:
    """One LLM call per claim to convert a sentence into a keyword query.

    Real search engines weight rare tokens. A sentence like
    'James Talarico is a Presbyterian seminarian.' dilutes the rare tokens
    with stopwords. We want '"James Talarico" Presbyterian seminarian'.
    Falls back to a stopword-stripping heuristic if the LLM output is
    suspicious (too long, contains JSON, contains prose).
    """
    resp = llm.create_chat_completion(
        messages=[{"role": "system", "content": REWRITE_SYS},
                  {"role": "user", "content": f"Claim: {claim}\n\nQuery:"}],
        max_tokens=40,
        temperature=0.3,
        top_p=0.9,
        repeat_penalty=1.15,
        stop=["<|im_end|>", "<|endoftext|>", "\n"],
        stream=False,
    )
    raw = resp["choices"][0]["message"]["content"].strip()
    # Strip any markdown/prefix the model might have added.
    raw = re.sub(r"^(query|search|keywords?)\s*[:\-]\s*", "", raw,
                 flags=re.I).strip()
    raw = raw.strip("`*").strip()

    # Sanity checks: query should be short and look like keywords, not prose.
    if (not raw or len(raw) > 200 or len(raw.split()) > 12
            or any(c in raw for c in "{}[]")
            or raw.lower().startswith(("here ", "the query", "i would"))):
        if debug:
            print(f"[debug: rewrite rejected {raw!r}, using heuristic]",
                  file=sys.stderr)
        return _heuristic_keywords(claim)
    return raw


# --- claim extraction (LLM pass) -------------------------------------------

EXTRACT_SYS = (
    "List factual claims from the text as a JSON array of strings. "
    "Include names, numbers, dates, roles, and events. "
    "Output ONLY valid JSON, e.g.: [\"Claim one.\", \"Claim two.\"]"
)


def _extract_json_array(text: str, debug: bool = False) -> list[str]:
    """Pull a list of claim strings out of an LLM response.

    Strategy (in order of preference):
      1. JSON array inside a ```json ... ``` fence.
      2. JSON array between the first '[' and last ']'.
      3. JSON array after best-effort repair (trailing commas, missing
         closing bracket, single quotes -> double quotes).
      4. Line-based fallback: any line that looks like a numbered or
         bulleted claim (e.g. '1. foo', '- bar', '* baz').

    Small 0.8B models routinely violate strict JSON; we'd rather take a
    fuzzy list than return zero claims.
    """
    def _maybe_log(label: str, value: str) -> None:
        if debug:
            print(f"[debug:_extract_json_array:{label}] {value!r}",
                  file=sys.stderr)

    # 1. Fenced JSON.
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    candidates: list[str] = []
    if m:
        candidates.append(m.group(1))

    # 2. First '[' through last ']'.
    i, j = text.find("["), text.rfind("]")
    if i != -1 and j != -1 and j > i:
        candidates.append(text[i:j + 1])

    # 3. Same span, with repairs.
    for cand in list(candidates):
        repaired = cand
        # Smart quotes -> ASCII.
        repaired = repaired.translate(str.maketrans({
            "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
        }))
        # Trailing comma before closing bracket: ["a", "b",] -> ["a", "b"]
        repaired = re.sub(r",\s*\]", "]", repaired)
        # Single-quoted strings: ['a', 'b'] -> ["a", "b"]
        if "'" in repaired and '"' not in repaired:
            repaired = repaired.replace("'", '"')
        if repaired != cand:
            candidates.append(repaired)

    # 4. Missing closing bracket entirely (model ran out of tokens).
    if i != -1 and j == -1:
        tail = text[i:]
        # Strip trailing partial element after the last quote+comma.
        last_close = max(tail.rfind('",'), tail.rfind('", '))
        if last_close != -1:
            candidates.append(tail[:last_close + 1] + "]")

    for cand in candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError as e:
            _maybe_log("json_fail", f"{e.msg} :: {cand[:200]}")
            continue
        if not isinstance(data, list):
            continue
        claims = [str(x).strip() for x in data
                  if isinstance(x, (str, int, float))]
        claims = [c for c in claims if c]
        if claims:
            return claims[:_MAX_CLAIMS]

    # 5. Line-based fallback. Look for numbered or bulleted lines.
    _maybe_log("fallback", "json parsing failed, scanning for list lines")
    line_re = re.compile(r"^\s*(?:\d+[\.\)\:]|[-*\u2022])\s+(.+?)\s*$")
    out: list[str] = []
    for line in text.splitlines():
        m2 = line_re.match(line)
        if not m2:
            continue
        claim = m2.group(1).strip().strip('"').strip("'").strip(",").strip()
        if len(claim) >= 8:  # filter out empty bullets / noise
            out.append(claim)
    return out[:_MAX_CLAIMS]


def extract_claims(llm: Llama, text: str, debug: bool = False) -> list[str]:
    """One non-streaming LLM call to get a JSON list of claims."""
    user = (
        f"Text:\n{text}\n\n"
        "Output a JSON array of factual claims from this text. "
        'Example format: ["Person X did Y.", "Event Z happened on date W."]'
    )
    resp = llm.create_chat_completion(
        messages=[{"role": "system", "content": EXTRACT_SYS},
                  {"role": "user", "content": user}],
        max_tokens=1024,  # was 600; got cut off mid-array on longer inputs
        # 0.4 is better than 0.6 for pure extraction/formatting: reduces the
        # model's tendency to "decide" to output something unusual like [].
        # repeat_penalty still needed to prevent phrase loops on longer inputs.
        temperature=0.4,
        top_p=0.95,
        repeat_penalty=1.15,
        stop=["<|im_end|>", "<|endoftext|>"],
        stream=False,
    )
    raw = resp["choices"][0]["message"]["content"]
    # Strip any <think>...</think> block before parsing — the model sometimes
    # emits chain-of-thought ahead of the JSON array, which confuses the
    # bracket-scanning logic in _extract_json_array.
    raw = strip_thinking(raw)
    if debug:
        print("[debug:extract_claims:raw_response]", file=sys.stderr)
        print(raw, file=sys.stderr)
        print("[debug:extract_claims:end]", file=sys.stderr)
    return _extract_json_array(raw, debug=debug)


# --- claim ranker (cuts to top-N before search) ----------------------------

# Each claim costs ~4 LLM calls (rewrite, summarize, synthesize, evaluate)
# plus the search round-trip. At 9 claims that's 36+ calls. Ranking to top-N
# drops the budget to ~13 calls and skips minor/vague claims that the small
# model can't verify reliably anyway. Thinking is enabled so the model
# can compare claims against each other before committing to indices.

RANK_SYS = (
    "You score claims by how well a small fact-checking system can verify "
    "them. Higher score = better for fact-checking. Use <think> to weigh "
    "tradeoffs, then output ONLY a JSON array of the chosen claim indices "
    "(1-based, most-verifiable first), with no other text.\n"
    "Prefer claims that have ALL of: specific proper nouns, concrete "
    "numbers/dates, and a clear unambiguous assertion. "
    "Penalize claims that are vague ('some', 'many'), interpretive "
    "('controversial', 'effectively'), or about feelings/intent."
)


def _extract_index_array(text: str, n_claims: int, k: int) -> list[int]:
    """Parse a JSON array of 1-based claim indices. Returns 0-based, deduped,
    bounded. Falls back to [0..k-1] if parsing fails — i.e. keep the first
    k claims in their original order, which is usually a sane default since
    extractors tend to emit lead-paragraph claims first."""
    i, j = text.find("["), text.rfind("]")
    if i != -1 and j > i:
        try:
            data = json.loads(text[i:j + 1])
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list):
            out: list[int] = []
            seen: set[int] = set()
            for v in data:
                try:
                    idx = int(v) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < n_claims and idx not in seen:
                    seen.add(idx)
                    out.append(idx)
                if len(out) >= k:
                    break
            if out:
                return out
    # Fallback: keep the first k claims.
    return list(range(min(k, n_claims)))


def rank_claims(llm: Llama, claims: list[str], k: int = _RANK_TO,
                debug: bool = False) -> list[int]:
    """Return up to k 0-based indices into claims, ordered best-first."""
    if len(claims) <= k:
        return list(range(len(claims)))
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    user = (f"Claims:\n{numbered}\n\n"
            f"Pick the top {k} most fact-checkable claims and output their "
            f"1-based indices as a JSON array, e.g. [3, 1, 7].")
    resp = llm.create_chat_completion(
        messages=[{"role": "system", "content": RANK_SYS},
                  {"role": "user", "content": user}],
        max_tokens=512,  # leave room for <think> block
        temperature=0.6,
        top_p=0.95,
        repeat_penalty=1.15,
        stop=["<|im_end|>", "<|endoftext|>"],
        stream=False,
    )
    raw = resp["choices"][0]["message"]["content"]
    if debug:
        print("[debug:rank_claims:raw_response]", file=sys.stderr)
        print(raw, file=sys.stderr)
        print("[debug:rank_claims:end]", file=sys.stderr)
    stripped = strip_thinking(raw)
    return _extract_index_array(stripped, len(claims), k)


# --- evidence summarization (per-claim, batched over snippets) -------------

# Per the ClaimCheck (arXiv:2510.01226) error analysis, small Qwen models fed
# raw search snippets tend to summarize the whole prompt rather than judge it.
# So we run a dedicated compression pass first: per claim, compress all K
# snippets in a single call (saves K-1 round trips vs one-call-per-snippet)
# and emit a numbered list of <=2-sentence summaries. Snippets that don't
# touch the claim get the literal token NONE and are discarded.

SUMMARIZE_EV_SYS = (
    "You read search snippets and decide whether each one actually "
    "mentions facts relevant to a claim. "
    "For each numbered snippet, output one line: '<n>. <facts>'.\n"
    "STRICT RULES:\n"
    "(1) Your output for snippet <n> must use SPECIFIC PHRASES, NAMES, OR "
    "NUMBERS that literally appear in snippet <n>. Do not paraphrase the "
    "claim — paraphrase the snippet.\n"
    "(2) If snippet <n> does not mention the claim's subject at all, or "
    "only mentions unrelated people/topics, output exactly '<n>. NONE'.\n"
    "(3) Do NOT invent facts. Do NOT restate the claim as if it were in the "
    "snippet. If the snippet talks about someone else entirely, that is "
    "NONE — even if the claim 'sounds reasonable'.\n"
    "(4) At most 2 short sentences per snippet.\n"
    "Output one line per snippet, in order. No preamble, no extra lines."
)


# Stopwords for the grounding overlap check. Same intuition as the keyword
# heuristic earlier: a real summary must share content-bearing tokens with
# its source snippet, not just stopwords.
_GROUND_STOP = frozenset({
    "a", "about", "after", "again", "against", "all", "also", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "between", "both", "but", "by", "can", "did", "do", "does",
    "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "him",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now",
    "of", "off", "on", "once", "only", "or", "other", "our", "ours", "out",
    "over", "own", "same", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "you", "your",
    "yours", "claim", "claims", "claimed", "would", "could", "should",
})


def _content_tokens(text: str) -> set[str]:
    """Lowercased alphanumeric tokens of length >= 4 that aren't stopwords.

    These are the 'content-bearing' words: proper nouns, numbers, and
    domain-specific terms. Used to check that a summary actually echoes
    its source snippet rather than echoing the claim or hallucinating.
    """
    tokens = re.findall(r"[\w$]+", text.lower())
    return {t for t in tokens
            if len(t) >= 4 and t not in _GROUND_STOP
            and not t.isdigit() or any(c.isdigit() for c in t) and len(t) >= 2}


def _claim_subjects(claim: str) -> list[str]:
    """Extract likely named entities from the claim: capitalized
    multi-word sequences and standalone capitalized words of >=4 chars.

    Examples:
      'James Talarico raised $27M' -> ['James Talarico']
      'Trump endorsed Senator John Cornyn' -> ['Trump', 'Senator John Cornyn']
      'Some billionaires fund PACs' -> ['PACs']  (proper nouns only)
    """
    # First, multi-word capitalized sequences (most useful — full names).
    multi = re.findall(r"\b(?:[A-Z][a-z'’]+(?:\s+[A-Z][a-z'’]+)+)\b", claim)
    # Then standalone capitalized words >=4 chars, EXCLUDING ones already
    # captured in multi-word matches.
    multi_words = set()
    for m in multi:
        multi_words.update(m.split())
    singles = [
        w for w in re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", claim)
        if w not in multi_words
    ]
    # Sentence-initial words are noisy ('The', 'Some', 'Republicans') —
    # drop common ones. Keep proper-noun-shaped ones like 'PACs', 'AIPAC'.
    _SENTENCE_STARTERS = {
        "The", "This", "That", "These", "Those", "Some", "Many", "Most",
        "Several", "Both", "Each", "Every", "Any", "All", "None", "Few",
        "Some", "Other", "Another", "Such", "What", "When", "Where", "Why",
        "Republicans", "Democrats", "President",
    }
    singles = [w for w in singles if w not in _SENTENCE_STARTERS]
    return multi + singles


def _mentions_any_subject(text: str, subjects: list[str]) -> bool:
    """Case-insensitive substring match — does `text` mention any subject?

    For multi-word subjects we accept the last word alone too (so 'James
    Talarico' matches a snippet that just says 'Talarico'). This handles
    the common case where articles use surnames after first introduction.
    """
    if not subjects:
        return True  # no subjects to check against — pass through
    low = text.lower()
    for s in subjects:
        if s.lower() in low:
            return True
        # Try the last word too (surname-only matching).
        last = s.rsplit(" ", 1)[-1].lower()
        if len(last) >= 4 and last in low:
            return True
    return False


def _is_grounded(summary: str, snippet: str, title: str = "",
                 url: str = "", claim: str = "",
                 min_overlap: int = 2) -> bool:
    """True if the summary is plausibly derived from its source snippet
    AND the source is actually on-topic for the claim.

    Layered checks against the small model's worst failure mode (restating
    the claim verbatim as if extracted from an unrelated snippet):

    (1) Subject grounding: at least one named entity from the claim must
        appear in the snippet+title+url. If the claim is about Talarico
        but no source field mentions Talarico, the snippet is off-topic
        regardless of any incidental keyword overlap. Catches the case
        where a snippet about Carson's PAC funding gets summarized as
        being about Talarico's PACs.

    (2) Hard floor: summary must share at least `min_overlap` content-
        bearing tokens with the snippet+title. A pure hallucination shares
        zero tokens with its 'source'.

    (3) Echo guard: when overlap is borderline (between min_overlap and
        ~4 tokens), require at least one overlapping token that does NOT
        appear in the claim itself. Catches 'snippet mentions Schumer +
        claim mentions Schumer' coincidental keyword matches.

    (4) Large overlap is trusted: ≥5 distinct content tokens shared is
        very unlikely to be coincidental. Real summaries of a snippet
        about the claim will have heavy overlap that may all happen to
        also be in the claim — we don't want to reject those.
    """
    # Subject grounding: source must mention an entity from the claim.
    if claim:
        subjects = _claim_subjects(claim)
        # Include url because slugs often contain the subject name
        # (e.g. /talarico-raised-27m/), which is the key disambiguator
        # when the snippet body has dropped the subject.
        source_text = f"{snippet} {title} {url}"
        if subjects and not _mentions_any_subject(source_text, subjects):
            return False

    src_tokens = _content_tokens(snippet) | _content_tokens(title)
    sum_tokens = _content_tokens(summary)
    if not sum_tokens:
        return False
    overlap = src_tokens & sum_tokens
    if len(overlap) < min_overlap:
        return False
    # Echo guard only when overlap is small enough to plausibly be
    # coincidental keyword matches.
    if claim and len(overlap) <= 4:
        claim_tokens = _content_tokens(claim)
        if overlap and overlap.issubset(claim_tokens):
            return False
    return True


def _parse_summaries(raw: str, n: int) -> list[str]:
    """Pull N numbered lines out of the model's response. Tolerant of the
    usual small-model failures: prose preamble, markdown bold wrappers,
    any bracket style around the digit (small models mirror whichever
    format they saw or invent their own), missing numbers, extra blank
    lines, spaced separators."""
    out: list[str] = [""] * n
    # Optional leading bold/whitespace, then the digit optionally wrapped
    # in any of [...], <...>, or (...), then a separator (which may be
    # surrounded by spaces). Examples that should match:
    #   "1. foo"      "1) foo"     "1: foo"     "1 - foo"
    #   "[1]. foo"    "[1] foo"    "<1>. foo"   "<1> foo"
    #   "(1) foo"     "**1.** foo" "**[1]** foo"
    line_re = re.compile(
        r"^\s*\**\s*[\[<\(]?\s*(\d+)\s*[\]>\)]?\**\s*"
        r"(?:[\.\)\:\-]\s*|\s+)(.+?)\s*$"
    )
    for line in raw.splitlines():
        m = line_re.match(line)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if 0 <= idx < n and not out[idx]:
            # Drop trailing bold markers the model might add.
            out[idx] = m.group(2).strip().strip("*").strip()
    return out


def summarize_evidence(
    llm: Llama,
    claim: str,
    results: list[tuple[str, str, str]],
    debug: bool = False,
) -> list[str]:
    """Return one summary per result; empty string == filtered out (NONE)."""
    if not results:
        return []
    block = "\n".join(
        f"[{i + 1}] {title}\n{snippet}"
        for i, (title, _url, snippet) in enumerate(results)
    )
    user = (f"Claim: {claim}\n\nSnippets:\n{block}\n\n"
            f"Compressed summaries (one line per snippet):")
    resp = llm.create_chat_completion(
        messages=[{"role": "system", "content": SUMMARIZE_EV_SYS},
                  {"role": "user", "content": user}],
        max_tokens=60 * len(results),  # ~2 sentences each, generous
        temperature=0.6,
        top_p=0.95,
        repeat_penalty=1.15,
        stop=["<|im_end|>", "<|endoftext|>"],
        stream=False,
    )
    raw = resp["choices"][0]["message"]["content"]
    if debug:
        print(f"[debug:summarize raw] {raw!r}", file=sys.stderr)
    parsed = _parse_summaries(raw, len(results))

    # Grounding check (deterministic, no LLM): a summary must (1) actually
    # echo its source snippet rather than the claim, AND (2) the source
    # must mention an entity from the claim. Small models love restating
    # the claim verbatim as if extracted from an unrelated snippet — this
    # is the single most dangerous failure mode for a fact-checker because
    # it produces confident SUPPORTED verdicts on off-topic sources.
    grounded: list[str] = []
    for i, (s, (title, url, snippet)) in enumerate(zip(parsed, results)):
        if not s or s.strip().upper().startswith("NONE"):
            grounded.append("")
            continue
        if _is_grounded(s, snippet, title, url=url, claim=claim):
            grounded.append(s)
        else:
            if debug:
                print(f"[debug:summarize ungrounded snippet {i + 1}: "
                      f"off-topic source or claim-echo -> NONE]",
                      file=sys.stderr)
            grounded.append("")
    return grounded


# --- synthesis (per-claim CoT, sets up the evaluator) ----------------------

# This is the "thinking" step the small model needs. Don't ask it to label
# yet; ask it to lay out what supports/contradicts the claim and what's
# missing. The evaluator then picks a label from this synthesis, not from
# raw snippets. ClaimCheck's ablation showed disabling reasoning on the
# evaluation step cost ~10 points; this is our equivalent.

SYNTHESIZE_SYS = (
    "You analyze evidence for a claim by answering three questions. "
    "Output EXACTLY three lines, no more, no less.\n"
    "\n"
    "Line 1 — 'Supporting: <answer>'\n"
    "  Question: Does any summary state facts that match or back the claim?\n"
    "  If yes, give one sentence quoting the relevant fact.\n"
    "  If no, write exactly: none\n"
    "\n"
    "Line 2 — 'Contradicting: <answer>'\n"
    "  Question: Does any summary state a fact that DIRECTLY contradicts "
    "the claim (says the opposite of what the claim says)?\n"
    "  Saying nothing about the claim is NOT contradicting. "
    "Adding extra detail is NOT contradicting. "
    "Mentioning unrelated people is NOT contradicting.\n"
    "  If yes, give one sentence quoting the contradicting fact.\n"
    "  If no, write exactly: none\n"
    "\n"
    "Line 3 — 'Gaps: <answer>'\n"
    "  Question: What information would you need but don't have?\n"
    "  If the summaries fully cover the claim, write exactly: none\n"
    "\n"
    "Use ONLY words and facts from the summaries above. Do not invent."
)


# One-shot example shown to the model in the user prompt. Small models
# follow demonstrations more reliably than rules; the example makes the
# 'Contradicting: none' path concrete instead of theoretical.
_SYNTH_EXAMPLE = """Example:
Claim: The Eiffel Tower was built in 1889.
Evidence summaries:
- The Eiffel Tower was completed in 1889 for the World's Fair.
- It stands 330 meters tall in Paris.

Analysis:
Supporting: The summaries state the Eiffel Tower was completed in 1889 for the World's Fair.
Contradicting: none
Gaps: none

Now do the same for:"""


def synthesize_evidence(llm: Llama, claim: str, summaries: list[str]) -> str:
    """Return a short structured analysis. Caller has already filtered empty
    summaries. Uses <think> for reasoning before producing the structured
    output (ClaimCheck ablation showed this is one of the two highest-value
    spots for reasoning)."""
    if not summaries:
        return ("Supporting: none. Contradicting: none. "
                "Gaps: no usable evidence retrieved.")
    bullets = "\n".join(f"- {s}" for s in summaries)
    user = (f"{_SYNTH_EXAMPLE}\n\n"
            f"Claim: {claim}\n\nEvidence summaries:\n{bullets}\n\n"
            f"Use <think> to check each line's question, then output "
            f"exactly three lines.")
    resp = llm.create_chat_completion(
        messages=[{"role": "system", "content": SYNTHESIZE_SYS},
                  {"role": "user", "content": user}],
        max_tokens=512,  # extra room for <think> block
        temperature=0.6,
        top_p=0.95,
        stop=["<|im_end|>", "<|endoftext|>"],
        stream=False,
    )
    raw = resp["choices"][0]["message"]["content"]
    return strip_thinking(raw)


# --- evaluate (verdict, given the synthesis) -------------------------------

# Four labels matching AVeriTeC and ClaimCheck. The judge sees the synthesis,
# not raw snippets, which makes the choice nearly mechanical. We allow a
# short scratchpad ("Reasoning: ...") before the verdict because forcing a
# 0.8B model to emit a label as the first token tends to lock it into one
# answer regardless of evidence.

EVALUATE_SYS = (
    "You pick one verdict for a claim, given an evidence analysis "
    "already structured as Supporting/Contradicting/Gaps. "
    "Read the analysis literally: if a line says 'none', that side has "
    "NO evidence — treat it as absent. "
    "Use <think> to reason briefly, then output ONE line: "
    "'Verdict: <X>' where X is EXACTLY ONE of "
    "SUPPORTED, REFUTED, CONFLICTING, INSUFFICIENT.\n"
    "Decision rules (apply in order):\n"
    "1. Supporting has content AND Contradicting says 'none' -> SUPPORTED\n"
    "2. Supporting says 'none' AND Contradicting has content -> REFUTED\n"
    "3. Supporting has content AND Contradicting has content -> CONFLICTING\n"
    "4. Both say 'none' -> INSUFFICIENT\n"
    "Do not invent contradictions where the analysis says none."
)

_VERDICT_RE = re.compile(
    r"\b(SUPPORTED|REFUTED|CONFLICTING|INSUFFICIENT)\b", re.I
)

# Match "Supporting: none" and "Contradicting: none" lines so we can short-
# circuit the LLM evaluator when the synthesis already encoded the verdict.
# Handles markdown bold around the label (**Supporting:** value), bold
# around the value (Supporting: **value**), and capitalized 'None'.
_SYNTH_LINE_RE = re.compile(
    r"^\s*\**\s*(supporting|contradicting|gaps)\s*\**\s*:\s*\**\s*"
    r"(.+?)\s*\**\s*$",
    re.I | re.M,
)


def _parse_synth_sides(analysis: str) -> tuple[str, str]:
    """Return (supporting_value, contradicting_value), stripped.

    Empty string indicates 'none' or a missing line, so caller can use
    `if supporting and not contradicting`. Strips markdown bold and
    trailing punctuation.
    """
    sides = {"supporting": "", "contradicting": ""}
    for m in _SYNTH_LINE_RE.finditer(analysis):
        key = m.group(1).lower()
        val = m.group(2).strip().strip("*").strip().rstrip(".").strip()
        if key in sides:
            sides[key] = "" if val.lower() in ("none", "n/a", "") else val
    return sides["supporting"], sides["contradicting"]


def evaluate_verdict(llm: Llama, claim: str, analysis: str,
                     debug: bool = False) -> str:
    """Return one of SUPPORTED / REFUTED / CONFLICTING / INSUFFICIENT.

    Fast path: when the synthesis is unambiguous (one side has 'none', the
    other has content), skip the LLM call and return the deterministic
    verdict. This sidesteps the 0.8B model's tendency to default to
    CONFLICTING when forced through a thinking pass on clear cases.

    LLM path: used when both sides have content (real conflict) or both
    say 'none' (no evidence). Uses <think> for reasoning.
    """
    supporting, contradicting = _parse_synth_sides(analysis)

    # Deterministic shortcuts.
    if supporting and not contradicting:
        if debug:
            print("[debug:eval shortcut -> SUPPORTED (no contradiction "
                  "in analysis)]", file=sys.stderr)
        return "SUPPORTED"
    if contradicting and not supporting:
        if debug:
            print("[debug:eval shortcut -> REFUTED (no support in analysis)]",
                  file=sys.stderr)
        return "REFUTED"
    if not supporting and not contradicting:
        if debug:
            print("[debug:eval shortcut -> INSUFFICIENT (no support, no "
                  "contradiction)]", file=sys.stderr)
        return "INSUFFICIENT"

    # Both sides have content; let the LLM judge whether the contradiction
    # is real or just the synthesizer padding.
    user = (f"Claim: {claim}\n\nEvidence analysis:\n{analysis}\n\n"
            f"Use <think> to check if Contradicting actually contradicts "
            f"the claim or just adds unrelated detail. Then output "
            f"'Verdict: <X>'.")
    resp = llm.create_chat_completion(
        messages=[{"role": "system", "content": EVALUATE_SYS},
                  {"role": "user", "content": user}],
        max_tokens=512,  # leave room for <think> block
        temperature=0.6,  # Qwen's recommended; 0.0 makes small models loop.
        top_p=0.95,
        stop=["<|im_end|>", "<|endoftext|>"],
        stream=False,
    )
    raw = strip_thinking(resp["choices"][0]["message"]["content"])
    # Prefer a match that appears after "Verdict:" to avoid catching the
    # label names from the system prompt echoed back to us.
    m = re.search(
        r"Verdict\s*:\s*\**\s*(SUPPORTED|REFUTED|CONFLICTING|INSUFFICIENT)",
        raw, re.I,
    )
    if not m:
        m = _VERDICT_RE.search(raw)
    return m.group(1).upper() if m else "INSUFFICIENT"


# --- orchestrator ----------------------------------------------------------

def task_check(llm: Llama, text: str, source: str, verbose: bool,
               debug: bool = False) -> None:
    clipped, was_clipped = clip(text)
    if was_clipped:
        print(f"[note: input clipped to ~{INPUT_CHAR_BUDGET} chars to fit context]",
              file=sys.stderr)
    if debug:
        print(f"[debug:task_check] input length: {len(clipped)} chars, "
              f"source: {source}", file=sys.stderr)

    print("[extracting claims...]", file=sys.stderr)
    t0 = time.perf_counter()
    raw_claims = extract_claims(llm, clipped, debug=debug)

    # Dedup: 0.8B models loop on phrases. Normalize whitespace/case before
    # comparing so "Foo. " and "foo" don't both survive.
    seen: set[str] = set()
    claims: list[str] = []
    for c in raw_claims:
        key = re.sub(r"\s+", " ", c.lower()).strip().rstrip(".")
        if key in seen:
            continue
        seen.add(key)
        claims.append(c)
    if debug and len(claims) < len(raw_claims):
        print(f"[debug: deduped {len(raw_claims) - len(claims)} repeated "
              f"claim(s)]", file=sys.stderr)

    if verbose or debug:
        print(f"[extracted {len(claims)} claim(s) in "
              f"{time.perf_counter() - t0:.1f}s]", file=sys.stderr)
    if debug and claims:
        for i, c in enumerate(claims, 1):
            print(f"[debug:claim {i}] {c}", file=sys.stderr)

    if not claims:
        print("[no verifiable claims found]", file=sys.stderr)
        if not debug:
            print("[hint: re-run with --debug to see the raw model output]",
                  file=sys.stderr)
        return

    # Rank: cut to top _RANK_TO most fact-checkable claims. Each claim costs
    # ~4 LLM calls downstream, so ranking 9 -> 3 cuts ~24 calls. Also drops
    # vague/interpretive claims the model can't verify reliably anyway.
    if len(claims) > _RANK_TO:
        rank_t0 = time.perf_counter()
        keep_indices = rank_claims(llm, claims, k=_RANK_TO, debug=debug)
        if verbose or debug:
            print(f"[ranked {len(claims)} -> {len(keep_indices)} claim(s) "
                  f"in {time.perf_counter() - rank_t0:.1f}s]",
                  file=sys.stderr)
        if debug:
            for rank, idx in enumerate(keep_indices, 1):
                print(f"[debug:ranked #{rank}] (was claim {idx + 1}) "
                      f"{claims[idx]}", file=sys.stderr)
        claims = [claims[i] for i in keep_indices]

    # Parallel search: each claim tries its assigned backend first (rotation
    # by claim index), falls back to the other on empty. Jitter inside
    # search_with_fallback prevents the thread pool from looking like a
    # scrape attack. Sequential LLM pipeline (summarize -> synthesize ->
    # evaluate) happens after, forced by the single llama.cpp context.
    # Build search queries via LLM rewrite (sequential — same llama context).
    # Real search engines weight rare tokens; sending whole sentences with
    # stopwords ('is a', 'who', etc.) tanks recall on rare proper nouns.
    rewrite_t0 = time.perf_counter()
    queries = [rewrite_query(llm, c, debug=debug) for c in claims]
    if verbose or debug:
        print(f"[rewrote {len(queries)} querie(s) in "
              f"{time.perf_counter() - rewrite_t0:.1f}s]", file=sys.stderr)
    if debug:
        for i, (c, q) in enumerate(zip(claims, queries), 1):
            print(f"[debug:query {i}] {q!r}  (from: {c[:80]!r})",
                  file=sys.stderr)

    if verbose or debug:
        print(f"[searching {len(claims)} claim(s) in parallel...]",
              file=sys.stderr)

    search_t0 = time.perf_counter()
    # results_per_claim[i] is (list_of_results, backend_name_that_returned)
    with ThreadPoolExecutor(max_workers=min(8, len(claims))) as pool:
        futures = [
            pool.submit(search_with_fallback, q, i, _TOP_K, debug)
            for i, q in enumerate(queries)
        ]
        results_per_claim = [f.result() for f in futures]
    if verbose or debug:
        n_ok = sum(1 for r, _ in results_per_claim if r)
        print(f"[searches done in {time.perf_counter() - search_t0:.1f}s, "
              f"{n_ok}/{len(claims)} returned results]", file=sys.stderr)

    # Stream verdicts live; per-claim pipeline is summarize -> synthesize ->
    # evaluate. Each call is a tight, focused prompt so the small model has
    # less room to wander.
    counts = {"SUPPORTED": 0, "REFUTED": 0,
              "CONFLICTING": 0, "INSUFFICIENT": 0}
    for i, (claim, (results, backend_used)) in enumerate(
            zip(claims, results_per_claim), start=1):

        if not results:
            verdict = "INSUFFICIENT"
            print(f"[claim {i}: no results from {backend_used} "
                  f"(blocked, captcha, or empty)]", file=sys.stderr)
            top_url = "-"
        else:
            if debug:
                print(f"[debug:claim {i} got {len(results)} result(s) from "
                      f"{backend_used}]", file=sys.stderr)
                for j, (title, url, snip) in enumerate(results, 1):
                    print(f"  [{j}] {title}", file=sys.stderr)
                    print(f"      {url}", file=sys.stderr)
                    print(f"      {snip[:160]}", file=sys.stderr)

            # 1. Compress snippets, drop irrelevant ones.
            summaries = summarize_evidence(llm, claim, results, debug=debug)
            kept = [(s, r) for s, r in zip(summaries, results) if s]
            if verbose or debug:
                dropped = len(results) - len(kept)
                if dropped:
                    print(f"[claim {i}: dropped {dropped}/{len(results)} "
                          f"snippet(s) as irrelevant]", file=sys.stderr)
            if debug:
                for j, s in enumerate(summaries, 1):
                    marker = "KEEP" if s else "DROP"
                    print(f"  [{marker} {j}] {s or '(NONE)'}",
                          file=sys.stderr)

            if not kept:
                verdict = "INSUFFICIENT"
                top_url = results[0][1]  # show user *something* to inspect
            else:
                kept_summaries = [s for s, _ in kept]
                # 2. Synthesize: short structured analysis.
                analysis = synthesize_evidence(llm, claim, kept_summaries)
                if verbose or debug:
                    print(f"[claim {i} analysis] {analysis}", file=sys.stderr)
                # 3. Evaluate: pick one of four labels.
                verdict = evaluate_verdict(llm, claim, analysis, debug=debug)
                top_url = kept[0][1][1]  # url of the first kept result

        counts[verdict] = counts.get(verdict, 0) + 1
        claim_short = claim if len(claim) <= 100 else claim[:97] + "..."
        print(f"{verdict:12s}  {claim_short}  [{top_url}]", flush=True)

    if verbose or debug:
        total = time.perf_counter() - t0
        print(f"[done in {total:.1f}s | "
              f"SUPPORTED={counts['SUPPORTED']} "
              f"REFUTED={counts['REFUTED']} "
              f"CONFLICTING={counts['CONFLICTING']} "
              f"INSUFFICIENT={counts['INSUFFICIENT']}]", file=sys.stderr)

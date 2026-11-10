"""Six fixed surface rules for a steered text that has stopped saying anything, read off the text alone."""
from collections import Counter

MIN_TOKENS = 5
TTR_MIN_TOKENS = 20             # the type/token rule applies from this length
MIN_TYPE_TOKEN_RATIO = 0.35
MAX_TOKEN_RUN = 5               # one token this many times in a row
NGRAM = 3
MAX_NGRAM_REPEATS = 4           # one 3-gram this many times over
MAX_NON_ASCII_SHARE = 0.30
MIN_ALPHA_SHARE = 0.50


def _tokens(text):
    """The whitespace tokens, lowercased; punctuation is kept, so `--- --- ---` is a repeated token."""
    return [token.lower() for token in str(text or "").split()]


def reasons(text):
    """The names of the rules the text trips, in a fixed order; `()` for a healthy text."""
    text = str(text or "")
    tokens = _tokens(text)
    out = []
    if len(tokens) < MIN_TOKENS:
        out.append("too_short")
    if len(tokens) >= TTR_MIN_TOKENS and len(set(tokens)) / len(tokens) < MIN_TYPE_TOKEN_RATIO:
        out.append("low_type_token_ratio")
    if _longest_run(tokens) >= MAX_TOKEN_RUN:
        out.append("repeated_token")
    if _max_ngram_count(tokens, NGRAM) >= MAX_NGRAM_REPEATS:
        out.append("repeated_ngram")
    if text:
        if sum(ord(c) > 127 for c in text) / len(text) > MAX_NON_ASCII_SHARE:
            out.append("non_ascii")
        if sum(c.isalpha() for c in text) / len(text) < MIN_ALPHA_SHARE:
            out.append("low_alpha")
    return tuple(out)


def _longest_run(tokens):
    longest = run = 0
    previous = object()
    for token in tokens:
        run = run + 1 if token == previous else 1
        previous = token
        longest = max(longest, run)
    return longest


def _max_ngram_count(tokens, n):
    if len(tokens) < n:
        return 0
    counts = Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return max(counts.values())

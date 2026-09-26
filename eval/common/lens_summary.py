"""The lens summariser's prompt: a ranked token list in, plain prose out, so a judge reads the lens as it
reads free-text readers. The summariser sees only the token list."""

SUMMARY_SYSTEM = (
    "You turn a ranked list of vocabulary tokens produced by an interpretability tool into plain English prose. "
    "You know nothing else about the context. Answer with the prose only."
)
SUMMARY_USER = (
    "TOKENS (most important first): {tokens}\n\n"
    "Write two to four English sentences saying what these tokens point to: name the specific words, names, "
    "numbers, or concepts they express. Translate tokens in other languages into English and keep the original "
    "in parentheses. Use only what the tokens say; do not add facts, guesses, or associations that the tokens "
    "do not contain."
)


def summary_request(tokens, meta=None):
    listed = ", ".join("'" + t.strip() + "'" for t in tokens)
    return {"system": SUMMARY_SYSTEM, "user": SUMMARY_USER.format(tokens=listed), "kind": "summary_req", "meta": meta}

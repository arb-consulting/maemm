"""Deterministic scoring (the word rule): normalisation, whole-word hits, pass@N, donor chance. See
`workspace_understanding/methodology.md` §5.1 and `workspace_modulation/methodology.md` §4.1."""

import re, unicodedata

_SCAFFOLD = [
    re.compile(r"<think>.*?</think>", re.S),
    re.compile(r"<\|im_start\|>\s*(system|user|assistant)\s*\n?"),
    re.compile(r"<\|im_end\|>"),
    re.compile(r"<\|endoftext\|>"),
]
_ROLE = re.compile(r"^\s*(system|user|assistant)\s*\n", re.I)
_QUOTES = {"“": '"', "”": '"', "‘": "'", "’": "'"}


def strip_scaffolding(text):
    for rx in _SCAFFOLD:
        text = rx.sub(" ", text)
    return _ROLE.sub("", text)


def normalise(text):
    text = unicodedata.normalize("NFKC", text)
    text = "".join(_QUOTES.get(ch, ch) for ch in text)
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _pattern(form):
    return re.compile(r"(?<!\w)" + re.escape(normalise(form)) + r"(?!\w)")


def whole_word_hit(text, forms):
    t = normalise(strip_scaffolding(text))
    return any(_pattern(f).search(t) for f in forms if normalise(f))


def stem_overlap(a, b):
    a, b = normalise(a), normalise(b)
    if min(len(a), len(b)) < 4:
        return False
    return a in b or b in a


def sample_hits(samples, forms):
    return [whole_word_hit(s, forms) for s in samples]


def pass_at_n(hits, n):
    return any(hits[:n])


def consistency(hits):
    return sum(bool(h) for h in hits) / len(hits) if hits else 0.0


def chance_over_donors(item_forms, donor_readouts, criterion):
    if not donor_readouts:
        return float("nan")
    return sum(bool(criterion(r, item_forms)) for r in donor_readouts) / len(donor_readouts)


def normalise_quote(q):
    return normalise(q).strip(" .,;:!?\"'()[]{}")


def quote_in(q, samples):
    nq = normalise_quote(q)
    return bool(nq) and any(nq in normalise(s) for s in samples)

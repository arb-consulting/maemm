"""Per-feature diagnostics for the bad-feature analysis, from the SAE max-acts tensor.

The unit of analysis is an SAE feature's *max-activating example block*: the 30 x 32-token windows
that `adamkarvonen/sae_max_acts` ships for every feature of the layer-27 Qwen3-8B SAE. Everything
here reduces that block to numbers you can sort a table by -- what token the feature actually peaks
on, how interchangeable its examples are, where in the window the peak sits.

Loading the eval_dirs dumps and the corpus scan lives in dumplib.py instead -- those are read by
the plot scripts too, and this module is about the max-acts tensor alone.

Used by find_bad_features.py and dump_feature_examples.py.
"""
import string

MODEL = "Qwen/Qwen3-8B"
MAXACTS_REPO = "adamkarvonen/sae_max_acts"
MAXACTS_FILE = "acts_Qwen_Qwen3-8B_layer_27_trainer_2_layer_percent_75_context_length_32.pt"

# Qwen's max-acts windows are padded with the chat-template end marker; it is not corpus text.
PAD_MARK = "<|im_end|>"

# Closed class, recovered from the labels in the original unactivatable_features.csv rather than
# taken from a standard stoplist -- the two disagree (that table calls " have", " into", " must" and
# " can" content words), and matching the published table matters more than lexicographic purity.
# A peak token outside this list falls through to capitalized_word / content_word.
FUNCTION_WORDS = {
    "a", "all", "an", "and", "any", "as", "at", "be", "been", "being", "but", "by", "for", "from",
    "his", "her", "if", "in", "is", "it", "my", "not", "of", "on", "or", "our", "that", "the",
    "there", "these", "this", "to", "was", "what", "where", "which", "who", "with", "your",
}


def load_maxacts(path=None, device="cpu"):
    """-> (max_tokens [F, 30, 32] int64, max_acts [F, 30, 32] float32). Downloads if path is None."""
    import torch
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(MAXACTS_REPO, MAXACTS_FILE, repo_type="dataset")
    d = torch.load(path, map_location=device, weights_only=False)
    return d["max_tokens"], d["max_acts"].float()


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


def decode_window(tok, ids, clean=True):
    """Decode one 32-token window.

    clean=True is for display: drop the chat pad marker that bounds short windows and flatten
    newlines so a window stays one CSV line. clean=False returns the raw decode, which is what
    uniqueness is counted on -- two windows that differ only in padding are still two windows.
    """
    t = tok.decode([int(i) for i in ids])
    return t.replace(PAD_MARK, "").replace("\n", " ").strip() if clean else t


def token_class(s):
    """Coarse class of a feature's modal peak token. Order of the tests is load-bearing.

    A leading space marks a word start in Qwen's BPE, so a token WITHOUT one that begins with a
    lowercase letter is mid-word ("erm", "lis"); one that begins with a capital is still a word
    ("PR", "Track"), and one beginning with punctuation (".com", "'s") is neither.
    """
    core = s.strip()
    if not core:
        return "whitespace"
    if core.isdigit():
        return "number"
    if all(c in string.punctuation for c in core):
        return "punctuation"      # ASCII only: a curly quote is not punctuation here, it is content
    if core.lower() in FUNCTION_WORDS:
        return "function_word"
    if core[0].isupper():
        return "capitalized_word"
    if not s.startswith(" ") and core[0].islower():
        return "subword_continuation"
    return "content_word"


def template_jaccard(tokens, n=3):
    """Mean pairwise Jaccard over the examples' token n-GRAM sets (trigrams by default).

    High = the 30 "different" examples are near-duplicates of one boilerplate string, so the feature
    has no generalisable meaning to verbalize -- it is a memorised template. n-grams rather than bare
    token sets because unigram overlap is dominated by common words: every pair of English windows
    shares " the", which puts a floor under the score and squashes the range that matters.

    Separation is wide and stable: template features land at 0.43-0.50, ordinary ones at 0.01.
    """
    sets = [set(zip(*[[int(t) for t in row][k:] for k in range(n)])) for row in tokens]
    m = len(sets)
    if m < 2:
        return float("nan")
    tot = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            u = len(sets[i] | sets[j])
            tot += (len(sets[i] & sets[j]) / u) if u else 0.0
    return tot / (m * (m - 1) / 2)


def feature_diagnostics(f, MT, MA, tok):
    """Every per-feature column of the bad-feature table, from the max-acts block alone."""
    import collections
    A, T = MA[int(f)], MT[int(f)]                       # [n_ex, ctx]
    # Not every feature has a full 30: the dump zero-pads features with fewer real examples, and a
    # dead row would otherwise contribute a spurious peak at position 0 and skew every column.
    live = [i for i in range(A.shape[0]) if float(A[i].max()) > 0]
    A, T = A[live], T[live]
    n_ex = len(live)
    peak_pos = A.argmax(1)
    peak_toks = [tok.decode([int(T[i, int(peak_pos[i])])]) for i in range(n_ex)]
    cnt = collections.Counter(peak_toks)
    modal, modal_n = cnt.most_common(1)[0]
    windows = [decode_window(tok, T[i]) for i in range(n_ex)]
    raw = [decode_window(tok, T[i], clean=False) for i in range(n_ex)]
    top = int(A.max(1).values.argmax())
    # the token AFTER the peak: a feature whose continuation is fixed is encoding a collocation
    nxt = []
    for i in range(n_ex):
        p = int(peak_pos[i])
        if p + 1 < T.shape[1]:
            nxt.append(tok.decode([int(T[i, p + 1])]))
    ncnt = collections.Counter(nxt)
    next_modal, next_n = ncnt.most_common(1)[0] if ncnt else ("", 0)
    # An illustrative span for the modal pattern: the strongest example that shows BOTH the modal
    # peak token and the modal continuation, cut two tokens before the peak and two after it.
    # Cosmetic only -- the detector is (modal_peak_token, next_token, next_consistency) above.
    cand = [i for i in range(n_ex) if peak_toks[i] == modal
            and (nxt[i] if i < len(nxt) else "") == next_modal]
    if cand:
        b = max(cand, key=lambda i: float(A[i].max()))
        pb = int(peak_pos[b])
        span = tok.decode([int(x) for x in T[b][max(0, pb - 2):pb + 3]])
    else:
        span = ""
    return {
        "feature": int(f),
        "corpus_peak": round(float(A.max()), 1),
        "template_J": round(template_jaccard(T), 3),
        "uniq_of": f"{len(set(raw))}/{n_ex}",
        "modal_peak_token": modal,
        "modal_frac": f"{modal_n}/{n_ex}",
        "token_class": token_class(modal),
        "distinct_peak_tokens": len(cnt),
        "mean_peak_pos": round(float(peak_pos.float().mean()), 1),
        "next_token": next_modal,
        "next_frac": f"{next_n}/{len(nxt)}" if nxt else "0/0",
        "next_consistency": (next_n / len(nxt)) if nxt else 0.0,
        "typical_span": span,
        "top_example": windows[top][:150],
    }

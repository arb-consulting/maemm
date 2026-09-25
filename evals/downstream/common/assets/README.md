# The shipped assets

Three files: the centring mean every read is centred with, the index of the held-out corpus every
evaluation's web text comes from, and that corpus's content check against the training text. Each is
integrity-checked on load against its recorded sha256, because a run of one package is compared with a run
of another and a different asset is a different baseline.

# The centring mean

`mu.f32` is the layer-42 centring mean of Qwen3.6-27B: 5,120 float32 values, little-endian, no header.
`mu.meta.json` records its sha256, norm and component statistics, and
`evals.downstream.common.background.load_centring_mean()` refuses a file whose digest differs.

    import numpy as np
    mu = np.fromfile("mu.f32", dtype=np.float32)     # (5120,)

It is the arithmetic mean of the layer-42 block-output residual over every non-sink token position of every
64-token window at stride 16 within the documents of a 16M-token held-out corpus (Ultra-FineWeb, English,
parts 0009-0010, in a fixed permuted document order), under the clean base model with nothing injected. Each
window is forwarded as `[sink] + window` and position 0 is dropped, which is how generated text is re-read.
At stride 16 an interior token appears in four windows, so the mean is weighted by window position rather
than uniform over corpus tokens. It is accumulated as a float64 running sum and cast to float32 once.

It is used for one thing: an activation target is `unit(activation - mu)`. Directions that are not
activations (steering vectors) are never centred. A re-read text is scored against a target by the raw
cosine of its residual (`evals/downstream/common/scorer.py`); `rollout_coherence` also centres that residual with `mu`,
and the cosine with both sides centred is its inversion axis.

The mean belongs to the model pinned in `evals/downstream/common/pins.py`, and the two are replaced together.

# The held-out corpus index

`heldout_docs.csv` is the held-out corpus: one row per document, `doc,part,row,len`, in the corpus's own
permuted document order. `part` 0 and 1 are the two public parquet files named in
`evals.downstream.common.retrieval.Stream.files` (`openbmb/Ultra-FineWeb`, English, parts 0009 and 0010) at the pinned
revision, `row` is the document's row in that file and `len` its token count under the Qwen3.6-27B
tokenizer with `add_special_tokens=False` and no truncation. 18,813 documents, 16,000,731 tokens.
`heldout_docs.meta.json` records its sha256 and those counts, and `evals.downstream.common.retrieval.load_index()`
refuses a file whose digest or counts differ.

    import csv
    rows = list(csv.DictReader(open("heldout_docs.csv")))   # 18,813 rows

The corpus itself is not shipped — it is rebuilt from the two public files by this index
(`retrieval.build_corpus`), and each document's token count is checked against `len`, which is the identity
check: a different tokenizer, a different revision or a row read out of the wrong file all show up there.
Its first 11,700 documents (9,999,741 tokens) are the search corpus every package's retrieval baseline
searches, cut into 64-token windows at stride 16; the 7,113 documents after them are where every package
draws its own evaluation text from, which is what makes queries and corpus disjoint by construction.

What is established about the corpus against the text the inverter, its dictionary and its banks were
trained on is two things, and the next section is the second. The first is that it is INDEX-disjoint: no
held-out row is a row of the dump that any training source read. That says nothing about content, because
Ultra-FineWeb is not deduplicated and the same text can sit in two rows.

# The content check

`heldout_overlap.csv` is one row per held-out document, `doc,coverage_n7,coverage_n13`, in the held-out
order: the share of the document's word 7-grams, and of its word 13-grams, that occur anywhere in the
training text of the checkpoint under evaluation, to six decimals. `heldout_overlap.meta.json` records its
sha256, the rule, the training sources with their row counts and the counts below, and
`evals.downstream.common.retrieval.load_overlap()` refuses a file whose digest or row count differs.

The measure is the training pipeline's own overlap check. A text is NFKC-normalised, lowercased and split on whitespace; its
word n-grams are hashed; a document's coverage at `n` is the fraction of its n-grams found in the training
text, computed over the whole document. The training text is the `target_text` of the checkpoint's
supervised mix and reinforcement pool, 8,941,132 rows over six sources (the meta file names them). It is
not public, so the file cannot be rebuilt from this tree: `heldout_overlap_build.py` is the script that
produced it, kept as provenance.

| | documents | 7-gram coverage ≥ 0.05 | ≥ 0.2 | ≥ 0.5 | ≥ 0.9 | 13-gram ≥ 0.05 | 13-gram ≥ 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| evaluation half | 7,113 | 371 (5.2 %) | 135 | 48 | 13 | 244 | 37 |
| search half | 11,700 | 633 (5.4 %) | 233 | 74 | 23 | 411 | 57 |
| all | 18,813 | 1,004 (5.3 %) | 368 | 122 | 36 | 655 | 94 |

The median coverage is 0 at both lengths in both halves. Over the whole corpus 94 documents (0.50 %) have
at least half of their 13-grams in the training text and 1,622 (8.6 %) share at least one; the training
pipeline's own run of the measure over the same corpus, against a slightly different set of training sources, gives
0.43 % and 9.21 %.

**The rule.** A document whose 7-gram coverage is at least 0.05 (`retrieval.OVERLAP_THRESHOLD`) is
EXCLUDED from every draw of evaluation text: `retrieval.overlap_excluded()` is that set and
`retrieval.select_documents(..., exclude=...)` skips its members in the draw's own order. The threshold is
the training pipeline's. Nothing is dropped from the
corpus and nothing is renumbered: the index, the document order, every count above and every window id
are as they were, and a flagged document is simply never handed to a package as a source of text. Of the
3,215 evaluation documents of at least 512 tokens — the pool a 512-token read draws from — 181 are
flagged and 3,034 remain.

**The search half is not filtered.** It is the corpus the retrieval baseline searches and the one the
paper's reconstruction evaluation scans, document for document, so removing documents from it would make
the baseline a different one from theirs. Nor is there a leak to close: a query is never read out of a
search document, and a window the training text also holds is a window the search is entitled to return.
The coverage of the search half is recorded so that the figure can be stated, not acted on.

Every package reports its search at two sizes (`retrieval.search_rows`): that full 10M prefix, which is its
own row, and the nested 8M prefix (`retrieval.SHARED_SIZE`, 7,999,674 tokens). The paper's reconstruction
evaluation scans the same two files in this same document order, under the same window grid and scoring
rule, at 1M/2M/4M/8M/16M, and its 16M corpus includes the documents this suite holds back for evaluation
text, so 8M is the largest size the two hold document for document and the one a search number is quoted
at beside that evaluation's. Both rankings come out of one forward pass, and both are of NON-OVERLAPPING
windows: at stride 16 a window shares up to 48 tokens with each neighbour, so a query's windows are chosen
greedily in rank order, one kept unless it shares a token with an already-kept window of its document
(`retrieval.suppress_overlapping`), within the prefix the ranking is over. Rank 1, and every top-1 number,
is the best window either way.

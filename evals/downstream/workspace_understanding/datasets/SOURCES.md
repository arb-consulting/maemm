# Vendored inputs

Prompt sets from `anthropics/jacobian-lens` (https://github.com/anthropics/jacobian-lens), commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e` ("Initial release", 2026-07-02), directory `data/evaluations/`,
Apache License 2.0 (`LICENSE-jlens.txt` is that repository's LICENSE). Files are byte-identical copies; the
sha256 below is verified at load by `items.py`. Vendored under `datasets/` rather than `data/` because
upstream's Git ignore pattern excludes any directory of that name.

| file | sha256 | items |
|---|---|---|
| lens-eval-association.json | d1a98cd4911b594282e74168091c77d849dae18ffe2acb5761074853f327d71c | 102 |
| lens-eval-multihop.json | 50b7e4c9255291c0ca2a8e94615be9f44531fa57bb1a844e4f9616056d987416 | 93 |
| LICENSE-jlens.txt | cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30 | – |

Readout rules and the metric definition come from that directory's README (association: the final prompt token,
the closing period; multihop: the token immediately preceding the unwritten answer; pass@k = mean over items of the
fraction of `intermediates` whose min-over-layers lens rank is <= k). See `../methodology.md` §2.

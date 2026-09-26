"""This package's additions to `eval.common.runs`: the whole-file jsonl write and the two stage-record
readers the chained and sharded stages need."""

import json, os

from eval.common.runs import RunDir as _RunDir
from eval.common.runs import json_safe, record_digest


class RunDir(_RunDir):
    """`eval.common.runs.RunDir` plus a whole-file jsonl write."""

    def write_jsonl(self, rel, records):
        """Write a whole jsonl file through a temp file and `os.replace`, so a reader never sees a prefix."""
        tmp = self.file(rel) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as h:
            for obj in records:
                h.write(json.dumps(json_safe(obj), ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(tmp, self.file(rel))


def stage_hashes(run, stages):
    """The digest of each named stage record as it stands (None where absent or not completed), for
    chaining a consuming stage's hash to the records it consumes. The whole record, not its config hash:
    a re-run under the same settings must still move every downstream key."""
    return {s: record_digest(run, s) for s in stages}


def stage_names(run, prefixes):
    """Every `stages/<name>.json` whose name starts with one of `prefixes`, sorted (the per-job records of
    a fanned-out stage, whose number the consumer does not know in advance)."""
    d = os.path.join(run.path, "stages")
    if not os.path.isdir(d):
        return []
    return sorted(n[:-5] for n in os.listdir(d) if n.endswith(".json") and n[:-5].startswith(tuple(prefixes)))

"""The naming instrument: a judge sees one concept's TARGET forms and one readout (free-text samples, or a
lens's prose summary) and answers `{"expressed": bool, "target": ..., "quote": ...}`. A positive verdict
is voided (`void_reason`) unless `target` is an asked-about form and `quote` a verbatim substring of the
readout. Packages also ask each cell against a foil (another item's targets), so net naming is own minus
foil. The judge sees no arm name, item or reader prompt; requests are addressed by `with_judge`.
"""

from eval.common.judge_client import decode_json_object
from eval.common.matcher import normalise, quote_in

NAMING_SYSTEM = """You are judging readouts from an activation-decoding tool for an interpretability benchmark. You get one item: its TARGET concept(s) and one readout taken at one position. For one kind of reader the readout is free-text samples; for the other it is a prose summary of its top vocabulary tokens. Decide whether ANY target concept is EXPRESSED in the readout. STRICT rules:
- Expressed means the target concept is NAMED: the exact target word or phrase as a standalone word (case or inflection of the SAME word is fine: "cats" for "cat"), an established synonym or alias that unambiguously denotes the SAME concept ("Mandarin" for "Chinese"), or a faithful translation of the target into another language ("二月" for "February").
- NOT expressed by a DIFFERENT or DERIVED word that merely contains the target string: "blackout" does not express "black"; "childhood" does not express "child".
- NOT expressed by word fragments or partial tokens, however distinctive.
- NOT expressed by thematically associated imagery, descriptions, or scene content that evokes the concept without naming it. Hypernyms, hyponyms, and category neighbours do not count.
- Judge only what is actually written. If uncertain, say not expressed.
- Every positive verdict must include a VERBATIM quote copied exactly from one sample (the smallest span that names the target). No quote, no pass.
Return JSON only: {"expressed": true|false, "target": "<which target, or null>", "quote": "<verbatim, or null>"}"""
# Readout kinds: each sets the READOUT label and the judge's token cap.
READOUT_LABEL = {"samples": "free-text samples", "summary": "prose summary of the top vocabulary tokens"}
KINDS = tuple(READOUT_LABEL)


def naming_request(targets, kind, samples):
    """One naming request: `targets` against `samples`. `kind` is "samples" or "summary"."""
    body = "\n".join(f"[{k + 1}] <<<{s}>>>" for k, s in enumerate(samples))
    user = f"TARGETS: {'; '.join(targets)}\n\nREADOUT ({READOUT_LABEL[kind]}):\n{body}"
    return {
        "system": NAMING_SYSTEM,
        "user": user,
        "kind": kind,
        "targets": list(targets),
        "samples": list(samples),
    }


def parse_verdict(text):
    """The reply's first JSON object whose `expressed` is a boolean, as a verdict dict (with `repaired`)."""
    obj, repaired = decode_json_object(text, accept=lambda o: isinstance(o.get("expressed"), bool))
    if obj is None:
        return {"expressed": None, "target": None, "quote": None, "parse_ok": False, "repaired": False}
    return {
        "expressed": obj["expressed"],
        "target": obj.get("target"),
        "quote": obj.get("quote"),
        "parse_ok": True,
        "repaired": repaired,
    }


def void_reason(v, targets, samples):
    """None, or why a positive verdict does not count: "bad_target" or "quote_not_found"."""
    if not v["expressed"]:
        return None
    tn = {normalise(t) for t in targets}
    if not v.get("target") or normalise(str(v["target"])) not in tn:
        return "bad_target"
    if not v.get("quote") or not quote_in(str(v["quote"]), samples):
        return "quote_not_found"
    return None

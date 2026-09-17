"""SAE-2M post-training chain: wait for /data/sae2m/shards/TRAIN_DONE (or the train call's return) -> spawn merge -> wait ->
spawn verify + maxacts in parallel -> wait -> notify. Ids -> ~/shared/overnight/sae2m_ids.json ("chain"); log ~/shared/overnight/sae2m_chain.log."""
import json, os, subprocess, time, modal
IDS="/home/celeste/shared/overnight/sae2m_ids.json"; LOG="/home/celeste/shared/overnight/sae2m_chain.log"; APP="maemm-sae2m"
def log(m):
    line=f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {m}"; print(line, flush=True); open(LOG,"a").write(line+"\n")
def ids(): return json.load(open(IDS))
def save(d): json.dump(d, open(IDS,"w"), indent=1)
def vol_has(path, name):
    return name in subprocess.run(["modal","volume","ls","maemm-data",path], capture_output=True, text=True).stdout
def wait_call(fid, what):
    fc=modal.FunctionCall.from_id(fid)
    while True:
        try: r=fc.get(timeout=0); log(f"{what} returned: {json.dumps(r, default=str)[:400]}"); return r
        except TimeoutError: time.sleep(120)
        except Exception as e: log(f"{what} FAILED: {type(e).__name__}: {str(e)[:400]}"); return None
d=ids(); ch=d.setdefault("chain", {})
# 1) wait for TRAIN_DONE
while not vol_has("sae2m/shards","TRAIN_DONE"):
    full=d["full"]; fid=full["id"] if isinstance(full,dict) else full
    try:
        modal.FunctionCall.from_id(fid).get(timeout=0); log("train call returned; re-checking TRAIN_DONE"); time.sleep(30)
        if not vol_has("sae2m/shards","TRAIN_DONE"): log("train call ended WITHOUT TRAIN_DONE — stopping (manual: merge require_done=False)"); subprocess.run(["notify-discord","SAE-2M: train call ended without TRAIN_DONE — chain halted; inspect /data/sae2m/shards"]); raise SystemExit(1)
    except TimeoutError: time.sleep(120)
    except Exception as e: log(f"train call error: {e}"); time.sleep(120)
log("TRAIN_DONE present")
# 2) merge
if "merge" not in ch:
    fc=modal.Function.from_name(APP,"merge").spawn(); ch["merge"]={"id":fc.object_id,"spawned":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; save(d); log(f"merge spawned {fc.object_id}")
r=wait_call(ch["merge"]["id"],"merge"); ch["merge"]["result"]=r; save(d)
if r is None: subprocess.run(["notify-discord","SAE-2M: MERGE failed — chain halted"]); raise SystemExit(1)
subprocess.run(["notify-discord","SAE-2M merged: /data/sae2m/trainer_0/ae.pt ready; verify + maxacts (8xB200, ~5 h) launching"])
# 3) verify + maxacts in parallel
if "verify" not in ch:
    fc=modal.Function.from_name(APP,"verify").spawn(); ch["verify"]={"id":fc.object_id,"spawned":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; save(d); log(f"verify spawned {fc.object_id}")
if "maxacts" not in ch:
    fc=modal.Function.from_name(APP,"maxacts").spawn(); ch["maxacts"]={"id":fc.object_id,"spawned":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; save(d); log(f"maxacts spawned {fc.object_id}")
rv=wait_call(ch["verify"]["id"],"verify"); ch["verify"]["result"]=rv; save(d)
subprocess.run(["notify-discord",f"SAE-2M verify: {json.dumps(rv, default=str)[:300]}"])
rm=wait_call(ch["maxacts"]["id"],"maxacts"); ch["maxacts"]["result"]=rm; save(d)
subprocess.run(["notify-discord",f"SAE-2M MAXACTS done: {json.dumps(rm, default=str)[:300]} — next: bank build"])
log("chain complete")

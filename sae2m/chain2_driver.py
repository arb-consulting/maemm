"""CHAIN 2 (2M-SAE downstream): wait maxacts_top5.pt -> bank build (maemm-sae2m-bank) -> verify -> compose mix_5m_sae2m (maemm-mix-5m-bank3)
-> FFT midtrain (maemm-sft-fullft, eff 4096, lr 1e-5, max_seq 192, init = 104M FFT final) + evaluator -> full-param RL 8x2048 + all-token reward
(spawn_rl_fullparam.py prod 1e-6 --split 2+6, app maemm-rl-disagg-fullparam4). ids -> ~/shared/overnight/sae2m_chain2_ids.json ; log -> ~/shared/overnight/sae2m_chain2.log
Idempotent: every stage is skipped when its key exists in the ids file (edit the file + restart to redo a stage)."""
import json, math, os, subprocess, time, modal
os.environ.setdefault("MODAL_PROFILE", "safety-sahan")
IDS = "/home/celeste/shared/overnight/sae2m_chain2_ids.json"; LOG = "/home/celeste/shared/overnight/sae2m_chain2.log"
MAXACTS_CALL = json.load(open("/home/celeste/shared/overnight/sae2m_ids.json"))["chain"]["maxacts"]["id"]
ARM1 = "realact104m_fullft_b4096_lr1e-5"; RUN = "mix5m_sae2m_midtrain_fft_from_fft104m"; POOL = "/data/banks/mix_5m_sae2m"; EFF = 4096
RL_NAME = "rl_fullparam_fft104m_sae2m_8x2048_anywin"; RL_KEY = "fft104m_sae2m_fullparam_8x2048_anywin"
def log(m):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {m}"; print(line, flush=True); open(LOG, "a").write(line + "\n")
def discord(m): subprocess.run(["notify-discord", m])
def wait_call(fc_id, what, poll=90):
    fc = modal.FunctionCall.from_id(fc_id)
    while True:
        try:
            r = fc.get(timeout=0); log(f"{what} finished: {str(r)[:400]}"); return r
        except TimeoutError:
            time.sleep(poll)
        except Exception as e:
            log(f"{what} FAILED: {type(e).__name__}: {str(e)[:600]}"); return None
def vol_ls(path): return subprocess.run(["modal", "volume", "ls", "maemm-data", path], capture_output=True, text=True).stdout
def vol_get(path, dst):
    subprocess.run(["modal", "volume", "get", "maemm-data", path, dst, "--force"], capture_output=True); return json.load(open(dst))
def save(): json.dump(d, open(IDS, "w"), indent=1)
d = json.load(open(IDS)) if os.path.exists(IDS) else {}
# ---- 0. maxacts --------------------------------------------------------------------------------------------------------
if "maxacts" not in d:
    r = wait_call(MAXACTS_CALL, "maxacts")
    if r is None or "maxacts_top5.pt" not in vol_ls("sae2m"):
        discord("chain2: 2M-SAE maxacts FAILED / maxacts_top5.pt missing — bank NOT built"); raise SystemExit(1)
    d["maxacts"] = {"call": MAXACTS_CALL, "result": str(r)[:1000]}; save()
    discord("chain2: 2M-SAE max-acts DONE (fixed stream) -> building the sae2m bank")
# ---- 1. bank build + verify -------------------------------------------------------------------------------------------
if "bank" not in d:
    fc = modal.Function.from_name("maemm-sae2m-bank", "build").spawn(out_name="sae2m_bank", maxacts="/data/sae2m/maxacts_top5.pt",
                                                                    windows_per_feature=3, min_tok=8, max_rows_per_family=3_000_000)
    d["bank"] = {"build": fc.object_id, "out": "/data/banks/sae2m_bank", "spawned": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; save()
    log(f"bank build spawned {fc.object_id}")
if "build_result" not in d["bank"]:
    r = wait_call(d["bank"]["build"], "bank build")
    if r is None or "build_stats.json" not in vol_ls("banks/sae2m_bank"):
        discord("chain2: sae2m bank build FAILED — compose NOT launched"); raise SystemExit(1)
    d["bank"]["build_result"] = str(r)[:2000]
    v = modal.Function.from_name("maemm-sae2m-bank", "verify").spawn(out_name="sae2m_bank"); d["bank"]["verify"] = v.object_id; save()
    vr = wait_call(v.object_id, "bank verify")
    d["bank"]["verify_result"] = str(vr)[:1000]; save()
    if vr is None:
        discord("chain2: sae2m bank VERIFY failed — compose NOT launched"); raise SystemExit(1)
    st = vol_get("banks/sae2m_bank/build_stats.json", "/tmp/bank5m/sae2m_bank_build_stats.json")
    d["bank"]["n_examples"] = st.get("n_examples"); d["bank"]["families"] = st.get("families"); save()
    discord(f"chain2: sae2m bank READY {st.get('n_examples')} rows {st.get('families')} -> composing mix_5m_sae2m")
# ---- 2. compose ---------------------------------------------------------------------------------------------------------
if "compose" not in d:
    spec = json.load(open("/tmp/bank5m/spec_mix5m_sae2m.json"))
    excl = {"/data/banks/everything_5m_fresh": "/data/banks/everything_5m_fresh/end_anchor_rows.json",
            "/data/banks/mlp42_5m_fresh": "/data/banks/mlp42_5m_fresh/mlp_anchor_rows.json"}
    fc = modal.Function.from_name("maemm-mix-5m-bank3", "build").spawn(out_name="mix_5m_sae2m", spec_json=json.dumps(spec), exclude_json=json.dumps(excl),
                                                                       trim_to_peak="cluster", trim_min_tok=8, seed=2031, threads=48, dense_frac=0.0)
    d["compose"] = {"call": fc.object_id, "bank": POOL, "spec": spec, "spawned": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; save()
    log(f"compose spawned {fc.object_id}")
if "n_examples" not in d["compose"]:
    r = wait_call(d["compose"]["call"], "compose")
    ls = vol_ls("banks/mix_5m_sae2m")
    if r is None or "build_stats.json" not in ls or "vecs.f32" not in ls:
        discord("chain2: compose of mix_5m_sae2m FAILED/incomplete — midtrain NOT launched"); log(f"compose incomplete: {ls[:300]}"); raise SystemExit(1)
    st = vol_get("banks/mix_5m_sae2m/build_stats.json", "/tmp/bank5m/mix5m_sae2m_build_stats.json")
    d["compose"]["n_examples"] = st["n_examples"]; d["compose"]["families"] = st["families"]; save()
    discord(f"chain2: mix_5m_sae2m composed: {st['n_examples']} rows {st['families']} -> launching FFT midtrain")
N_ROWS = d["compose"]["n_examples"]; steps = math.ceil(N_ROWS / EFF)
# ---- 3. FFT midtrain ----------------------------------------------------------------------------------------------------
if "midtrain" not in d:
    extra = ("--full-ft --prefix-cache --grad-ckpt 0 --autocast-bf16 --log-steps 1 --grad-accum 16 --pad-multiple 1 --prefix-share-step "
             f"--fsdp-prefetch 2 --init-adapter /data/sft_mix/{ARM1}/final")
    t = modal.Function.from_name("maemm-sft-fullft", "train").spawn(run_name=RUN, data_dir=POOL, n_ckpts=4, epochs=1, batch_size=32, lr=1e-5,
                                                                     max_seq=192, backend="nccl", extra_args=extra)
    e = modal.Function.from_name("maemm-eval-ckpt-fullft", "fullmodel_daemon").spawn(ckpt_dir=f"/data/sft_mix/{RUN}", tag=f"sft_{RUN}",
                                                                                     wandb_name=f"{RUN}_eval", final_step=steps)
    d["midtrain"] = {"train": t.object_id, "eval": e.object_id, "run": RUN, "save": f"/data/sft_mix/{RUN}", "data": POOL, "eff_batch": EFF, "micro_batch": 32,
                     "grad_accum": 16, "lr": 1e-5, "max_seq": 192, "steps_expected": steps, "init": f"/data/sft_mix/{ARM1}/final", "app": "maemm-sft-fullft",
                     "eval_app": "maemm-eval-ckpt-fullft", "spawned": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "recipe": "FSDP2 full fine-tune midtrain on mix_5m_sae2m (131k-SAE rows replaced by the 2M-feature SAE enc+dec bank; realact 1M, cluster, verbatim BSF, fire-filtered MLP)"}
    save(); log(f"MIDTRAIN spawned: train {t.object_id} eval {e.object_id} ({steps} steps expected)")
    discord(f"chain2: FFT midtrain launched on mix_5m_sae2m ({RUN}, {N_ROWS} rows, {steps} steps ≈ {steps*12/3600:.1f} h at 12 s/step); full-param RL follows")
# ---- 4. full-parameter RL -----------------------------------------------------------------------------------------------
if "rl" not in d:
    wait_call(d["midtrain"]["train"], "midtrain")
    ok = False
    for _ in range(40):
        if "SAVE_DONE" in vol_ls(f"sft_mix/{RUN}/final"): ok = True; break
        time.sleep(60)
    if not ok:
        discord("chain2: midtrain final missing — full-param RL NOT launched"); log("midtrain final missing"); raise SystemExit(1)
    env = dict(os.environ); env["RL_FULLPARAM_APP"] = "maemm-rl-disagg-fullparam4"
    out = subprocess.run(["python3", "/home/celeste/maemm-pub/scripts/launchers/spawn_rl_fullparam.py", "prod", "1e-6", "--split", "2+6",
                          "--policy-base", f"/data/sft_mix/{RUN}/final", "--name", RL_NAME, "--ids-key", RL_KEY, "--",
                          "--groups-per-step", "2048", "--reward-window-last", "0"], capture_output=True, text=True, env=env, cwd="/home/celeste/maemm-pub")
    log("RL spawn: " + out.stdout.strip() + out.stderr.strip()[-400:])
    rec = json.load(open("/home/celeste/shared/overnight/rl_fullparam_ids.json")).get(RL_KEY, {})
    d["rl"] = rec; save()
    discord(f"chain2: midtrain DONE -> FULL-PARAM RL launched: {RL_NAME} (lr 1e-6, 2+6, 8x2048, all-token reward, 300 steps) train {rec.get('train')} eval {rec.get('eval')}")
log("chain2 complete")

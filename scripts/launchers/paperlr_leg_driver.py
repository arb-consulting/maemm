"""LEG DRIVER for the 7400-step full-param RL arm (user 09:2xZ: match the paper's big run: 2,048 prompts, 7,400 steps) at the ScaleRL-paper optimizer setting (user 09:1xZ Sep 10: "pls do this. we train
for 2000 steps at huge batch"): lr 5e-7 constant after 100-step warmup, AdamW eps 1e-15, weight decay 0.01, 8x2048 (16,384 rollouts/step),
same init/pool/recipe as the .425 arm. Modal kills each call at 24 h (~650 steps at ~125 s/step) -> this driver spawns leg 1 via the
launcher, then on every leg end respawns from the latest full-model checkpoint with --step-offset N --wandb-id <same run> (+ a fresh
fullmodel evaluator daemon, same tag -> resumes its state) until step 2000 is saved. AdamW moments reset at leg boundaries (no optimizer
state in the full-model checkpoints) -- recorded in the run notes. Log ~/shared/overnight/paperlr_driver.log; ids rl_fullparam_ids.json["fft104m_mix5m_fullparam_8x2048_paperlr"]."""
import json, os, re, subprocess, time, modal
KEY = "fft104m_mix5m_fullparam_8x2048_paperlr"; NAME = "rl_fullparam_fft104m_mix5m_8x2048_paperlr"
IDS = "/home/celeste/shared/overnight/rl_fullparam_ids.json"; LOG = "/home/celeste/shared/overnight/paperlr_driver.log"
APP, EVAL_APP, TOTAL = "maemm-rl-disagg-fullparam4", "maemm-eval-ckpt-fullrl", 7400
POLICY0 = "/data/sft_mix/mix5msft_midtrain_fft_from_fft104m/final"; POOL = "/data/banks/mix_eq_1p45m"; SAVE = f"/data/ckpts_{NAME}"
PAPER = "--groups-per-step 2048 --warmup-steps 100 --adam-eps 1e-15 --weight-decay 0.01"
def log(m):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {m}"; print(line, flush=True); open(LOG, "a").write(line + "\n")
def ids(): return json.load(open(IDS))
def save_ids(d): json.dump(d, open(IDS, "w"), indent=1)
def latest_ckpt():
    ls = subprocess.run(["modal", "volume", "ls", "maemm-data", SAVE.lstrip("/").replace("data/", "", 1) if SAVE.startswith("/data/") else SAVE], capture_output=True, text=True).stdout
    steps = sorted({int(m) for m in re.findall(r"step_(\d+)", ls)})
    done = []
    for s in reversed(steps):
        sub = subprocess.run(["modal", "volume", "ls", "maemm-data", f"{SAVE[len('/data/'):]}/step_{s}"], capture_output=True, text=True).stdout
        if "SAVE_DONE" in sub: return s
    return None
def wandb_id():
    import wandb
    api = wandb.Api(); rs = list(api.runs("celestedeschamphelaere-personal/maxact-fast", filters={"display_name": NAME}, order="-created_at"))
    return rs[-1].id if rs else None   # the FIRST run created with this name
def spawn_leg(offset, policy, wid):
    extra = (f"--recipe scalerl --loss cispo --cispo-eps-max 5 --loss-agg prompt --adv-mode batch --zero-var-filter --npr-threshold 0.9 --npr-pass-cos 0.7 "
             f"--max-lag 2 --fp32-head --autocast-bf16 --length-control penalty --kl-coef 0 --entropy-coef 0 --entropy-target 0 --groups-per-step 512 "
             f"--group-size 8 --warmup-steps 25 --len-penalty-start 8 --len-penalty-per-tok 0.00025 --max-new-tokens 192 --reward-window-last 5 "
             f"--prefix-cache --score-length-bucket --cuda-graphs --max-num-seqs 512 --rollout-block-groups 32 --save-every 0 --transcript-every 5 "
             f"--lr 5e-7 --save-steps {','.join(['25','50'] + [str(x) for x in range(100, 1001, 100)] + [str(x) for x in range(1250, TOTAL + 1, 250)] + [str(TOTAL)])} --run-name {NAME} --save-dir {SAVE} {PAPER} "
             f"--total-steps {TOTAL} --step-offset {offset}" + (f" --wandb-id {wid}" if wid else ""))
    t = modal.Function.from_name(APP, "train").spawn(n_rollout=2, n_trainer=6, total_steps=TOTAL, extra_args=extra, pool_dir=POOL, policy_base=policy, full_param=True)
    e = modal.Function.from_name(EVAL_APP, "fullmodel_daemon").spawn(ckpt_dir=SAVE, tag=NAME, wandb_name=f"{NAME}_eval", final_step=TOTAL,
                                                                    extra_args="--eval-cache /data/eval_universal_ho/eval_sets_heldout_v2.pt --no-extra-evals")
    return t.object_id, e.object_id
d = ids()
if KEY not in d:
    t, e = spawn_leg(0, POLICY0, None)
    d[KEY] = {"train": t, "eval": e, "run": NAME, "save": SAVE, "policy_base": POLICY0, "pool": POOL, "lr": "5e-7", "group_size": 8, "groups_per_step": 2048,
              "steps": TOTAL, "app": APP, "paper_match": "2048 prompts x 8 gens (paper: 2048 x 16), 7400 steps", "full_param": True, "split": "2+6", "extra_flags": PAPER, "spawned": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "legs": [{"leg": 1, "train": t, "eval": e, "step_offset": 0, "policy": POLICY0, "spawned": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}],
              "recipe": "ScaleRL-paper optimizer on the re-cut-bank full-param init: lr 5e-7 const after 100-step warmup, AdamW eps 1e-15, wd 0.01, 8x2048, 2000 steps; 24 h legs resumed from the latest full-model ckpt (AdamW moments reset per leg)"}
    save_ids(d); log(f"LEG 1 spawned: train {t} eval {e}"); subprocess.run(["notify-discord", f"7400-step paper-optimizer arm LAUNCHED (leg 1): {NAME} lr 5e-7 / warmup 100 / eps 1e-15 / wd .01 / 8x2048 on {APP}; train {t}"])
while True:
    d = ids(); leg = d[KEY]["legs"][-1]; fc = modal.FunctionCall.from_id(leg["train"])
    try:
        r = fc.get(timeout=0); log(f"leg {leg['leg']} call returned: {str(r)[:200]}")
    except TimeoutError:
        time.sleep(300); continue
    except Exception as ex:
        log(f"leg {leg['leg']} call ENDED: {type(ex).__name__}: {str(ex)[:300]}")
    last = latest_ckpt(); log(f"latest SAVE_DONE ckpt: {last}")
    if last is None:
        subprocess.run(["notify-discord", f"paper-optimizer arm: leg {leg['leg']} ended with NO checkpoint — not respawning; check `modal app logs {APP}`"]); log("no ckpt -> stop"); break
    if last >= TOTAL:
        subprocess.run(["notify-discord", f"paper-optimizer arm DONE: {NAME} reached step {last}/{TOTAL}"]); log("DONE"); break
    try: modal.FunctionCall.from_id(leg["eval"]).cancel(terminate_containers=True)
    except Exception as ex: log(f"eval cancel: {ex}")
    wid = wandb_id(); t, e = spawn_leg(last, f"{SAVE}/step_{last}", wid)
    d[KEY]["legs"].append({"leg": leg["leg"] + 1, "train": t, "eval": e, "step_offset": last, "policy": f"{SAVE}/step_{last}", "wandb_id": wid, "spawned": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    d[KEY]["train"], d[KEY]["eval"] = t, e; save_ids(d)
    log(f"LEG {leg['leg'] + 1} spawned from step_{last}: train {t} eval {e} wandb {wid}")
    subprocess.run(["notify-discord", f"paper-optimizer arm: leg {leg['leg'] + 1} respawned from step_{last} (24 h cap); train {t}"])
    time.sleep(600)

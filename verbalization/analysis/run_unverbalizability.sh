#!/usr/bin/env bash
# The 27B unverbalizability argument, end to end, on ONE criterion (criterion.py): a feature fails
# when none of its sampled texts clears the SAE gate (tau = 1.5846); dead features are excluded.
#
#   bash verbalization/analysis/run_unverbalizability.sh <scratch>
#
# <scratch> holds what is not in the repo: examples_{2k,rwtest2k}.jsonl (corpus windows), p512/ (the
# 512 set's sae_self + LLM scores) and ws/ (per-token sae_self for how_structural_pass).
set -euo pipefail
S=${1:?scratch dir}
A=verbalization/analysis
D=verbalization/report/data
R=verbalization/report
M=$D/sae_match_27b.npz
cd "$(dirname "$0")/../.."
export PYTHONPATH=$A

echo "## 1. rarity"
python $A/plot_8b_rarity.py --perdir 2k=$D/perdir_27b_rl-last16.json --sae-match $M --criteria gate --drop-dead --stem fig_27b_rarity_gate_2k --out $R
python $A/plot_8b_rarity.py --perdir rwtest2k=$D/perdir_27b_rl-last16_rwtest2k.json --sae-match $M --criteria gate --drop-dead --stem fig_27b_rarity_gate_rwtest2k --out $R

echo "## 2. training"
python $A/arm_compare.py --before $D/perdir_27b_rl-last16.json --after $D/perdir_27b_armA_2k.json --sae-match $M --out $D/gate_armA_compare_2k.json
python $A/arm_compare.py --before $D/perdir_27b_rl-last16_rwtest2k.json --after $D/perdir_27b_armA_rwtest2k.json --sae-match $M --out $D/gate_armA_compare_rwtest2k.json
python $A/arm_compare.py --before $D/perdir_27b_rl-last16.json --after $D/perdir_27b_armB_2k.json --sae-match $M --clusters $D/clusters_27b_2k.jsonl --out $D/gate_armB_compare_2k.json
python $A/arm_compare.py --before $D/perdir_27b_rl-last16.json --after $D/perdir_27b_armB_replay_2k.json --sae-match $M --clusters $D/clusters_27b_2k.jsonl --out $D/gate_armB_replay_compare_2k.json

echo "## 3. topic clusters"
for k in 8 16; do python $A/kind_vs_rarity.py --perdir $D/perdir_27b_rl-last16_rwtest2k.json --sae-match $M --kinds $D/kinds_27b_rwtest2k_k$k.jsonl --examples $S/examples_rwtest2k.jsonl --out $D/gate_kind_vs_rarity_rwtest2k_k$k.json; done
for k in 64 128; do python $A/unverbalizable_groups.py --perdir $D/perdir_27b_rl-last16_rwtest2k.json --sae-match $M --kinds $D/kinds_27b_rwtest2k_k$k.jsonl --examples $S/examples_rwtest2k.jsonl --out $D/gate_groups_rwtest2k_k$k.json; done

echo "## 4. mechanisms and structural tokens"
for s in 2k:perdir_27b_rl-last16.json:examples_2k rwtest2k:perdir_27b_rl-last16_rwtest2k.json:examples_rwtest2k; do
  IFS=: read t pd ex <<< "$s"
  python $A/mechanism_groups.py --perdir $D/$pd --mechanics $D/mechanics_27b_$t.jsonl --sae-match $M --out $D/gate_mechanism_groups_$t.json
  python $A/structural_tags.py --perdir $D/$pd --mechanics $D/mechanics_27b_$t.jsonl --sae-match $M --examples $S/$ex.jsonl --out $D/gate_structural_tags_$t.json
done
python $A/how_structural_pass.py $S/ws

echo "## 5. blind categories, and MAEMM vs LLM"
python $A/label_categories.py set --perdir $D/perdir_27b_rl-last16.json --labels $D/labels_27b_2k.jsonl --sae-match $M --out $D/gate_label_categories_2k.json
python $A/label_categories.py set --perdir $D/perdir_27b_rl-last16_rwtest2k.json --labels $D/labels_27b_rwtest2k.jsonl --sae-match $M --out $D/gate_label_categories_rwtest2k.json
python $A/label_categories.py p512 --mirror $S/p512 --labels $D/labels_27b_512.jsonl --mechanics $D/mechanics_27b_512.jsonl --diversity $D/diversity_27b_512.json --sae-match $M --out $D/gate_label_categories_512.json

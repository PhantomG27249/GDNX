#!/bin/bash
# GDN3 auto-research loop — restart-based, stateless-agent + disk-ledger.
# Each iteration is ONE bounded `pi -p` turn that runs exactly one experiment,
# reads/writes only the research/ ledger, and exits. Because state lives on disk
# (leaderboard.jsonl + research_log.md), pi's context compaction never matters and
# a crash/OOM/hang only costs one turn.
#
# Start:  nohup bash research/run_auto_research.sh > research/loop.out 2>&1 &
# Stop:   touch research/STOP     (checked between turns)   or kill the process.
set -u
cd /home/dev/gdn3_two_timescale_release
export PATH="/home/dev/.local/share/pi-node/node-v22.23.1-linux-x64/bin:$PATH"  # where the real `pi` launcher lives
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1   # pin all proxy training to physical GPU1 (GPU0 drives the display).
                                # In-process this remaps to cuda:0, so any --device the agent passes lands on GPU1.

PROMPT="Do ONE bounded unit of GDN3 auto-research this turn, per research/RESEARCH.md. \
First skim research/leaderboard.jsonl (and current_task.md if it exists) so you don't repeat work. \
Usually the unit is one experiment: pick the next config by hypothesis — NEVER repeat a config \
already in the leaderboard — run the proxy, then append the result line to research/leaderboard.jsonl \
(REQUIRED). Update research_log.md ONLY for a new best / surprise / phase change — no per-turn prose. \
If mid multi-turn task (kernel/math), do the next concrete step and update current_task.md. Escalate \
phases rather than idle or repeat. Then STOP — do not start a second unit."

i=0
while true; do
  [ -f research/STOP ] && { echo "STOP file present — exiting loop @ $(date -Is)"; break; }
  i=$((i+1))
  echo "===== turn $i @ $(date -Is) =====" >> research/loop.log
  timeout 4200 pi -p --session-id gdn3-research --approve \
    --provider glm --model "canada-quant/GLM-5.2-W4A16-MTP" \
    --tools bash,read,write,edit \
    "$PROMPT" >> research/loop.log 2>&1
  echo "----- turn $i done rc=$? @ $(date -Is) -----" >> research/loop.log
  sleep 10
done

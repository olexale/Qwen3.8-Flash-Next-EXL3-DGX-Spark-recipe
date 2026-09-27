#!/usr/bin/env bash
# Real-traffic A/B of an engine setting: one pi_sessions.py pass per arm, each on a freshly
# restarted server, then the per-phase comparison. Run on the Mac (it has pi); the server is
# restarted over ssh. Nothing else should use the server meanwhile. At the end the server is
# restarted with RESTORE_ENV.
#
#   docker/tabbyapi/tools/pi_sessions_ab.sh "EXL3_PLD=0" "EXL3_PLD=1"
#   JOBS=3 TASKS=review,review,review docker/tabbyapi/tools/pi_sessions_ab.sh "EXL3_PLD=0" "EXL3_PLD=1"
#
# Arm A is the first env string (usually the candidate off), B the second. BASE_ENV is added
# to both and to the restore. The report prints tok/s per phase with bootstrap CIs, then ms per
# round (the setting's cost) and tokens per round (its benefit): one pass moves tok/s by up to
# ~8% with what the model happens to write, so a cost change shows in ms/round long before it
# shows in tok/s.
set -u
A_ENV=${1:?usage: $0 "ARM_A_ENV" "ARM_B_ENV"}
B_ENV=${2:?usage: $0 "ARM_A_ENV" "ARM_B_ENV"}
HOST=${HOST:-gx10-b2fe.local}
REPO=${REPO:-dev/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe}
PORT=${PORT:-18300}
BASE_ENV=${BASE_ENV:-EXL3_PREFIX_DIAG=2}
RESTORE_ENV=${RESTORE_ENV:-}
JOBS=${JOBS:-1}
TASKS=${TASKS:-review}
TOOLS=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HOME/scratch/pi-sessions/ab-$(date -u +%Y%m%dT%H%M)}
mkdir -p "$OUT"

restart() {  # $1 env, $2 name
  ssh "$HOST" "cd ~/$REPO && ./stop_tabby.sh && env $1 ./start_tabby.sh" > "$OUT/restart-$2.log" 2>&1
  for _ in $(seq 60); do
    if curl -s -m 5 "http://$HOST:$PORT/v1/models" | grep -q '"id"'; then
      echo "$(date -u +%T)Z server up ($2: $1)"; return 0
    fi
    sleep 5
  done
  echo "$(date -u +%T)Z server did not come up ($2), see $OUT/restart-$2.log"; return 1
}

for arm in A B; do
  if [ $arm = A ]; then E=$A_ENV; else E=$B_ENV; fi
  restart "$BASE_ENV $E" $arm || exit 1
  python3 "$TOOLS/pi_sessions.py" run --label $arm --jobs "$JOBS" --tasks "$TASKS" --out "$OUT"
  ssh "$HOST" 'docker logs qwen38-tabby 2>&1' > "$OUT/tabby-$arm.log"
done
restart "$BASE_ENV $RESTORE_ENV" restore

cat "$OUT"/tabby-A.log "$OUT"/tabby-B.log \
  | python3 "$TOOLS/pi_sessions.py" lines --manifest "$OUT/manifest.jsonl" > "$OUT/decode-stats.log"
python3 "$TOOLS/decode_report.py" "$OUT/decode-stats.log" | tee "$OUT/report.txt"
python3 - "$OUT/decode-stats.log" "$TOOLS" <<'EOF' | tee -a "$OUT/report.txt"
import sys
sys.path.insert(0, sys.argv[2])
from decode_report import parse_line, PHASES
rs = [r for r in map(parse_line, open(sys.argv[1])) if r]
for arm in "AB":
    x = [r for r in rs if r["arm"] == arm]
    total = sum(sum(r["time"].values()) for r in x) or 1
    for ph in PHASES:
        t = sum(r["time"][ph] for r in x)
        n = sum(r["cnt"].get(ph, {}).get("rounds", 0) for r in x)
        k = sum(r["tok"][ph] for r in x)
        if n:
            print(f"arm {arm} {ph:8} ms/round {1000 * t / n:5.1f}  tok/round {k / n:4.2f}  "
                  f"share of decode time {100 * t / total:4.1f}%")
EOF
echo "results in $OUT"

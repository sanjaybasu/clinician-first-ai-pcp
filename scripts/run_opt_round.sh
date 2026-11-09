#!/bin/zsh
# One optimization round on the nonconsensus HealthBench pool (never the Consensus subset). Resumable.
# Usage: scripts/run_opt_round.sh <round name> <n conversations> <configuration ...>
set -e
cd ~/waymark-local/packaging/compass-pcp
source scripts/env.sh
scripts/ensure_kev.sh
R=${1:?round}; N=${2:?n}; shift 2
OUT=$COMPASS_PCP_DATA/opt_$R
[ "$R" = r1 ] && OUT=$COMPASS_PCP_DATA/opt_pool  # round 1 predates per-round folders
EXCL=()  # every conversation used in an earlier round
for f in $COMPASS_PCP_DATA/opt_pool/raw_llm.json $COMPASS_PCP_DATA/opt_r*/raw_llm.json(N); do
  [ -f "$f" ] && [ "$(dirname $f)" != "$OUT" ] && EXCL+=("$f")
done
EXARGS=(); (( ${#EXCL} )) && EXARGS=(--exclude-from $EXCL)
echo "== $(date '+%F %T') $R n=$N $@ (excluding ${#EXCL} earlier rounds)"
caffeinate -i .venv/bin/python scripts/run_healthbench.py --subset nonconsensus --split dev --repeats 1 --n $N --threads 8 \
  --grader openai:gpt-4.1-2025-04-14 --configs "$@" --allow-unsigned-scripts $EXARGS \
  --backbone anthropic:claude-opus-5-5 --deliberate anthropic:claude-opus-5-5 --second google:gemini-3.8-flash \
  --out $OUT
echo "== $(date '+%F %T') $R done"

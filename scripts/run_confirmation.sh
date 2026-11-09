#!/bin/zsh
# Pre-registered confirmation (DECISIONS.md, optimization protocol step 5): all 743 Consensus development conversations,
# one reply each, the last-round configuration against raw Opus under the shared output policy. Resumable.
set -e
cd ~/waymark-local/packaging/compass-pcp
source scripts/env.sh
scripts/ensure_kev.sh
echo "== $(date '+%F %T') confirmation"
caffeinate -i .venv/bin/python scripts/run_healthbench.py --subset consensus --split dev --repeats 1 --threads 8 \
  --grader openai:gpt-4.1-2025-04-14 --configs raw_llm ai_pcp_v3 --allow-unsigned-scripts \
  --backbone anthropic:claude-opus-5-5 --deliberate anthropic:claude-opus-5-5 --second google:gemini-3.8-flash \
  --out $COMPASS_PCP_DATA/confirmation_dev743
echo "== $(date '+%F %T') confirmation done"

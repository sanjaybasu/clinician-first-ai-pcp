# Source this to set up a compass-pcp evaluation shell. Holds no secrets. Keys already in the environment are kept;
# otherwise they are read from the files where they live on the Project Lead's machine, when those files exist.
# Usage: source scripts/env.sh
set -a
[ -z "$GOOGLE_API_KEY" ] && [ -f ~/waymark-local/packaging/ai_compass/.env ] && \
  eval "$(grep -E '^(export )?GOOGLE_API_KEY=' ~/waymark-local/packaging/ai_compass/.env | sed 's/^export //')"
[ -z "$OPENAI_API_KEY" ] && [ -f ~/.config/ebm-verify/secrets.env ] && \
  eval "$(grep -E '^(export )?OPENAI_API_KEY=' ~/.config/ebm-verify/secrets.env | sed 's/^export //')"
[ -z "$ANTHROPIC_API_KEY" ] && ANTHROPIC_API_KEY="$(zsh -ic 'printf %s "$ANTHROPIC_API_KEY"' 2>/dev/null)"
COMPASS_PCP_DATA="${COMPASS_PCP_DATA:-$HOME/waymark-local/packaging/compass-pcp-data}"
SIMPLE_EVALS_PARENT="${SIMPLE_EVALS_PARENT:-$COMPASS_PCP_DATA/vendor/se}"
ANCHOR_REPO="${ANCHOR_REPO:-$HOME/waymark-local/packaging/anchor}"
KEV_DIR="${KEV_DIR:-$HOME/waymark-local/packaging/omission-risk-set-nba/tools/kev}"
set +a
for v in GOOGLE_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY; do [ -n "${(P)v}" ] || echo "env.sh: $v is not set" >&2; done
true

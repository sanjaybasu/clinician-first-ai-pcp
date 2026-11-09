#!/bin/zsh
# Fetch the two third-party checkouts the evaluation needs, at the commits the freeze was built with.
#   simple-evals (HealthBench grader and data loader): 652c89d0ca9df547706735883097e9537d40dc47
#   kev (local System-1 judge, runs on Apple silicon via MLX): 84847f0a883d900f7de5b7a57eaa341ca7f9a6b4
# Usage: COMPASS_PCP_DATA=<data dir> scripts/setup_vendor.sh
set -e
D=${COMPASS_PCP_DATA:-~/waymark-local/packaging/compass-pcp-data}
mkdir -p $D/vendor/se
[ -d $D/vendor/simple-evals ] || git clone https://github.com/openai/simple-evals.git $D/vendor/simple-evals
git -C $D/vendor/simple-evals checkout -q 652c89d0ca9df547706735883097e9537d40dc47
[ -e $D/vendor/se/simple_evals ] || ln -s $D/vendor/simple-evals $D/vendor/se/simple_evals
[ -d $D/vendor/kev ] || git clone https://github.com/jaredpalmer/kev.git $D/vendor/kev
git -C $D/vendor/kev checkout -q 84847f0a883d900f7de5b7a57eaa341ca7f9a6b4
echo "set KEV_DIR=$D/vendor/kev and COMPASS_PCP_DATA=$D before sourcing scripts/env.sh"

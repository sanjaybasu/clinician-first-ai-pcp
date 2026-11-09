#!/bin/zsh
# Export the frozen commit, code and protocol only, to a public repository on the sanjaybasu account.
# Usage: scripts/export_public_repo.sh <frozen commit> <output dir>
# Leaves out: the Compass patch (private code), the data management plan (internal infrastructure), the Cohort A split
# manifest (message ids), and local run scripts with machine paths. Fails if the leak scan finds anything.
set -e
COMMIT=${1:?frozen commit}; OUT=${2:?output dir}
cd "$(dirname "$0")/.."
rm -rf "$OUT"; mkdir -p "$OUT"
git archive "$COMMIT" | tar -x -C "$OUT"
rm -rf "$OUT/patches" "$OUT/docs/regulatory/DMP-PCP-001_data_management_plan.md" \
       "$OUT/docs/regulatory/cohort_a_split_manifest.json" "$OUT/RESUME.md" "$OUT/scripts/resume_runs.sh" \
       "$OUT/scripts/run_powered_dev.sh" "$OUT/scripts/run_final_check.sh"
# Leak scan: emails other than the corresponding author's, phone numbers, message ids, internal hosts, machine paths.
if grep -rn -I -E "cm[a-z0-9]{23}|waymarkcare\.in|10\.104\.|/Users/[a-z]|\b[0-9]{3}[-.][0-9]{3}[-.][0-9]{4}\b" "$OUT"; then
  echo "leak scan failed"; exit 1
fi
if grep -rn -I -o -E "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(com|org|edu)" "$OUT" | grep -v -E "sanjay\.basu@(ucsf\.edu|waymarkcare\.com)"; then
  echo "leak scan failed: email"; exit 1
fi
echo "export clean: $OUT (from $COMMIT)"

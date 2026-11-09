"""Supervised in-distribution hazard screen: 5-fold CV on the Cohort A tuning half only.

Features: local sentence embeddings (bge-base, all-mpnet) of the patient message, optionally stacked with the
zero-shot Kev probabilities and Claude tier already cached for the same messages. Model: L2 logistic regression with
class weighting; out-of-fold probabilities give an honest AUROC on the tuning half. Everything runs on this machine.
The sealed test half is never read.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compass_pcp.harness.metrics import auroc, partial_auroc_high_sensitivity, specificity_at_sensitivity_floor

D = Path.home() / "waymark-local/packaging/compass-pcp-data"
CASES = Path.home() / "waymark-local/notebooks/compass-ai-pcp/golden_set/data/realworld_cases_n2000_clean.json"
tune_ids = set(json.loads((D / "cohort_a_split/cohort_a_tune_ids.json").read_text()))
cases = [c for c in json.loads(CASES.read_text()) if c["message_id"] in tune_ids]
y = np.array([int(c["detection_truth"]) for c in cases])
kev = {json.loads(l)["id"]: json.loads(l) for l in open(D / "hazard_cohortA_tune/per_case_kev.jsonl")}
cl = {json.loads(l)["id"]: json.loads(l) for l in open(D / "hazard_cohortA_tune/per_case_anthropic_claude-sonnet-5-5.jsonl")}
zs = np.array([[kev[c["message_id"]]["score_any"], kev[c["message_id"]]["score_urgent"], cl[c["message_id"]]["score_urgent"] / 4] for c in cases])

results = {}
for name in ("BAAI/bge-base-en-v1.5", "sentence-transformers/all-mpnet-base-v2"):
    emb = SentenceTransformer(name, device="cpu").encode([c["message"] or "" for c in cases], batch_size=32, normalize_embeddings=True)
    for feats, label in ((emb, "emb"), (np.hstack([emb, zs]), "emb+zeroshot")):
        oof = np.zeros(len(y))
        for seed in (0, 1, 2):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for tr, te in skf.split(feats, y):
                m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(feats[tr], y[tr])
                oof[te] += m.predict_proba(feats[te])[:, 1] / 3
        sc, yy = oof.tolist(), [bool(v) for v in y]
        op = specificity_at_sensitivity_floor(sc, yy, 0.90)
        results[f"{name.split('/')[-1]}|{label}"] = {
            "auroc": auroc(sc, yy), "pauroc_0.90": partial_auroc_high_sensitivity(sc, yy, 0.9),
            "spec_at_sens_0.90": op["specificity"],
        }
for name, sc in (("kev zero-shot", zs[:, 0]), ("claude zero-shot", zs[:, 2])):
    results[name] = {"auroc": auroc(sc.tolist(), [bool(v) for v in y]),
                     "pauroc_0.90": partial_auroc_high_sensitivity(sc.tolist(), [bool(v) for v in y], 0.9),
                     "spec_at_sens_0.90": specificity_at_sensitivity_floor(sc.tolist(), [bool(v) for v in y], 0.9)["specificity"]}
(D / "hazard_cohortA_tune/supervised_cv.json").write_text(json.dumps(results, indent=2))
for k, v in results.items():
    print(f"{k:45s} AUROC {v['auroc']:.3f}  pAUROC {v['pauroc_0.90']:.3f}  spec@sens0.90 {v['spec_at_sens_0.90']:.3f}")

from pathlib import Path
import json
import pandas as pd

ROOT = Path(r"C:\Users\LENOVO\Projects\tennis-predictor-cli")
MODELS = ROOT / "data" / "models"

BRANCHES = [
    {"tour": "atp", "source": "sackmann", "surface": "Hard"},
    {"tour": "atp", "source": "sackmann", "surface": "Clay"},
    {"tour": "atp", "source": "sackmann", "surface": "Grass"},
    {"tour": "atp", "source": "tml",      "surface": "Hard"},
    {"tour": "atp", "source": "tml",      "surface": "Clay"},
    {"tour": "atp", "source": "tml",      "surface": "Grass"},
    {"tour": "wta", "source": "sackmann", "surface": "Hard"},
    {"tour": "wta", "source": "sackmann", "surface": "Clay"},
    {"tour": "wta", "source": "sackmann", "surface": "Grass"},
]

def build_suffix(source: str, surface: str | None) -> str:
    parts = []
    if source != "sackmann":
        parts.append(source)
    if surface is not None:
        parts.append(surface.lower())
    return "_" + "_".join(parts) if parts else ""

def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def get_logit_test_ll(path: Path):
    payload = load_json(path)
    if payload is None:
        return None
    return float(payload["test"]["log_loss"])

def get_xgb_raw_test_ll(path: Path):
    payload = load_json(path)
    if payload is None:
        return None
    return float(payload["raw_test"]["log_loss"])

rows = []

for b in BRANCHES:
    tour = b["tour"]
    source = b["source"]
    surface = b["surface"]
    suffix = build_suffix(source, surface)

    logit_base_path = MODELS / f"{tour}_logit_baseline{suffix}_metrics.json"
    logit_tuned_path = MODELS / f"{tour}_logit_tuned{suffix}_metrics.json"
    xgb_base_path = MODELS / f"{tour}_xgb_baseline{suffix}_metrics.json"
    xgb_tuned_path = MODELS / f"{tour}_xgb_tuned{suffix}_metrics.json"

    logit_base_ll = get_logit_test_ll(logit_base_path)
    logit_tuned_ll = get_logit_test_ll(logit_tuned_path)
    xgb_base_raw_ll = get_xgb_raw_test_ll(xgb_base_path)
    xgb_tuned_raw_ll = get_xgb_raw_test_ll(xgb_tuned_path)

    logit_candidates = []
    if logit_base_ll is not None:
        logit_candidates.append(("logit_baseline", logit_base_ll))
    if logit_tuned_ll is not None:
        logit_candidates.append(("logit_tuned", logit_tuned_ll))

    xgb_candidates = []
    if xgb_base_raw_ll is not None:
        xgb_candidates.append(("xgb_baseline_raw", xgb_base_raw_ll))
    if xgb_tuned_raw_ll is not None:
        xgb_candidates.append(("xgb_tuned_raw", xgb_tuned_raw_ll))

    best_logit_model, best_logit_ll = min(logit_candidates, key=lambda x: x[1]) if logit_candidates else (None, None)
    best_xgb_model, best_xgb_ll = min(xgb_candidates, key=lambda x: x[1]) if xgb_candidates else (None, None)

    if best_logit_ll is None and best_xgb_ll is None:
        overall_winner = None
        overall_best_ll = None
    elif best_xgb_ll is None:
        overall_winner = best_logit_model
        overall_best_ll = best_logit_ll
    elif best_logit_ll is None:
        overall_winner = best_xgb_model
        overall_best_ll = best_xgb_ll
    elif best_logit_ll <= best_xgb_ll:
        overall_winner = best_logit_model
        overall_best_ll = best_logit_ll
    else:
        overall_winner = best_xgb_model
        overall_best_ll = best_xgb_ll

    rows.append({
        "tour": tour.upper(),
        "source": source,
        "surface": surface,
        "logit_baseline_test_ll": logit_base_ll,
        "logit_tuned_test_ll": logit_tuned_ll,
        "best_logit_model": best_logit_model,
        "best_logit_test_ll": best_logit_ll,
        "xgb_baseline_raw_test_ll": xgb_base_raw_ll,
        "xgb_tuned_raw_test_ll": xgb_tuned_raw_ll,
        "best_xgb_model": best_xgb_model,
        "best_xgb_raw_test_ll": best_xgb_ll,
        "overall_precal_winner": overall_winner,
        "overall_best_test_ll": overall_best_ll,
    })

df = pd.DataFrame(rows)
out_path = MODELS / "precalibration_branch_winners.csv"
df.to_csv(out_path, index=False)

print(df.to_string(index=False))
print()
print(f"Saved to: {out_path}")

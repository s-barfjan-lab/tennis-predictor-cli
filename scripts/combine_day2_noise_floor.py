from pathlib import Path
import pandas as pd

models_dir = Path(r"C:\Users\LENOVO\Projects\tennis-predictor-cli\data\models")

files = [
    "atp_xgb_tuned_noise_floor_hard_summary.csv",
    "atp_logit_noise_floor_clay_summary.csv",
    "atp_xgb_noise_floor_grass_summary.csv",
    "atp_xgb_noise_floor_tml_hard_summary.csv",
    "atp_xgb_noise_floor_tml_clay_summary.csv",
    "atp_xgb_noise_floor_tml_grass_summary.csv",
    "wta_logit_noise_floor_hard_summary.csv",
    "wta_xgb_noise_floor_clay_summary.csv",
    "wta_logit_noise_floor_grass_summary.csv",
]

rows = []

for fname in files:
    path = models_dir / fname
    df = pd.read_csv(path)

    metric_map = {row["metric"]: row for _, row in df.iterrows()}

    branch_name = fname.replace("_noise_floor_summary.csv", "").replace(".csv", "")

    rows.append({
        "branch": branch_name,
        "test_log_loss_mean": metric_map["test_log_loss"]["mean"],
        "test_log_loss_std": metric_map["test_log_loss"]["std"],
        "test_log_loss_min": metric_map["test_log_loss"]["min"],
        "test_log_loss_max": metric_map["test_log_loss"]["max"],
        "test_accuracy_mean": metric_map["test_accuracy"]["mean"],
        "test_accuracy_std": metric_map["test_accuracy"]["std"],
        "test_roc_auc_mean": metric_map["test_roc_auc"]["mean"],
        "test_roc_auc_std": metric_map["test_roc_auc"]["std"],
    })

out = pd.DataFrame(rows)
out_path = models_dir / "day2_noise_floor_combined.csv"
out.to_csv(out_path, index=False)

print(out.to_string(index=False))
print()
print(f"Saved to: {out_path}")

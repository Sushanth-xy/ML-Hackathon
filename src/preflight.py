"""Checks that everything tune_decoder.py needs is in place. Run: python src/preflight.py"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ok = True


def check(cond, msg):
    global ok
    print(("OK    " if cond else "FAIL  ") + msg)
    ok = ok and cond


check((ROOT / "src" / "hmm.py").is_file(), "src/hmm.py exists")
try:
    sys.path.insert(0, str(ROOT / "src"))
    import hmm
    check(all(hasattr(hmm, n) for n in ["build_network", "reachable", "decode", "DEFAULT_CONFIG"]),
          "hmm.py defines build_network, reachable, decode, DEFAULT_CONFIG")
except Exception as e:
    check(False, f"hmm.py imports cleanly ({e})")

try:
    import optuna, pandas  # noqa
    check(True, "optuna and pandas installed")
except ImportError as e:
    check(False, f"missing package: {e}  ->  pip install optuna pandas")

check(any((ROOT / "data").glob("*train_candidates*.csv")), "data/train_candidates.csv")
check(any((ROOT / "data").glob("*student*candidates*.csv")), "data/student_test_candidates.csv")
check((ROOT / "results" / "models" / "preds_best.csv").is_file(), "results/models/preds_best.csv")
check(any((ROOT / "Student folder").rglob("*.arcs")), "raw .arcs files under 'Student folder'")
check((ROOT / "results" / "hmm").is_dir(), "results/hmm is a FOLDER")
check((ROOT / "artifacts").is_dir(), "artifacts/ is a folder")

# old-script check
td = (ROOT / "training" / "tune_decoder.py").read_text(encoding="utf-8", errors="ignore")
check("read_parquet" not in td and "src.metrics" not in td,
      "tune_decoder.py is the NEW version (no read_parquet / src.metrics)")

print("\nALL GOOD, run tune_decoder.py" if ok else "\nFix the FAIL lines above first.")
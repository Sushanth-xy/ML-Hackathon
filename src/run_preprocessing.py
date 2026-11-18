"""
One command to do the whole preprocessing part:
    python src/run_preprocessing.py                      # auto-finds the dataset
    python src/run_preprocessing.py "C:/path/to/Student folder"
    python src/run_preprocessing.py --eda-only           # skip rebuilding the CSVs

Steps:  1) build_dataset  -> data/*.csv           (candidate table + labels + features)
        2) eda            -> results/eda/*        (tables, plots, Folium maps)
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset, eda

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = args[0] if args else None
    t = time.time()
    if "--eda-only" not in sys.argv:
        print("=== STEP 1/2: building candidate dataset ==="); build_dataset.main(root)
    print("\n=== STEP 2/2: EDA and quality evidence ==="); eda.main(root)
    print(f"\nAll done in {(time.time()-t)/60:.1f} min.")
    print("Check: data/ (3 csv files) and results/eda/ (csv, png, html maps)")
"""Download atlases from CZ CELLxGENE Discover by dataset version id (Extended Data Table 8).
Usage: python download.py <atlas name | all> [--out data/]"""
import argparse, csv, os, subprocess
ap = argparse.ArgumentParser(); ap.add_argument("atlas"); ap.add_argument("--out", default="data"); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); here = os.path.dirname(os.path.abspath(__file__))
for r in csv.DictReader(open(os.path.join(here, "atlases.csv"))):
    if a.atlas in ("all", r["atlas"]):
        url = f"https://datasets.cellxgene.cziscience.com/{r['cellxgene_dataset_version_id']}.h5ad"; dst = os.path.join(a.out, r["atlas"] + ".h5ad")
        print(f"{r['atlas']}: {url} -> {dst}", flush=True)
        subprocess.run(["curl", "-L", "-C", "-", "-o", dst, url], check=True)

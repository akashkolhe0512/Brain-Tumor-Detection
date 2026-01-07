# 00_inspect_structure.py
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm
import argparse

def scan_dataset(root: Path, dataset_id: str) -> pd.DataFrame:
    rows = []
    # classes = immediate subfolders containing images
    for cls_dir in sorted([p for p in root.rglob("*") if p.is_dir() and any(p.glob("*.jpg"))]):
        label = cls_dir.name
        for f in cls_dir.glob("*.jpg"):
            if not f.is_file():
                continue
            rows.append({
                "dataset_id": dataset_id,
                "class_raw": label,
                "path": str(f.resolve())
            })
    return pd.DataFrame(rows)

def image_info_safe(path: Path):
    try:
        with Image.open(path) as im:
            w, h = im.size
            mode = im.mode
            fmt = im.format  # e.g. "JPEG"
        return fmt, w, h, mode, None
    except Exception as e:
        return None, None, None, None, str(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_dir", default="Datasets", type=str, help="Root folder containing DatasetA, DatasetB")
    args = ap.parse_args()

    root = Path(args.datasets_dir)
    subroots = [p for p in root.iterdir() if p.is_dir()]

    dfs = []
    for sr in subroots:
        df = scan_dataset(sr, dataset_id=sr.name)
        dfs.append(df)

    if not dfs:
        print("No datasets found.")
        return
    meta = pd.concat(dfs, ignore_index=True)

    # Add image info
    infos = []
    for p in tqdm(meta["path"], desc="Probing images"):
        fmt, w, h, mode, err = image_info_safe(Path(p))
        infos.append({"format": fmt, "width": w, "height": h, "mode": mode, "error": err})
    info_df = pd.DataFrame(infos)
    meta = pd.concat([meta, info_df], axis=1)

    out_dir = Path("artifacts"); out_dir.mkdir(exist_ok=True)
    meta.to_csv(out_dir / "manifest_raw.csv", index=False)

    # Folder structure summary
    print("\n=== Folder Structure (dataset_id / class_raw -> count) ===")
    counts = meta.groupby(["dataset_id", "class_raw"]).size().reset_index(name="count")
    print(counts.to_string(index=False))

    # Errors/corrupt files
    bad = meta[meta["error"].notna()]
    if len(bad):
        print("\n=== Potentially corrupt/unreadable images ===")
        print(bad[["path","error"]].head(20).to_string(index=False))
    else:
        print("\nNo corrupt images detected by PIL open().")

    # Basic size stats
    size_stats = meta.dropna(subset=["width","height"]).assign(
        megapixels=lambda d: (d["width"]*d["height"])/1e6
    ).groupby(["dataset_id","class_raw"]).agg(
        n=("path","count"),
        w_med=("width","median"),
        h_med=("height","median"),
        mp_med=("megapixels","median")
    ).reset_index()
    print("\n=== Image size stats (median) ===")
    print(size_stats.to_string(index=False))

    print(f"\nSaved raw manifest to: {out_dir / 'manifest_raw.csv'}")

if __name__ == "__main__":
    main()
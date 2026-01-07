# 01_merge_clean_split.py
from pathlib import Path
import pandas as pd
import hashlib, shutil, re
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
import argparse

# Manual mapping of dataset folder names → unified labels
CLASS_NAME_MAP = {
    "glioma_tumor": "glioma",
    "glioma": "glioma",
    "meningioma_tumor": "meningioma",
    "meningioma": "meningioma",
    "pituitary_tumor": "pituitary",
    "pituitary": "pituitary",
    "no_tumor": "no_tumor",
    "notumor": "no_tumor",
}

def sha256sum(path: Path, chunk=1<<20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def load_manifest_raw(path="artifacts/manifest_raw.csv"):
    df = pd.read_csv(path)
    # Filter out unreadable or tiny images (<64px min dimension)
    df = df[df["error"].isna()].copy()
    df["min_side"] = df[["width","height"]].min(axis=1)
    df = df[df["min_side"] >= 64]
    return df

def derive_patient_group(filename: str) -> str:
    """
    Optional: group images to reduce leakage.
    If no patient IDs exist, this just groups by base filename prefix.
    """
    name = Path(filename).stem.lower()
    m = re.search(r"(patient|pat|sub|p)[_-]?(\d{1,5})", name)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return name.split("_")[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data_merged", type=str)
    ap.add_argument("--manifest_raw", default="artifacts/manifest_raw.csv", type=str)
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(exist_ok=True, parents=True)

    df = load_manifest_raw(args.manifest_raw)

    # Map to unified labels
    df["label"] = df["class_raw"].map(CLASS_NAME_MAP)
    if df["label"].isna().any():
        print("⚠️ Some classes not mapped! Please update CLASS_NAME_MAP.")
        print(df[df["label"].isna()]["class_raw"].unique())
        return

    # Deduplicate by hash
    print("Hashing images for deduplication...")
    df["sha256"] = [sha256sum(Path(p)) for p in tqdm(df["path"], desc="Hashing")]
    before = len(df)
    df = df.drop_duplicates(subset=["sha256"])
    print(f"Deduplicated: {before} -> {len(df)}")

    # Add grouping (for stratified split)
    df["group"] = df["path"].apply(derive_patient_group)

    # Stratified Group K-Fold split (train/val/test = 80/10/10)
    X, y, groups = df["path"], df["label"], df["group"]

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    train_idx, temp_idx = list(sgkf.split(X, y, groups))[0]

    df_train = df.iloc[train_idx].copy()
    df_temp = df.iloc[temp_idx].copy()

    sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=args.seed)
    val_idx, test_idx = list(sgkf2.split(df_temp["path"], df_temp["label"], df_temp["group"]))[0]

    df_val = df_temp.iloc[val_idx].copy()
    df_test = df_temp.iloc[test_idx].copy()

    print("Split sizes:", len(df_train), len(df_val), len(df_test))

    # Create folder structure
    for split_name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
        for lab in CLASS_NAME_MAP.values():
            (out_root / split_name / lab).mkdir(parents=True, exist_ok=True)

        print(f"Copying {split_name} files...")
        for _, row in tqdm(d.iterrows(), total=len(d)):
            src = Path(row["path"])
            lab = row["label"]
            dst = out_root / split_name / lab / f"{row['sha256']}{src.suffix.lower()}"
            if not dst.exists():
                shutil.copy2(src, dst)

    # Save manifest
    merged = pd.concat([
        df_train.assign(split="train"),
        df_val.assign(split="val"),
        df_test.assign(split="test"),
    ], ignore_index=True)
    out_manifest = Path("artifacts") / "manifest_merged.csv"
    merged.to_csv(out_manifest, index=False)
    print(f"\n[OK] Saved merged manifest: {out_manifest}")

    print(f" Merged dataset ready in: {out_root}")

if __name__ == "__main__":
    main()
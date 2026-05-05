"""
setup_data.py — Build a local training CSV from on-disk Coswara data.

Usage:
    python3 ml_service/setup_data.py [--coswara_dir PATH] [--out PATH]

Outputs:
    ml_service/local_data.csv  (disease, audio_path, source columns)
"""
import os
import argparse
import pandas as pd

COSWARA_DEFAULT = os.path.expanduser(
    "~/Downloads/TRANSFER/cough-ai-expo/Coswara-Data"
)
OUT_DEFAULT = os.path.join(os.path.dirname(__file__), "local_data.csv")

COVID_STATUSES = {"positive_mild", "positive_moderate", "positive_asymp"}
COUGH_TYPES = {"cough-heavy", "cough-shallow"}


def build_coswara_csv(coswara_dir: str) -> pd.DataFrame:
    meta_path = os.path.join(coswara_dir, "combined_data.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    meta = pd.read_csv(meta_path)

    def map_label(status):
        if status == "healthy":
            return "healthy"
        if status in COVID_STATUSES:
            return "covid"
        return None

    meta["label"] = meta["covid_status"].apply(map_label)
    meta = meta[meta["label"].notna()]
    id_to_label = dict(zip(meta["id"], meta["label"]))

    audio_dir = os.path.join(coswara_dir, "all_audio")
    rows = []
    for fname in sorted(os.listdir(audio_dir)):
        if not fname.endswith(".wav"):
            continue
        parts = fname.rsplit("_", 1)
        if len(parts) != 2:
            continue
        pid, ctype = parts[0], parts[1].replace(".wav", "")
        if ctype not in COUGH_TYPES:
            continue
        if pid not in id_to_label:
            continue
        rows.append({
            "disease": id_to_label[pid],
            "audio_path": os.path.join(audio_dir, fname),
            "source": "coswara",
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coswara_dir", default=COSWARA_DEFAULT)
    parser.add_argument("--out", default=OUT_DEFAULT)
    args = parser.parse_args()

    print(f"Scanning Coswara: {args.coswara_dir}")
    df = build_coswara_csv(args.coswara_dir)

    print(f"\nClass distribution:")
    print(df["disease"].value_counts().to_string())
    print(f"\nTotal samples: {len(df)}")

    # Verify all files exist
    missing = [r for r in df["audio_path"] if not os.path.exists(r)]
    if missing:
        print(f"\nWARNING: {len(missing)} files missing — removing from CSV")
        df = df[df["audio_path"].apply(os.path.exists)]

    df.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()

"""Collect block activations over the labelled jaguar Re-ID images.

The PNGs are RGBA cut-outs whose alpha is a binary mask over black: each is cropped
to its mask, squashed to a square with PIL's resize and composited onto black.
Identities come from train.csv, sexes from data/jaguar_sex_map.csv.
"""

import argparse
import csv
from pathlib import Path
import unicodedata

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.collect import add_collect_args, collect_activations
from src.data.features import log

SEXES = ["Male", "Female"]

#: train.csv spellings the sex map does not carry; confirmed typos, both female.
ALIASES = {"lua": "luna", "estella": "estela"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="Directory holding train/ and train.csv.")
    p.add_argument("--sex-map", default="data/jaguar_sex_map.csv", help="CSV with Name,Gender.")
    add_collect_args(p)
    return p.parse_args()


def _norm_name(s: str) -> str:
    """Fold accents and drop a parenthesised alias so train.csv and the sex map meet."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()
    s = s.split("(")[0].strip()
    return ALIASES.get(s, s)


def build_records(data_dir: Path, sex_map: Path) -> list[dict]:
    """One record per labelled image whose identity has a known sex."""
    with (data_dir / "train.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with sex_map.open(encoding="utf-8") as f:
        lookup = {_norm_name(r["Name"]): r["Gender"] for r in csv.DictReader(f)}

    records, dropped = [], {}
    for r in rows:
        name = r["ground_truth"]
        sex = lookup.get(_norm_name(name))
        if sex not in SEXES:
            dropped[name] = dropped.get(name, 0) + 1
            continue
        records.append({"path": str(data_dir / "train" / r["filename"]),
                        "identity": name, "sex": sex})
    log(f"dropped {sum(dropped.values())} images of identities without a sex: {dropped}")
    return records


class JaguarSet(Dataset):
    def __init__(self, records: list[dict], image_size: int) -> None:
        self.records = records
        self.image_size = image_size
        self.label_names = SEXES

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> tuple:
        record = self.records[i]
        img = Image.open(record["path"])
        img = img.crop(img.getchannel("A").getbbox()).convert("RGB")
        # PIL's own (bicubic) resize, not torchvision's: it is what the paper's
        # jaguar figure was computed with, and the two differ by ~1% in features.
        img = img.resize((self.image_size, self.image_size))
        return transforms.functional.to_tensor(img), SEXES.index(record["sex"]), record["path"]

    def class_distribution(self) -> dict[str, int]:
        return {sex: sum(r["sex"] == sex for r in self.records) for sex in SEXES}


def main() -> None:
    args = parse_args()
    records = build_records(Path(args.data_dir), Path(args.sex_map))
    log(f"{len(records)} images, {len({r['identity'] for r in records})} identities")

    result = collect_activations(args, JaguarSet(records, args.image_size), args.data_dir)
    identity_of = {r["path"]: r["identity"] for r in records}
    result["columns"] = {"identity": [identity_of[p] for p in result["paths"]]}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out_path)
    log(f"saved {result['meta']['num_samples']} samples -> {out_path.resolve()}")


if __name__ == "__main__":
    main()

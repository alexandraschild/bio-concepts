"""Collect block activations over a per-track sample of PolarBearVidID.

Frames are <id>/<id>C<cam>T<track>F<frame>.jpg; frames within a track are
near-duplicates, so only --per-track evenly spaced ones are kept. Every crop shows
the enclosure, so each row also records its zoo for leave-one-zoo-out CV.
"""

import argparse
import csv
from pathlib import Path
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.collect import add_collect_args, collect_activations, square_resize
from src.data.features import log

FRAME_RE = re.compile(r"^(\d+)C(\d+)T(\d+)F(\d+)\.jpg$", re.IGNORECASE)
SEXES = ["Male", "Female"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True,
                   help="Directory of <id>/ image folders, holding animal_db.csv.")
    p.add_argument("--per-track", type=int, default=8,
                   help="Evenly spaced frames kept per (id, track) (default: 8).")
    add_collect_args(p)
    return p.parse_args()


def scan_frames(data_dir: Path, per_track: int) -> list[dict]:
    """One record per kept frame, evenly spaced within each (id, track)."""
    records: list[dict] = []
    for id_dir in sorted(d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()):
        by_track: dict[int, list[dict]] = {}
        for img in id_dir.glob("*.jpg"):
            m = FRAME_RE.match(img.name)
            if m is None:
                continue
            bear, _cam, track, frame = (int(g) for g in m.groups())
            by_track.setdefault(track, []).append({"path": str(img), "bear": bear, "frame": frame})
        for _, frames in sorted(by_track.items()):
            frames.sort(key=lambda r: r["frame"])
            picks = np.linspace(0, len(frames) - 1, min(per_track, len(frames))).round().astype(int)
            records.extend(frames[i] for i in dict.fromkeys(picks.tolist()))
    return records


class FrameSet(Dataset):
    def __init__(self, records: list[dict], transform) -> None:
        self.records = records
        self.transform = transform
        self.label_names = SEXES

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> tuple:
        record = self.records[i]
        img = Image.open(record["path"]).convert("RGB")
        return self.transform(img), SEXES.index(record["sex"]), record["path"]

    def class_distribution(self) -> dict[str, int]:
        return {sex: sum(r["sex"] == sex for r in self.records) for sex in SEXES}


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    with (data_dir / "animal_db.csv").open(encoding="utf-8") as f:
        animals = {int(r["id"]): r for r in csv.DictReader(f)}

    records = scan_frames(data_dir, args.per_track)
    for r in records:
        animal = animals[r["bear"]]
        r.update(identity=animal["name"], zoo=animal["zoo"], sex=animal["sex"].strip().capitalize())
    log(f"{len(records)} frames, {len({r['bear'] for r in records})} bears across "
        f"{len({r['zoo'] for r in records})} zoos")

    result = collect_activations(args, FrameSet(records, square_resize(args.image_size)),
                                 args.data_dir)
    by_path = {r["path"]: r for r in records}
    result["columns"] = {"identity": [by_path[p]["identity"] for p in result["paths"]]}
    result["groups"] = {"zoo": [by_path[p]["zoo"] for p in result["paths"]]}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out_path)
    log(f"saved {result['meta']['num_samples']} samples -> {out_path.resolve()}")


if __name__ == "__main__":
    main()

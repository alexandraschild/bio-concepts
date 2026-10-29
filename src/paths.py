"""The run layout every stage derives its default paths from."""

import argparse
from dataclasses import dataclass
from pathlib import Path

#: An attribution method's CSV directory.
EVAL_DIRNAME = "eval"


@dataclass(frozen=True)
class Run:
    """results/<backbone>/<probe_id> and everything derived from that probe."""

    root: Path

    @property
    def probes(self) -> Path:
        return self.root / "probes"

    @property
    def split_file(self) -> Path:
        return self.probes / "split.json"

    @property
    def eval(self) -> Path:
        return self.root / "eval"

    @property
    def eval_holdout(self) -> Path:
        return self.eval / "holdout"

    def eval_transfer(self, backbone_tag: str) -> Path:
        return self.eval / f"transfer_{backbone_tag}"

    @property
    def eval_id_cv(self) -> Path:
        return self.eval / "id_cv"

    def attributions(self, method: str) -> Path:
        return self.root / "attributions" / method

    def attribution_eval(self, method: str) -> Path:
        return self.attributions(method) / EVAL_DIRNAME

    def xda(self, method: str) -> Path:
        return self.attributions(method) / "xda"

    @property
    def steering_embeddings(self) -> Path:
        return self.root / "steering" / "embeddings"

    @property
    def steering_controls(self) -> Path:
        """Control-direction sweeps: one subdirectory per direction, each with the
        embeddings and eval layout of a normal sweep."""
        return self.root / "steering" / "controls"

    @property
    def steering_eval(self) -> Path:
        return self.root / "steering" / "eval"


def add_run_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run", default=None, metavar="DIR",
        help="Run directory (results/<backbone>/<probe_id>). Every input and output of "
             "this stage defaults to its place inside it; explicit path flags override.",
    )


def run_of(args: argparse.Namespace) -> Run | None:
    return Run(Path(args.run)) if args.run else None


def split_path_for(probe_path: Path) -> Path:
    """The one manifest src.probes.train writes per probe directory."""
    return probe_path.parent / "split.json"

"""XDA: heatmaps for the image regions behind a data attribution.

Representer attribution is bilinear, alpha_i <x_i, x_t>, so seeding a cotangent
in a Gradient x Input framework cancels the pooled factor and yields one map per
side: u_test = alpha_i * x_i on the test image, u_train = alpha_i * x_t on the
training one. Relevance is AttnLRP with CP-LRP attention via lxt, read out as
grad x input on the pixels around the image's mean colour. Representer caches only.
"""

import argparse
import sys
import types
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# lxt.efficient's __init__ drags in transformers, whose huggingface-hub pin
# clashes with timm's. We always pass an explicit patch_map, so stub it out.
_stub = types.ModuleType("lxt.efficient.models")
_stub.get_default_map = lambda module: (_ for _ in ()).throw(
    ValueError("src.attribution.xda always passes an explicit patch_map")
)
sys.modules.setdefault("lxt.efficient.models", _stub)

from lxt.efficient.core import monkey_patch
from lxt.efficient.patches import (
    dropout_forward,
    layer_norm_forward,
    non_linear_forward,
    patch_method,
)
from lxt.efficient.rules import stop_gradient
from timm.layers.norm import LayerNorm as TimmLayerNorm
from timm.models.eva import (
    EvaAttention,
    apply_rot_embed_cat,
    maybe_add_mask,
    resolve_self_attn_mask,
)

from src.data.features import log
from src.attribution.cache import AttributionCache
from src.paths import add_run_arg, run_of
from src.data.collect import _load_local_model

# The published DualXDA layout, reused. Only display_img is not: it drops the
# result of img.permute(1, 2, 0), so _display_img puts the assignment back.
from dualxda import DualDA
from dualxda.utils import colourgradarrow
from zennit.image import imgify
from torchvision import transforms
import matplotlib

matplotlib.use("Agg")  # headless: cluster / no display
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

#: ViT patch size, for folding a per-pixel relevance map back onto the token grid.
PATCH = 16

#: Proponents and opponents drawn per side.
NSAMPLES = 3
OVERLAY_ALPHA = 0.4


# --------------------------------------------------------------------------- #
# AttnLRP instrumentation for the timm DINOv3 ViT
# --------------------------------------------------------------------------- #
def eva_attention_forward(self, x, rope=None, attn_mask=None, is_causal=False):
    """EvaAttention.forward with the CP-LRP rule: q and k are detached."""
    B, N, C = x.shape
    if self.qkv is not None:
        if self.q_bias is None:
            qkv = self.qkv(x)
        else:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
            if self.qkv_bias_separate:
                qkv = self.qkv(x) + qkv_bias
            else:
                qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
    else:
        q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
    q, k = self.q_norm(q), self.k_norm(k)
    if rope is not None:
        npt = self.num_prefix_tokens
        half = getattr(self, "rotate_half", False)
        q = torch.cat(
            [q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope, half=half)], dim=2
        ).type_as(v)
        k = torch.cat(
            [k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope, half=half)], dim=2
        ).type_as(v)

    q, k = stop_gradient(q), stop_gradient(k)

    if self.fused_attn:
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = maybe_add_mask(attn, resolve_self_attn_mask(N, attn, attn_mask, is_causal=is_causal))
        x = attn.softmax(dim=-1) @ v
    x = x.transpose(1, 2).reshape(B, N, C)
    return self.proj_drop(self.proj(self.norm(x)))


ATTNLRP_MAP = {
    # timm's LayerNorm subclasses nn.LayerNorm but overrides forward, so patching
    # the parent is silently inert and the identity rule never runs.
    TimmLayerNorm: partial(patch_method, layer_norm_forward),
    nn.LayerNorm: partial(patch_method, layer_norm_forward),
    nn.Dropout: partial(patch_method, dropout_forward),
    nn.GELU: partial(patch_method, non_linear_forward, keep_original=True),
    EvaAttention: partial(patch_method, eva_attention_forward),
}


# --------------------------------------------------------------------------- #
# Forward to the pooled feature, backward to a relevance map
# --------------------------------------------------------------------------- #
class Explainer:
    """One patched backbone, reused for every image and seed. forward() hooks block L's
    tokens, so the later blocks are a dead branch."""

    def __init__(self, model: nn.Module, layer_idx: int) -> None:
        self.model = model
        self.layer_idx = layer_idx
        self.device = next(model.parameters()).device

    def forward(self, image: torch.Tensor) -> dict[str, Any]:
        x = image.unsqueeze(0).to(self.device).requires_grad_(True)
        store: dict[str, torch.Tensor] = {}
        handle = self.model.blocks[self.layer_idx].register_forward_hook(
            lambda m, i, o: store.__setitem__("block", o)
        )
        try:
            self.model.forward_features(x)
        finally:
            handle.remove()
        return {"image": image, "pixels": x, "pooled": store["block"].mean(dim=1)[0]}

    def heatmap(self, ctx: dict[str, Any], seed: torch.Tensor, retain: bool = True):
        """Relevance for one seed cotangent, summed onto the 14x14 patch grid."""
        seed = seed.to(self.device)
        grad = torch.autograd.grad(ctx["pooled"], ctx["pixels"], grad_outputs=seed,
                                   retain_graph=retain)[0]
        # grad x input expands around whatever counts as "zero" -- black, on
        # unnormalised inputs, which zeroes relevance on a black gorilla.
        origin = ctx["image"].mean(dim=(1, 2), keepdim=True).to(self.device)
        relevance = ((ctx["pixels"].detach() - origin) * grad).sum(1)[0]
        side = relevance.shape[0] // PATCH
        grid = relevance.reshape(side, PATCH, side, PATCH).sum(dim=(1, 3))
        return grid.detach().float().cpu()


def build_model(ckpt: str, timm_name: str, device: str) -> nn.Module:
    model = _load_local_model(ckpt, device, timm_name)
    model.eval()
    monkey_patch(model, ATTNLRP_MAP, verbose=False)
    log(f"loaded + LRP-patched {timm_name} on {device}")
    return model


# --------------------------------------------------------------------------- #
# Inputs: the cache, the probe direction, the images
# --------------------------------------------------------------------------- #
def load_probe_direction(probe_path: Path) -> torch.Tensor:
    """The probe's weight vector, seed of the centre class-LRP panel."""
    payload = torch.load(probe_path, map_location="cpu", weights_only=False)
    dim = int(payload["scaler_mean"].reshape(-1).shape[0])
    w = payload["probe_tensor"].to(torch.float64).numpy().reshape(-1)[:dim]
    return torch.tensor(w, dtype=torch.float32)


def load_image(path: str, image_size: int) -> torch.Tensor:
    """``Resize + ToTensor``, exactly what ``src.data.collect`` fed the model."""

    tf = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    return tf(Image.open(path).convert("RGB"))


def select_targets(cache: AttributionCache, fragments: list[str]) -> np.ndarray:
    """Test rows whose filename contains one of `fragments`."""
    names = [Path(p).name for p in cache.target_meta.path]
    rows = np.asarray([i for i, n in enumerate(names) if any(f in n for f in fragments)])
    if len(rows) == 0:
        # The split is open-set, so a training identity is never a test point.
        raise SystemExit(
            "No test image matches. The test set holds "
            f"{', '.join(sorted(set(map(str, cache.target_meta.identity))))}"
        )
    return rows


def _display_img(ax, image: torch.Tensor) -> None:
    """``dualxda.utils.display_img`` with its dropped ``permute`` assignment fixed."""
    img = torch.clip(image.detach(), min=0.0, max=1.0).squeeze()
    if img.shape[0] == 3:
        ax.imshow(img.permute(1, 2, 0))
    else:
        ax.imshow(img, cmap="gray_r")


def _caption(ax, text: str, size: float) -> None:
    """Caption the centre panels from inside the image: a title would land on a rule."""
    ax.text(
        0.5, 0.015, text, transform=ax.transAxes, ha="center", va="bottom", fontsize=size,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
    )


def _framed(ax) -> None:
    ax.axis("on")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(2)


def _overlay(ax, image: torch.Tensor, grid: torch.Tensor) -> None:
    """Image underneath, imgify heatmap on top, symmetric around 0 at the map's max."""
    _display_img(ax, image)
    side = image.shape[-1]
    if grid.shape[-1] != side:
        grid = F.interpolate(
            grid[None, None].float(), size=(side, side),
            mode="bilinear",
        )[0, 0]
    lim = float(np.abs(grid.numpy()).max()) + 1e-12
    ax.imshow(imgify(grid.numpy(), cmap="bwr", symmetric=True, vmin=-lim, vmax=lim),
              alpha=OVERLAY_ALPHA)
    _framed(ax)


def _render_dualxda_layout(
    *,
    panels: list[dict[str, Any]],
    test_img: torch.Tensor,
    class_map: torch.Tensor,
    names: tuple[str, str],
    target: dict[str, Any],
    out_dir: Path,
    layer: str,
) -> Path:
    """The published DualXDA figure geometry, from ``DualDA.xda``: 6 rows x
    2*NSAMPLES+5 columns."""
    nsamples, caption_size = NSAMPLES, 9
    size = 2
    ncol = 2 * nsamples + 5
    # Columns 1.5 image heights apart, so the side panels' captions do not collide.
    fig = plt.figure(figsize=(ncol * (2 * size / 3) * 1.5, 4 * size))
    gs = gridspec.GridSpec(6, ncol, figure=fig)
    gs.update(wspace=0.05, hspace=0.2)

    # panels arrive left-to-right (weakest opponent .. strongest proponent);
    # split them back into the two blocks their column arithmetic expects.
    opponents = panels[:nsamples][::-1]  # strongest (most negative) first
    proponents = panels[nsamples:]  # strongest first

    # centre: test image, then the class heatmap, each 2 columns wide
    ax = fig.add_subplot(gs[1:3, nsamples + 1 : nsamples + 3])
    _display_img(ax, test_img)
    _framed(ax)
    _caption(
        ax,
        f"#{target['index']}  {target['identity']} "
        f"({target['age_sex']})\npred {target['pred']}",
        caption_size,
    )
    ax = fig.add_subplot(gs[1:3, nsamples + 3 : nsamples + 5])
    _overlay(ax, test_img, class_map)
    _caption(ax, f"probe heatmap -> {names[1]}", caption_size)

    for block, column in (
        (opponents, lambda i: nsamples - i),
        (proponents, lambda i: nsamples + 5 + i),
    ):
        for i, panel in enumerate(block):
            col = column(i)
            _overlay(fig.add_subplot(gs[0, col]), panel["image"], panel["map_train"])
            ax = fig.add_subplot(gs[1, col])
            ax.axis("off")
            ax.text(
                0.5, 0.05,
                f"Attribution: {panel['attr']:.2f},\nLabel: {panel['label']}\n{panel['identity']}",
                ha="center", va="center", fontsize=caption_size,
            )
            _display_img(fig.add_subplot(gs[2, col]), panel["image"])
            _framed(fig.gca())
            _overlay(fig.add_subplot(gs[3, col]), test_img, panel["map_test"])

    header = DualDA.get_fontsize_from_nsamples(nsamples)
    for span, text in (
        (gs[1, nsamples + 5 : ncol], "$\\bf{POSITIVELY}$ relevant training samples"),
        (gs[1, 1 : nsamples + 1], "$\\bf{NEGATIVELY}$ relevant training samples"),
    ):
        ax = fig.add_subplot(span)
        ax.text(0.5, 0.8, text, ha="center", va="center", transform=ax.transAxes, fontsize=header)
        ax.axis("off")

    # their gradient arrows, pointing outward from the centre
    ax = fig.add_subplot(gs[1, nsamples + 5 : ncol])
    colourgradarrow(ax, (0, 0.5), (nsamples, 0.5), cmap="Reds_r", n=100, lw=10)
    ax.axis("off")
    ax = fig.add_subplot(gs[1, 1 : nsamples + 1])
    colourgradarrow(ax, (nsamples, 0.5), (0, 0.5), cmap="Blues_r", n=100, lw=10)
    ax.axis("off")

    for row, text in ((0, "Train"), (3, "Test")):
        ax = fig.add_subplot(gs[row, 0])
        ax.text(0.8, 0.5, text, rotation=90, fontsize=30, ha="center", va="center")
        ax.axis("off")

    for span in (gs[0:2, 0:ncol], gs[2:4, 0:ncol]):
        ax = fig.add_subplot(span)
        ax.axhline(y=1 / 2, xmin=0.04, xmax=1, linestyle="-", linewidth=2, color="black")
        ax.axis("off")
    ax = fig.add_subplot(gs[0:4, 0:2])
    ax.axvline(x=1 / 2, ymin=0.0, ymax=1, linestyle="-", linewidth=2, color="black")
    ax.axis("off")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"xda_{layer}_test{target['index']}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def render(
    *,
    explainer: Explainer,
    cache: AttributionCache,
    w: torch.Tensor,
    t: int,
    image_size: int,
    out_dir: Path,
) -> Path:
    """One figure for test row ``t``: opponents left, test centre, proponents right."""
    row = cache.matrix[t]
    order = np.argsort(row)
    opponents = order[:NSAMPLES][::-1].tolist()  # most negative first
    proponents = order[-NSAMPLES:][::-1].tolist()  # most positive first

    names = cache.class_names
    test_img = load_image(cache.target_meta.path[t], image_size)
    ctx_test = explainer.forward(test_img)
    pooled_test = ctx_test["pooled"].detach()

    class_map = explainer.heatmap(ctx_test, w)

    target = {
        "index": int(cache.target_meta.index[t]),
        "identity": str(cache.target_meta.identity[t]),
        "age_sex": str(cache.target_meta.age_sex[t]),
        "pred": names[int(cache.target_pred[t])],
    }

    panels: list[dict[str, Any]] = []
    for k in opponents[::-1] + proponents:  # left-to-right: weakest opp .. strongest prop
        train_img = load_image(cache.source_meta.path[int(k)], image_size)
        ctx_train = explainer.forward(train_img)
        pooled_train = ctx_train["pooled"].detach()

        # attr/<x_i, x_t> is alpha_i at the optimum; computing it from the cached
        # attribution instead cross-checks the cache against this forward pass.
        dot = float((pooled_test * pooled_train).sum())
        scale = float(row[k]) / dot
        map_test = explainer.heatmap(ctx_test, pooled_train * scale)
        map_train = explainer.heatmap(ctx_train, pooled_test * scale, retain=False)

        panels.append(
            {
                "image": train_img,
                "map_train": map_train,
                "map_test": map_test,
                "attr": float(row[k]),
                "identity": str(cache.source_meta.identity[int(k)]),
                "age_sex": str(cache.source_meta.age_sex[int(k)]),
                "label": names[int(cache.source_meta.label[int(k)])],
            }
        )
        del ctx_train

    path = _render_dualxda_layout(
        panels=panels, test_img=test_img, class_map=class_map, names=names,
        target=target, out_dir=out_dir, layer=cache.layer,
    )
    log(
        f"  test #{target['index']} ({target['identity']}, {target['age_sex']}) "
        f"-> {path.name}  attr [{panels[0]['attr']:+.0f} .. {panels[-1]['attr']:+.0f}]"
    )
    return path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_run_arg(ap)
    ap.add_argument("--layer", default=None, help="Layer to read under --run, e.g. blocks.9")
    ap.add_argument("--cache", default=None,
                    help="One layer of a representer attribution cache. "
                         "Default: <run>/attributions/representer/<layer>")
    ap.add_argument("--test-image", nargs="+", required=True,
                    help="Filename fragments of the test images to explain, "
                         "e.g. PL02_R066_20220911_147_378_830031")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None,
                    help="Default: <run>/attributions/representer/xda, else <cache>/xda")
    args = ap.parse_args()
    if not args.cache and not (args.run and args.layer):
        ap.error("--cache, or --run together with --layer, is required")
    return args


def main() -> None:
    args = parse_args()
    run = run_of(args)
    cache_dir = Path(args.cache) if args.cache else run.attributions("representer") / args.layer
    cache = AttributionCache.load(cache_dir, matrix=True)

    probe_config = cache.config["probe"]
    acts_meta = torch.load(
        probe_config["activations_path"], map_location="cpu", weights_only=False
    )["meta"]
    w = load_probe_direction(Path(probe_config["probe_path"]))

    model = build_model(acts_meta["model_path"], acts_meta["timm_model"], args.device)
    explainer = Explainer(model, int(cache.layer.split(".")[-1]))

    targets = select_targets(cache, args.test_image)
    if args.out:
        out_dir = Path(args.out)
    elif run is not None:
        out_dir = run.xda("representer")   # filenames carry the layer
    else:
        out_dir = cache_dir / "xda"
    log(f"{cache.method} {cache.layer}: {len(targets)} test point(s), "
        f"{NSAMPLES}+{NSAMPLES} sources each -> {out_dir}")

    for t in targets:
        render(explainer=explainer, cache=cache, w=w, t=int(t),
               image_size=int(acts_meta["image_size"]), out_dir=out_dir)


if __name__ == "__main__":
    main()

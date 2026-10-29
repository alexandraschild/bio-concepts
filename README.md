# What Does Animal Re-Identification Learn? Linear Biological Concepts and Their Origins in Visual Representations

Code for the paper [What Does Animal Re-Identification Learn? Linear Biological Concepts and Their Origins in Visual Representations
](https://arxiv.org/abs/2609.06020).

## Setup

`dualxda-pip/` is a submodule that `uv sync` installs from the working tree, so clone with it:

```
git clone --recurse-submodules https://github.com/alexandraschild/bio-concepts.git
cd bio-concepts
uv sync
source .venv/bin/activate
```

## Data and checkpoints

The Re-ID fine-tuned checkpoint and the deduplicated gorilla dataset are available at
<https://gitlab.hpi.de/robert.nolting/ba-robert>.

- **Gorillas:** a deduplicated version of [Gorilla-SPAC-Wild](https://huggingface.co/datasets/gorilla-watch/Gorilla-SPAC-Wild).
  `data/gorilla_mappings_new.csv` maps each identity to its age-sex class and
  `data/spac23+24-stratified_open_split.json` holds the open-set identity split.
- **Jaguars:** the labelled images of the jaguar Re-ID challenge (Rueda-Toicen et al., 2026);
  `data/jaguar_sex_map.csv` maps each named jaguar to its sex.
- **Polar bears:** [PolarBearVidID](https://doi.org/10.3390/ani13050801) (Zuerl et al., 2023).
- **Backbones:** the stock DINOv3-Base (`vit_base_patch16_dinov3.lvd1689m`) and its Re-ID
  fine-tuned counterpart (link above).

## Reproducing the paper

Fill in the five paths at the top of `repro_scripts/paths.sh`, then run the scripts from the
repo root in order. Steps 1-4 submit their heavy stages through `slurm/submit.sh`, so start the
next step only once the previous step's jobs have finished.

```
repro_scripts/1_collect_activations.sh    # activation caches: gorillas (both backbones), jaguars, polar bears
repro_scripts/2_train_probes.sh           # every probe the paper reports
repro_scripts/3_run_experiments.sh        # probe evaluation, steering, attribution
repro_scripts/4_evaluate_experiments.sh   # random steering controls and the metrics the figures read
repro_scripts/5_draw_figures.sh           # every figure, named by its number in the paper
```

`5_draw_figures.sh` labels each command with the figure it draws. Figures 13, 16 and 17 were
drawn outside this repository from the tables these scripts write.

## Pipeline

Each stage is a Python module run from the repo root (`python -m src.<stage>`), reading what
the previous one wrote:

| stage | module | writes |
|---|---|---|
| activation collection | `src.data.collect` (`collect_jaguar`, `collect_polarbear`) | per-block activations of every image |
| probe training | `src.probes.train` | `probes/probe_<layer>.pt`: LR, SVM or difference-in-means |
| probe evaluation | `src.probes.evaluate holdout` / `id-cv` | `eval/holdout/`, `eval/transfer_<backbone>/`, `eval/id_cv/` |
| steering | `src.steering.steer`, `steer_controls`, `evaluate_reid` | steered Re-ID embeddings and their flip / off-manifold rates |
| data attribution | `src.attribution.attribute`, `evaluate` | representer and exact leave-one-identity-out caches and their metrics |
| DualXDA heatmaps | `src.attribution.xda` | heatmap figures |
| figures | `src.report.{probes,steering,attribution}` | figures and tables |

Every probe gets a run directory, which each stage reads from and writes to (`--run`);
`src/paths.py` defines the layout:

```
results/dinov3_base_finetuned/sex_lr_c1/
  probes/                 probe_<layer>.pt, split.json
  eval/                   holdout/, transfer_<backbone>/, id_cv/
  attributions/<method>/  <layer>/ caches, eval/ metrics, xda/ heatmaps
  steering/               embeddings/, eval/, controls/random_s<seed>/
```

`slurm/submit.sh <stage> [args...]` runs a module as a Slurm job with that stage's resources;
add your cluster's partition and account to `slurm/job.slurm`.

## Citation

```bibtex
@article{nolting2026reid,
  title   = {What Does Animal Re-Identification Learn? Linear Biological Concepts and Their Origins in Visual Representations},
  author  = {Nolting, Robert and Schild, Alexandra and Weckbecker, Moritz and Schall, Maximilian and de Melo, Gerard},
  journal = {arXiv preprint arXiv:2609.06020},
  year    = {2026}
}
```
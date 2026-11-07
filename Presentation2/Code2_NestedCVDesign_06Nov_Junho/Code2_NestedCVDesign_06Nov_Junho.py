#!/usr/bin/env python3
"""
Presentation 2 (Mathlete, Junho): recomputes every number used in the nested-CV search design.

Run on Quest from the repo's Presentation 2 base-code folder (it imports data_utils from there and reads the
labels CSV, patch folder and split files on /projects/e32998):
    cd Presentation2_base_code
    python ../Presentation2/Code2_NestedCVDesign_06Nov_Junho/Code2_NestedCVDesign_06Nov_Junho.py

Files in this folder
  pilot_sbatch/            the seven pilot jobs as submitted (run from a copy of the base code whose config.py
                           takes runs_dir from the RUN_BASE environment variable; otherwise unchanged)
  pilot_logs/              their Slurm output (one .out per job; the job id is in the file name)
  pilot_inner_A1.npz       the pilot split: train on groups 3-5, validate on group 2, test := validation
  pilot_predictions.csv    per-case predictions of the seed-42 pilot (job 7471053) at its best checkpoint

Sections
  1. Data: cases, outer groups, patches per slice, what the patch and slice caps remove
  2. Parameters: count per module for embedding sizes 64-512
  3. Weight decay: shrink exp(-lr*wd*T) at the old range and at the new range D = lr*wd*224
  4. Class weight: inverse-frequency weight of an inner training set
  5. Initial design: scrambled Sobol stratification vs random points; Optuna's default TPE good-set size
  6. Pilot runs: runtime, best epoch, per-case CE noise, seed spread (parsed from the Slurm logs)

Pilot runs (Slurm, CPU, 16 cores): inner model 1 of outer fold A, i.e. train on groups 3-5 (56 cases),
validate on group 2 (19 cases); group 1 (= A) is never used; test is set equal to validation.
    python main.py --load_splits pilot_inner_A1.npz --max_slices_per_stain 5 [--seed S] [--lr LR]
"""
import argparse, glob, math, os, re, sys
sys.path.insert(0, os.getcwd())  # base-code folder (data_utils, config)
import numpy as np

SPLITS = "/projects/e32998/STAT390_Krish/Code/Code9_no_leakage/no_train_val_leakage_splits"  # optuna_training.py L670
PATCHES = "/projects/e32998/patches_benign_split"
LABELS = "extra_benign_case_grade_match.csv"


def section(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def data_stats():
    from data_utils import load_labels, get_all_patch_files, group_patches_by_slice, build_slice_to_class_map, build_case_dict
    section("1. Data")
    lab = load_labels(LABELS)
    patches = group_patches_by_slice(get_all_patch_files(PATCHES), PATCHES)
    s2c = build_slice_to_class_map(patches, lab)
    cd, lm = build_case_dict(list(s2c.keys()), patches, s2c)
    n_b = sum(v == 0 for v in lm.values()); n_h = sum(v == 1 for v in lm.values())
    print(f"cases {len(cd)}: benign {n_b}, high-grade {n_h}")
    groups = [sorted(int(x) for x in np.load(f, allow_pickle=True)["test_cases"])
              for f in sorted(glob.glob(os.path.join(SPLITS, "data_splits_new_0*.npz")))]
    for i, g in enumerate(groups):
        print(f"group {i+1}: {len(g)} cases, benign {sum(lm[c] == 0 for c in g)}")
    per_slice = np.array([len(p) for st in cd.values() for sl in st.values() for p in sl])
    print(f"patches per slice: median {np.median(per_slice):.0f}, max {per_slice.max()}  "
          f"-> per_slice_cap 500/800 removes {int((per_slice > 500).sum())} slices' patches")
    cut = {s: sum(len(st.get(s, [])) > 5 for st in cd.values()) for s in ("h&e", "melan", "sox10")}
    print(f"max_slices_per_stain = 5 truncates: {cut} (total {sum(cut.values())} case-stain sets)")
    return groups, lm


def param_counts():
    section("2. Trainable parameters (projector 4096->d, 3 gated attentions with hidden 128, classifier)")
    for d in (64, 128, 256, 512):
        proj = 4096 * d + d; att = 3 * (2 * (d * 128 + 128) + 129); cls = 2 * d + 2
        print(f"d={d:3d}: projector {proj:,}  attention {att:,}  classifier {cls:,}  total {proj+att+cls:,}"
              f"  (projector share {proj/(proj+att+cls):.0%})")


def decay():
    section("3. Weight decay: AdamW multiplies weights by (1 - lr*wd) each step (torch/optim/adam.py L411)")
    x = 2.08e-4 * 3.05e-4 * 56 * 15
    print(f"config (lr 2.08e-4, wd 3.05e-4, 56 cases x 15 epochs): shrink {1 - math.exp(-x):.4%}")
    for n in (55, 56, 57):  # inner/outer training sets in the 93-case splits; old search ran 30 epochs (optuna_training.py L678)
        x = 1e-3 * 2e-3 * n * 30
        print(f"old range strongest corner (lr 1e-3, wd 2e-3), {n} cases x 30 epochs: shrink {1 - math.exp(-x):.3%}")
    print(f"per step, decay moves a weight by lr*wd*|w| vs Adam's step <~ lr: ratio wd*|w| = 2e-3 * 0.05 = {2e-3 * 0.05:.0e}")
    for D in (1e-2, 1.0):
        print(f"new range D = {D:g}: shrink by the kept checkpoint {1 - math.exp(-D):.0%}; "
              f"wd at lr=2e-4: {D / (2e-4 * 224):.2f}")


def learning_rate():
    section("3b. Learning-rate range")
    std = 1 / math.sqrt(3 * 4096)  # nn.Linear init U(-1/sqrt(fan_in), 1/sqrt(fan_in)), projector fan_in 4096
    print(f"lr 1e-5 over 30 epochs x 56 steps: at most about {1e-5 * 56 * 30:.4f}; projector init std {std:.4f}; ratio {1e-5 * 56 * 30 / std:.2f}")
    print(f"lr cut from 2.08e-4 to 5e-5: factor {2.08e-4 / 5e-5:.2f}")


def class_weight(groups, lm):
    section("4. Class weight = n_H / n_B of an inner training set (balanced CE)")
    tr = groups[2] + groups[3] + groups[4]
    n_b = sum(lm[c] == 0 for c in tr); n_h = len(tr) - n_b
    print(f"pilot inner training set: {len(tr)} cases, benign {n_b}, high-grade {n_h}, w_B/w_H = {n_h/n_b:.3f}")


def initial_design():
    section("5. Initial design")
    import torch
    from torch.quasirandom import SobolEngine
    for seed in (42, 0, 1):
        x = SobolEngine(5, scramble=True, seed=seed).draw(16).numpy()
        one_per_bin = all((np.bincount((x[:, j] * 16).astype(int), minlength=16) == 1).all() for j in range(5))
        grid2d = (np.bincount((x[:, 0] * 4).astype(int) * 4 + (x[:, 1] * 4).astype(int), minlength=16) == 1).all()
        print(f"Sobol seed {seed}: one point per 1/16 bin in all 5 coordinates: {one_per_bin}; one per 4x4 cell (coords 1-2): {grid2d}")
    print(f"random 16 points, expected empty bins: exact {16 * (15/16) ** 16:.2f}")
    rng = np.random.default_rng(0)
    sim = np.mean([16 - len(set((rng.random(16) * 16).astype(int))) for _ in range(100000)])
    print(f"random 16 points, empty bins by simulation (100,000 draws): {sim:.2f}")
    import optuna
    from optuna.samplers._tpe.sampler import default_gamma
    print(f"Optuna {optuna.__version__} default good-set size: " +
          ", ".join(f"n={n}: {default_gamma(n)}" for n in (8, 16, 24, 40)))
    print("proposed good-set size max(6, ceil(0.15 n)): " + ", ".join(f"n={n}: {max(6, math.ceil(0.15 * n))}" for n in (16, 24, 40)))
    from optuna.samplers._tpe.sampler import default_weights
    for k in (2, 6):  # parzen_estimator.py L59-69: observation weights plus prior_weight = 1, normalised
        w = np.append(default_weights(k), 1.0); w = w / w.sum()
        print(f"good set of {k}: each trial carries {w[0]:.3f} of l(x), the prior {w[-1]:.3f}")


def pilots(log_dir, pred_csv):
    section("6. Pilot runs")
    for f in sorted(glob.glob(os.path.join(log_dir, "P2pilot_*.out"))):
        txt = open(f).read()
        ep = [float(v) for v in re.findall(r"Epoch Time: ([\d.]+)s", txt)]
        best = re.findall(r"New best validation loss: ([\d.]+) at epoch (\d+)", txt)
        tot = re.search(r"Total execution time: ([\d.]+)s", txt)
        if not ep:
            continue
        print(f"{os.path.basename(f)}: {len(ep)} epochs, {np.median(ep[1:] or ep):.0f} s/epoch (median after epoch 1), "
              f"best {best[-1][0] if best else '-'} at epoch {best[-1][1] if best else '-'}, "
              f"total {float(tot.group(1))/60 if tot else float('nan'):.1f} min")
    mins, eps = [], []
    for f in sorted(glob.glob(os.path.join(log_dir, "P2pilot_*.out"))):
        t = open(f).read(); tot = re.search(r"Total execution time: ([\d.]+)s", t)
        if tot:
            mins.append(float(tot.group(1)) / 60); eps.append(len(re.findall(r"Epoch Time", t)))
    if mins:
        lo, hi = min(mins), max(mins)
        print(f"runtime per run {lo:.1f}-{hi:.1f} min, {min(eps)}-{max(eps)} epochs; 800 runs = {800*lo/60:.0f}-{800*hi/60:.0f} node-hours; "
              f"one outer fold, 40 trials with 4 inner models in parallel = {40*lo/60:.1f}-{40*hi/60:.1f} h")
    base = os.path.join(log_dir, "P2pilot_slices5_7471053.out")
    if os.path.exists(base):
        t = open(base).read()
        ce = [float(v) for v in re.findall(r"Train CE: ([\d.]+)", t)]; ent = [float(v) for v in re.findall(r"Train Patch Entropy: ([\d.]+)", t)]
        va = [float(v) for v in re.findall(r"Val Loss: ([\d.]+)", t)]
        print(f"base pilot: lambda*H / train CE at epoch 1 = {1e-3*ent[0]/ce[0]:.4%}, at the last epoch = {1e-3*ent[-1]/ce[-1]:.1%}; "
              f"train CE {ce[0]:.2f} -> {ce[-1]:.4f}; largest change of val loss between neighbouring epochs after epoch 2: "
              f"{max(abs(a-b) for a, b in zip(va[1:], va[2:])):.2f}")
    hi_lr = glob.glob(os.path.join(log_dir, "P2pilot_lr1e-3_*.out"))
    if hi_lr:
        ce = [float(v) for v in re.findall(r"Train CE: ([\d.]+)", open(hi_lr[0]).read())]
        print(f"lr 1e-3 pilot train CE by epoch: {np.round(ce, 3).tolist()}")
    seeds = []
    for f in sorted(glob.glob(os.path.join(log_dir, "P2pilot_seed*.out"))) + glob.glob(os.path.join(log_dir, "P2pilot_slices5_*.out")):
        b = re.findall(r"New best validation loss: ([\d.]+)", open(f).read())
        if b and "Total execution" in open(f).read():
            seeds.append(float(b[-1]))
    if len(seeds) > 1:
        print(f"best validation loss over {len(seeds)} seeds: {np.round(seeds, 3)}, SD {np.std(seeds, ddof=1):.3f}")
    import pandas as pd
    if os.path.exists(pred_csv):
        d = pd.read_csv(pred_csv)
        p = np.where(d.true_label == 1, d.prob_class1, d.prob_class0).clip(1e-12)
        ce = -np.log(p); y = d.true_label.values
        b, h = ce[y == 0], ce[y == 1]
        bal = 0.5 * (b.mean() + h.mean()); se = 0.5 * math.sqrt(b.var(ddof=1) / len(b) + h.var(ddof=1) / len(h))
        print(f"per-case CE at the kept checkpoint ({len(ce)} cases): balanced CE {bal:.3f}, SE {se:.3f}; "
              f"unweighted mean {ce.mean():.3f}, SE {ce.std(ddof=1)/math.sqrt(len(ce)):.3f}")
        if len(seeds) > 1:
            sd = np.std(seeds, ddof=1)
            d_seed = math.sqrt(2) * sd / math.sqrt(4)
            print(f"two trials with the same hyperparameters, each the mean of 4 inner models: "
                  f"SD of their difference from seeds alone = sqrt(2)*{sd:.3f}/sqrt(4) = {d_seed:.3f}; 2 SD = {2 * d_seed:.3f}")
            print(f"case noise per trial (4 folds, ~75 cases): {se:.3f}/sqrt(4) = {se / 2:.3f}; it is shared by trials scored on the "
                  f"same cases, so it partly cancels in a comparison (upper bound for the difference: {math.sqrt(2) * se / 2:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--pilot_logs", default=os.path.join(here, "pilot_logs"))
    ap.add_argument("--pilot_predictions", default=os.path.join(here, "pilot_predictions.csv"))
    ap.add_argument("--skip_data", action="store_true")
    a = ap.parse_args()
    if not a.skip_data:
        g, lm = data_stats(); class_weight(g, lm)
    param_counts(); decay(); learning_rate(); initial_design(); pilots(a.pilot_logs, a.pilot_predictions)

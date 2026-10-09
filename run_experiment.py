"""Benchmark how faithful saliency explanations are for a pretrained ResNet-18.

Usage (Colab / local GPU):
    python run_experiment.py --n-images 100
Quick pipeline check without downloads (random weights + random images):
    python run_experiment.py --smoke-test
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, models, transforms
from torchvision.datasets.utils import download_and_extract_archive

from xai.methods import get_methods
from xai.metrics import CASCADE, _spearman, deletion_insertion_curves, randomize_layers

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
# Imagenette folder names are sorted WordNet IDs; these are their ImageNet-1k indices.
IMAGENETTE_TO_IMAGENET = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
CLASS_NAMES = ["tench", "springer", "cassette player", "chain saw", "church",
               "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]
MEAN, STD = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])


def load_data(args):
    if args.smoke_test:
        x = torch.randn(args.n_images, 3, 224, 224)
        y = torch.randint(0, 10, (args.n_images,))
        return TensorDataset(x, y)
    root = os.path.join(args.data_dir, "imagenette2-320")
    if not os.path.isdir(root):
        download_and_extract_archive(IMAGENETTE_URL, args.data_dir)
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(MEAN.tolist(), STD.tolist()),
    ])
    ds = datasets.ImageFolder(os.path.join(root, "val"), transform=tf)
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(ds), generator=g)[: args.n_images * 2]  # extra, since we drop misclassified
    return Subset(ds, idx.tolist())


def load_model(args, device):
    weights = None if args.smoke_test else models.ResNet18_Weights.IMAGENET1K_V1
    return models.resnet18(weights=weights).to(device).eval()


def to_image(x):
    """Undo normalisation for plotting: (3,H,W) tensor -> (H,W,3) array in [0,1]."""
    return (x.cpu() * STD[:, None, None] + MEAN[:, None, None]).permute(1, 2, 0).clamp(0, 1).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-images", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--ig-steps", type=int, default=32)
    p.add_argument("--curve-steps", type=int, default=20)
    p.add_argument("--sanity-images", type=int, default=16)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = load_model(args, device)
    methods = get_methods(model, ig_steps=args.ig_steps)
    loader = DataLoader(load_data(args), batch_size=args.batch_size, shuffle=False)

    # ---- 1. keep only correctly classified images (explaining mistakes is a different question)
    xs, ts = [], []
    n_seen = n_correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            t = torch.tensor([IMAGENETTE_TO_IMAGENET[int(i)] for i in y], device=device)
            pred = model(x).argmax(1)
            keep = torch.ones_like(t, dtype=torch.bool) if args.smoke_test else pred == t
            n_seen += len(t); n_correct += int((pred == t).sum())
            xs.append(x[keep]); ts.append(t[keep])
            if sum(len(a) for a in xs) >= args.n_images:
                break
    X, T = torch.cat(xs)[: args.n_images], torch.cat(ts)[: args.n_images]
    print(f"top-1 accuracy on sampled images: {n_correct / n_seen:.1%}; explaining {len(X)} correct ones")

    # ---- 2. saliency maps + deletion/insertion
    rows, curves = [], {m: {"deletion": [], "insertion": []} for m in methods}
    examples = {m: None for m in methods}
    for start in range(0, len(X), args.batch_size):
        x, t = X[start:start + args.batch_size], T[start:start + args.batch_size]
        for name, fn in methods.items():
            sal = fn(x, t)
            if examples[name] is None:
                examples[name] = sal[:4].cpu()
            for mode in ("deletion", "insertion"):
                probs, auc = deletion_insertion_curves(model, x, t, sal, args.curve_steps, mode)
                curves[name][mode].append(probs)
                for i, a in enumerate(auc.tolist()):
                    rows.append({"image": start + i, "method": name, "metric": f"{mode}_auc", "value": a})
        print(f"  processed {min(start + args.batch_size, len(X))}/{len(X)} images")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "per_image_metrics.csv"), index=False)
    summary = df.pivot_table(index="method", columns="metric", values="value", aggfunc=["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for stat, metric in summary.columns]
    summary = summary[["deletion_auc_mean", "deletion_auc_std", "insertion_auc_mean", "insertion_auc_std"]]
    summary.round(4).to_csv(os.path.join(args.out, "summary.csv"))
    print("\nFaithfulness (deletion: lower is better, insertion: higher is better)")
    print(summary.round(3).to_string())

    # ---- 3. sanity check: cascading model randomisation
    xs_, ts_ = X[: args.sanity_images], T[: args.sanity_images]
    ref = {n: fn(xs_, ts_).flatten(1) for n, fn in methods.items() if n != "Random"}
    sanity = []
    for depth in range(1, len(CASCADE) + 1):
        rand_model = randomize_layers(models.resnet18().to(device), model.state_dict(), CASCADE[:depth], seed=args.seed)
        rand_methods = get_methods(rand_model, ig_steps=args.ig_steps)
        for n in ref:
            corr = _spearman(ref[n], rand_methods[n](xs_, ts_).flatten(1)).mean().item()
            sanity.append({"randomised_up_to": CASCADE[depth - 1], "method": n, "spearman": corr})
    sanity = pd.DataFrame(sanity)
    sanity.to_csv(os.path.join(args.out, "sanity_check.csv"), index=False)
    print("\nSanity check: rank correlation with original map (should DROP as layers are randomised)")
    print(sanity.pivot(index="randomised_up_to", columns="method", values="spearman")
          .loc[CASCADE].round(3).to_string())

    # ---- 4. figures
    names = list(methods)
    fig, axes = plt.subplots(4, len(names) + 1, figsize=(3 * (len(names) + 1), 12))
    for r in range(min(4, len(X))):
        axes[r, 0].imshow(to_image(X[r]))
        label = "random input" if args.smoke_test else CLASS_NAMES[IMAGENETTE_TO_IMAGENET.index(T[r].item())]
        axes[r, 0].set_title(label)
        for c, n in enumerate(names, start=1):
            axes[r, c].imshow(to_image(X[r]), alpha=0.5)
            axes[r, c].imshow(examples[n][r], cmap="jet", alpha=0.5)
            if r == 0:
                axes[r, c].set_title(n)
    for a in axes.flat:
        a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "examples.png"), dpi=120); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    frac = np.linspace(0, 1, args.curve_steps + 1)
    for ax, mode in zip(axes, ("deletion", "insertion")):
        for n in names:
            ax.plot(frac, torch.cat(curves[n][mode]).mean(0).numpy(), label=n)
        ax.set_title(f"{mode.capitalize()} curve")
        ax.set_xlabel(f"fraction of pixels {'removed' if mode == 'deletion' else 'inserted'}")
        ax.set_ylabel("p(target class)"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "curves.png"), dpi=120); plt.close()

    plt.figure(figsize=(6, 4))
    for n in ref:
        s = sanity[sanity.method == n]
        plt.plot(s.randomised_up_to, s.spearman, marker="o", label=n)
    plt.ylabel("Spearman corr. with original map"); plt.xlabel("randomised down to layer")
    plt.title("Model-randomisation sanity check"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "sanity_check.png"), dpi=120); plt.close()
    print(f"\nSaved CSVs and figures to {args.out}/")


if __name__ == "__main__":
    main()

# How faithful are saliency maps?
### Benchmarking Grad-CAM, Integrated Gradients and Vanilla Gradients in PyTorch

A pretrained ResNet-18 classifies real ImageNet photos from the 10-class Imagenette
subset. Three explanation methods produce saliency maps showing which pixels drive each
prediction. The question is how well each map reflects what the model *actually* uses,
so the maps are tested quantitatively instead of being judged by eye.

![Example saliency maps](results/examples.png)

## Methods

All explanation methods are implemented from scratch in [`saliency maps/methods.py`](saliency maps/methods.py).

| Method | Idea |
|---|---|
| Vanilla Gradient (Simonyan et al., 2014) | Gradient of the class score with respect to each input pixel |
| Integrated Gradients (Sundararajan et al., 2017) | Gradients averaged along a path from a baseline image to the input, times (input − baseline) |
| Grad-CAM (Selvaraju et al., 2017) | Last conv block's feature maps weighted by their mean gradient; implemented with forward/backward hooks |
| Random | Control condition that any real method should beat |

The evaluations are implemented in [`saliency maps/metrics.py`](saliency maps/metrics.py).

- **Deletion / insertion curves** (Petsiuk et al., 2018). Deletion removes the pixels a
  map ranks most important first and tracks the model's confidence; lower AUC is better.
  Insertion adds them back into a blurred image; higher AUC is better.
- **Model-randomisation sanity check** (Adebayo et al., 2018). The network's weights are
  re-initialised layer by layer, from the output down, and we measure how similar each
  explanation stays to the original (Spearman rank correlation). A faithful explanation
  should change once the model's knowledge is destroyed.

## Results

The faithfulness numbers are computed on 100 correctly classified Imagenette validation
images with a pretrained ResNet-18 (torchvision, ImageNet-1k weights).

| Method | Deletion AUC ↓ | Insertion AUC ↑ |
|---|---|---|
| Grad-CAM | 0.127 ± 0.106 | **0.575 ± 0.247** |
| Integrated Gradients | **0.067 ± 0.063** | 0.381 ± 0.219 |
| Vanilla Gradient | 0.100 ± 0.093 | 0.319 ± 0.202 |
| Random (control) | 0.115 ± 0.112 | 0.185 ± 0.137 |

![Deletion and insertion curves](results/curves.png)

The sanity check reports the Spearman correlation with the original map as more layers
are randomised (16 images).

| Randomised down to | Grad-CAM | Integrated Gradients | Vanilla Gradient |
|---|---|---|---|
| fc | 0.26 | 0.56 | 0.44 |
| layer4 | **0.02** | 0.38 | 0.21 |
| layer3 | 0.15 | 0.35 | 0.20 |
| layer2 | 0.09 | 0.33 | 0.18 |
| layer1 | 0.01 | 0.32 | 0.14 |
| conv1 | 0.22 | 0.31 | 0.10 |

![Sanity check](results/sanity_check.png)

## Key findings

1. **The two faithfulness metrics disagree.** Grad-CAM is clearly best on insertion, at
   about 3× the random baseline. On deletion, however, it scores slightly *worse than
   random*. Random deletion scatters grey pixels across the whole image, which destroys
   texture everywhere and makes confidence collapse for reasons unrelated to the
   explanation. Grad-CAM removes one contiguous region, so the model can still recognise
   the object from the rest. Which method "wins" therefore depends on how faithfulness is
   measured.
2. **Grad-CAM passes the sanity check; the gradient methods only partly do.** Grad-CAM's
   maps become almost uncorrelated with the original once `layer4` is randomised.
   Integrated Gradients keeps a correlation of about 0.3 even after *all* layers are
   randomised, so part of what it shows comes from the input image itself (edges,
   textures) rather than from what the model learned. This is consistent with Adebayo
   et al. (2018).

## Limitations

- Deletion uses a mean-colour baseline, which creates out-of-distribution images.
- The study uses one model and one dataset, and only correctly classified images.
- The sanity check uses only 16 images, and Grad-CAM's curve is not monotonic (it rises
  again at `conv1`), so this result is indicative rather than conclusive.

## Run it

The easiest way is Google Colab with a T4 GPU via [`colab_runner.ipynb`](colab_runner.ipynb).
To run it directly:

```bash
pip install -r requirements.txt
python run_experiment.py --smoke-test        # random weights/images, no downloads: checks the pipeline
python run_experiment.py --n-images 100      # real experiment (~340 MB Imagenette download)
```

## Repository structure

```
saliency maps/methods.py        Vanilla Gradient, Integrated Gradients, Grad-CAM, random baseline
saliency maps/metrics.py        deletion/insertion curves, model randomisation, Spearman correlation
run_experiment.py     full pipeline: data, model, explanations, metrics, figures
results/              CSVs and figures from the run reported above
```

## References

- Simonyan, Vedaldi & Zisserman (2014). *Deep Inside Convolutional Networks.*
- Sundararajan, Taly & Yan (2017). *Axiomatic Attribution for Deep Networks.*
- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks.*
- Petsiuk, Das & Saenko (2018). *RISE: Randomized Input Sampling for Explanation of Black-box Models.*
- Adebayo et al. (2018). *Sanity Checks for Saliency Maps.*

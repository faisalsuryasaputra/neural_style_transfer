# Neural Style Transfer with SqueezeNet (PyTorch)

A PyTorch implementation of **Neural Style Transfer** from
[*Image Style Transfer Using Convolutional Neural Networks* (Gatys et al., CVPR 2016)](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).
The project takes a **content image** and a **style image** and produces a new image that keeps the content of the first and the artistic style of the second.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/neural-style-transfer-squeezenet/blob/main/neural_style_transfer.ipynb)

## How it works

Instead of training a network, we run **gradient descent directly on the pixels** of the output image. A pretrained **SqueezeNet 1.1** (ImageNet, weights frozen) acts only as a feature extractor. The optimized loss is a weighted sum of three terms:

```
total loss = content loss + style loss + total-variation loss
```

| Loss | What it measures | What it preserves |
|---|---|---|
| **Content loss** | Squared difference between feature maps of the output and the content image at one layer | Objects and structure of the content image |
| **Style loss** | Squared difference between **Gram matrices** (channel correlations) of the output and the style image, summed over several layers | Textures, colors, and brush strokes of the style image |
| **Total-variation (TV) loss** | Squared difference between neighboring pixels (vectorized, no loops) | Smoothness and less noise in the output |

The Gram matrix of a feature map `F` (unrolled to `V` of shape `C x (H*W)`) is `G = V · Vᵀ`, normalized by `H*W*C`.
Optimization uses **Adam** (learning rate 3.0, dropped to 0.1 at iteration 180).

## What is implemented

- `features_from_img`: load, preprocess, and extract feature maps from an image
- `content_loss`: content term of the loss
- `gram_matrix`: unroll (`view`), transpose (`permute`), and `matmul` with normalization
- `style_loss`: multi-layer, weighted Gram-matrix loss
- `tv_loss`: total-variation regularization using tensor slicing
- `style_transfer`: full optimization loop with Adam and a learning-rate decay
- Experiments: **feature inversion** and **texture synthesis**

The notebook contains explanations before and after every step, plus comments inside the code.

## Unit-test results

Each function is checked against reference values (`style-transfer-checks.npz`):

| Test | Max error | Expected |
|---|---|---|
| Content loss | 0.00018 | 0.00018 |
| Gram matrix | 0.00041 | 0.00041 |
| Style loss | 0.00001 | 0.00001 |
| TV loss | 0.00000 | 0.00000 |

## Results

Content image: *Tübingen* (Germany). Images are generated at a small size (192 px) with 200 iterations, so they are intentionally lightweight.

**Content and style sources (example: Starry Night)**

![Sources](images/sources_content_style_example.png)

| Composition VII + Tübingen | The Scream + Tübingen | Starry Night + Tübingen |
|:---:|:---:|:---:|
| ![Composition VII](images/result_composition_vii.png) | ![The Scream](images/result_the_scream.png) | ![Starry Night](images/result_starry_night.png) |

**Style transfer = feature inversion (content) + texture synthesis (style)**

| Feature inversion (style weights = 0, random start) | Texture synthesis (content weight = 0, random start) |
|:---:|:---:|
| ![Feature inversion](images/feature_inversion.png) | ![Texture synthesis](images/texture_synthesis.png) |

## Hyperparameters (Starry Night example)

| Parameter | Value |
|---|---|
| Content layer | 3 |
| Content weight | 6e-2 |
| Style layers | 1, 4, 6, 7 |
| Style weights | 300000, 1000, 15, 3 |
| TV weight | 2e-2 |
| Image size | 192 (content), 192 (style) |
| Iterations | 200 |

Tips: a higher content weight gives a more realistic result, higher style weights give a more stylized result, and a higher TV weight gives a smoother result.

## Getting started

### Google Colab (recommended)

1. Click the **Open In Colab** badge above.
2. Select a GPU runtime: *Runtime → Change runtime type → T4 GPU* (CPU also works, but slower).
3. Run all cells (*Runtime → Run all*). The notebook downloads the images, reference file, and SqueezeNet weights automatically.

### Local

```bash
git clone https://github.com/YOUR_USERNAME/neural-style-transfer-squeezenet.git
cd neural-style-transfer-squeezenet
pip install -r requirements.txt
jupyter notebook neural_style_transfer.ipynb
```

The notebook uses `wget` to download the sample images, so on Windows download them manually or run it in Colab.

## Project structure

```
.
├── neural_style_transfer.ipynb   # full implementation, explanations, and outputs
├── images/                       # result images used in this README
├── requirements.txt
└── README.md
```

## Tech stack

Python, PyTorch, torchvision (SqueezeNet 1.1), NumPy, SciPy, Matplotlib, Pillow

## References

- L. A. Gatys, A. S. Ecker, M. Bethge. *Image Style Transfer Using Convolutional Neural Networks*. CVPR 2016.
- F. N. Iandola et al. *SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size*. arXiv 2016.
- A. Mahendran, A. Vedaldi. *Understanding Deep Image Representations by Inverting Them*. CVPR 2015.

## Acknowledgements

This project was completed as a course assignment (Task 4 – Neural Style Transfer). The assignment template, sample images, and reference check file are provided by the course instructor (ADF).

## Author

**[Your Name]**

# PanoHead: Geometry-Aware 3D Full-Head Synthesis in 360°

**Team 3 - Digital Humans Course**
- Sevval Uyanik - suyanik@ethz.ch
- Trygve Eriksen - teriksen@ethz.ch
- Umut Demirbas - udemirbas@ethz.ch

<a href="https://arxiv.org/abs/2303.13071"><img src="https://img.shields.io/badge/arXiv-2303.13071-b31b1b" height=22.5></a>
<a href="https://creativecommons.org/licenses/by/4.0"><img src="https://img.shields.io/badge/LICENSE-CC--BY--4.0-yellow" height=22.5></a>
<a href="https://www.youtube.com/watch?v=Y8NXiBOEWoE"><img src="https://img.shields.io/static/v1?label=CVPR 2023&message=8 Minute Video&color=red" height=22.5></a>

![Teaser image](./misc/teaser.png)

**PanoHead: Geometry-Aware 3D Full-Head Synthesis in 360°**

Sizhe An, Hongyi Xu, Yichun Shi, Guoxian Song, Umit Y. Ogras, Linjie Luo

https://sizhean.github.io/panohead

### Abstract

*Synthesis and reconstruction of 3D human head has gained increasing interests in computer vision and computer graphics recently. Existing state-of-the-art 3D generative adversarial networks (GANs) for 3D human head synthesis are either limited to near-frontal views or hard to preserve 3D consistency in large view angles. We propose PanoHead, the first 3D-aware generative model that enables high-quality view-consistent image synthesis of full heads in 360° with diverse appearance and detailed geometry using only in-the-wild unstructured images for training.*

---

## Table of Contents

1. [Requirements & Setup](#requirements--setup)
2. [Getting Started](#getting-started)
3. [Core Features](#core-features)
   - [Basic Generation](#basic-generation)
   - [Head Reconstruction (PTI)](#head-reconstruction-pti)
   - [Head Interpolation](#head-interpolation)
4. [Expression Editing (Team 3 Implementation)](#expression-editing-team-3-implementation)
5. [Advanced Usage](#advanced-usage)

---

## Requirements & Setup

### System Requirements

- **OS**: Linux (recommended for performance)
- **GPU**: 1–8 high-end NVIDIA GPUs (V100, RTX3090, A100 tested)
- **Python**: 64-bit Python 3.8+
- **PyTorch**: 1.11.0 or later
- **CUDA**: 11.3 or later

### Installation

1. **Clone the repository** and navigate to the directory:
   ```bash
   cd PanoHead
   ```

2. **Create conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate panohead
   ```

3. **Download pre-trained models**:
   Download from [Google Drive](https://drive.google.com/drive/folders/1m517-F1NCTGA159dePs5R5qj02svtX1_?usp=sharing) and place in the `models/` directory.

### Interactive Terminal Setup (Cluster)

```bash
# Request GPU resources on cluster
srun -A digital_human --time=01:00:00 --mem=16G --gpus 2080ti:1 --pty bash

# Activate environment
conda activate panohead
```

---

## Getting Started

### Quick Start Examples

**Generate videos from random seeds**:
```bash
python gen_videos.py \
  --network models/easy-khair-180-gpc0.8-trans10-025000.pkl \
  --seeds 0-3 \
  --grid 2x2 \
  --output=out \
  --cfg Head \
  --trunc 0.7
```

**Generate images and 3D shapes**:
```bash
python gen_samples.py \
  --outdir=out \
  --trunc=0.7 \
  --shapes=true \
  --seeds=0-3 \
  --network models/easy-khair-180-gpc0.8-trans10-025000.pkl
```

---

## Core Features

### Basic Generation

Generate high-quality 3D head images from random latent codes.

**Scripts**:
- `gen_videos.py` - Generate MP4 videos of rotating heads
- `gen_samples.py` - Generate static images and 3D shapes (.mrc files)

**Key options**:
- `--network` - Path to pre-trained model (.pkl)
- `--seeds` - Latent code seeds (e.g., `0-3` for 4 heads)
- `--trunc` - Truncation value for controlling diversity (0.0-1.0)
- `--cfg` - Configuration preset (Head)
- `--outdir` - Output directory

### Head Reconstruction (PTI)

Reconstruct a 3D full head from a single RGB image using Pivot Tuning Inversion (PTI).

**Script**: `projector.py` or `projector_withseg.py`

**Basic Usage**:
```bash
python projector.py \
  --network models/easy-khair-180-gpc0.8-trans10-025000.pkl \
  --target my_face.jpg \
  --num-steps 500 \
  --num-steps-pti 500 \
  --outdir pti_output \
  --save-video true
```

**Output**:
- `projected_w.npz` - Latent code (w vector) of your face
- `projected_w.mp4` - Optimization progress video
- `fintuned_generator.pkl` - Fine-tuned generator for your face

---

## Expression Editing (Team 3 Implementation)

### Overview

This extension enables discovery and manipulation of facial expressions in PanoHead's latent space. The toolkit allows you to:

1. **Extract expression editing vectors** from multiple facial expressions
2. **Apply vectors** to generate expression morphing videos

### Complete Pipeline

Run the entire workflow in one command:

```bash
bash pipeline_expression_editing.sh \
  models/easy-khair-180-gpc0.8-trans10-025000.pkl \
  dataset/expressions \
  pti_out_expressions \
  expression_vectors \
  dataset/crop_img \
  pti_out_crop \
  interpolations_crop_img
```

This executes all four steps automatically:
1. PTI projection of expression images
2. PTI projection of target identity images
3. Vector extraction
4. Interpolation generation

### Step-by-Step Workflow

#### Step 1: Prepare Expression Dataset

Organize expression images with camera parameters:

```
dataset/expressions/
├── dataset.json          # Image-to-camera mapping
├── neutral_face.jpg
├── smile_face.jpg
├── anger_face.jpg
├── surprise_face.jpg
└── ... (other expressions)
```

**`dataset.json` Format**:
```json
{
  "labels": [
    ["neutral_face.jpg", [25_camera_matrix_values]],
    ["smile_face.jpg", [25_camera_matrix_values]],
    ["anger_face.jpg", [25_camera_matrix_values]],
    ...
  ]
}
```

**Camera Matrix** (25 values):
- **16 values**: 4×4 camera-to-world transformation matrix
  ```
  [R00, R01, R02, T0]
  [R10, R11, R12, T1]
  [R20, R21, R22, T2]
  [0,   0,   0,   1 ]
  ```
- **9 values**: 3×3 camera intrinsics matrix
  ```
  [fx,  0,  cx]
  [0,  fy,  cy]
  [0,   0,   1 ]
  ```

**Generate Camera Matrices**:
- Use 3DDFA_V2 (available in [here](https://github.com/umutdemirbas/3DDFA_V2)) to estimate from 2D face images
- Use manual calibration if available
- Use default front-facing camera for quick testing

#### Step 2: Project Expressions to Latent Space

Project each expression image using PTI optimization:

```bash
python projector_withseg.py \
  --network models/easy-khair-180-gpc0.8-trans10-025000.pkl \
  --target_img dataset/expressions/smile \
  --idx 1 \
  --num-steps 500 \
  --num-steps-pti 500 \
  --outdir pti_out_expressions \
  --save-video true
```

**Output**:
- `pti_out_expressions/[model]/[idx]/projected_w.npz` - Latent code
- `pti_out_expressions/[model]/[idx]/PTI_render/` - Progress videos

**Note**: The `--idx` parameter corresponds to the index in `dataset.json` labels array.

#### Step 3: Extract Expression Vectors

Compute editing vectors as differences between expressions and a base (usually neutral):

```bash
python extract_expression_vectors.py \
  --pti-dir pti_out_expressions \
  --base-idx 2 \
  --outdir expression_vectors \
  --visualize true
```

**Parameters**:
- `--pti-dir` - Directory containing PTI embeddings
- `--base-idx` - Index of base expression in dataset.json (e.g., 2 for neutral)
- `--outdir` - Output directory for vectors and visualizations
- `--visualize` - Generate analysis plots (vector magnitudes, distance matrix)

**Output**:
- `expression_vectors_[base_name].pkl` - Serialized vectors (dict format)
- `vector_magnitudes.png` - Magnitude visualization
- `vector_distances.png` - Distance matrix heatmap
- `individual vectors/` - Individual .npy files per expression

**Understanding the Output**:
- Each vector represents the editing direction for an expression
- Vectors are computed as: `expression_w - neutral_w`
- Can be applied with different strengths to control intensity

#### Step 4: Apply Expression Vectors

Generate interpolation videos by gradually applying expression vectors:

```bash
python apply_expression_vectors.py \
  --network models/easy-khair-180-gpc0.8-trans10-025000.pkl \
  --base-w pti_out_crop/[model]/2/projected_w.npz \
  --vectors expression_vectors/expression_vectors_neutral.pkl \
  --expression smile \
  --strength 1.0 \
  --steps 50 \
  --freeze-layers 6 \
  --outdir videos_out \
  --fps 30
```

**Parameters**:
- `--base-w` - Base identity latent code (your face)
- `--vectors` - Expression vectors pickle file
- `--expression` - Which expression to apply
- `--strength` - Intensity of expression (0.0-2.0, default 1.0)
- `--steps` - Interpolation frames
- `--freeze-layers` - Number of W+ layers to preserve (maintains identity)
- `--fps` - Video frame rate

**Output**:
- `videos_out/[expression]_strength[X].mp4` - Interpolation video
- `videos_out/individual_expressions/` - Frame PNGs (optional)

---

## References

**Original PanoHead Paper**: https://arxiv.org/abs/2303.13071

**Project Page**: https://sizhean.github.io/panohead

**Citation**:
```bibtex
@inproceedings{an2023panohead,
  title={PanoHead: Geometry-Aware 3D Full-Head Synthesis in 360°},
  author={An, Sizhe and Xu, Hongyi and Shi, Yichun and Song, Guoxian and Ogras, Umit Y and Luo, Linjie},
  booktitle={CVPR},
  year={2023}
}
```

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
  --outdir=out \
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

### Head Interpolation

Generate smooth transitions between two heads.

**Script**: `gen_interpolation.py`

```bash
python gen_interpolation.py \
  --network models/easy-khair-180-gpc0.8-trans10-025000.pkl \
  --trunc 0.7 \
  --outdir interpolation_out
```

---

## Expression Editing (Team 3 Implementation)

### Overview

This extension enables discovery and manipulation of facial expressions in PanoHead's latent space. The toolkit allows you to:

1. **Extract expression editing vectors** from multiple facial expressions
2. **Apply vectors** to generate expression morphing videos
3. **Compare expressions** visually side-by-side
4. **Combine vectors** from multiple identities for robustness

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

**Default Front-Facing Camera** (if precise calibration unavailable):
```json
[1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.7, 0.0, 0.0, 0.0, 1.0, 4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0]
```

**Generate Camera Matrices**:
- Use 3DDFA_V2 (available in sibling directory) to estimate from 2D face images
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

#### Step 5: Compare Expressions

Visualize all expressions applied to the same face:

```bash
python compare_expressions.py \
  --network models/easy-khair-180-gpc0.8-trans10-025000.pkl \
  --base-w pti_out_crop/[model]/2/projected_w.npz \
  --vectors expression_vectors/expression_vectors_neutral.pkl \
  --strength 1.0 \
  --outdir comparison_out \
  --layout grid \
  --cols 3
```

**Parameters**:
- `--layout` - `grid`, `horizontal`, or `vertical`
- `--cols` - Number of columns (for grid layout)
- `--strength` - Uniform strength for all expressions

**Output**:
- `comparison_grid_strength[X].png` - Grid layout image
- `individual_expressions/` - Individual expression PNGs

### Combining Vectors from Multiple Identities

For more robust expression vectors, combine vectors extracted from multiple people:

```bash
python combine_expression_vectors.py \
  --input-dirs pti_out_person1/expression_vectors pti_out_person2/expression_vectors \
  --outdir robust_vectors \
  --min-instances 2
```

**Output**:
- `robust_expression_vectors.pkl` - Averaged vectors
- `robust_vectors/` - Individual vector files
- `robust_vectors_metadata.json` - Statistics and reliability scores

---

## Advanced Usage

### Using Networks from Python

```python
import pickle
import torch
import numpy as np

# Load model
with open('models/easy-khair-180-gpc0.8-trans10-025000.pkl', 'rb') as f:
    data = pickle.load(f)
    G = data['G_ema'].cuda()

# Generate from random latent code
z = torch.randn([1, G.z_dim]).cuda()
c = torch.cat([cam2world.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)  # camera params
img = G(z, c)['image']  # NCHW, [-1, +1], shape [1, 3, 512, 512]
mask = G(z, c)['image_mask']  # NHW, [0, 255], shape [1, 512, 512]
```

### Loading Projected W Vectors

```python
import numpy as np

# Load PTI projection result
data = np.load('pti_out_expressions/model/0/projected_w.npz')
w = data['w']  # Shape: [1, num_layers, 512]
```

### Loading Expression Vectors

```python
import pickle

# Load expression vectors
with open('expression_vectors/expression_vectors_neutral.pkl', 'rb') as f:
    vectors = pickle.load(f)

# Apply to new w vector
base_w = np.load('my_face_w.npz')['w']
smile_vector = vectors['smile']
edited_w = base_w + 1.0 * smile_vector
```

### Utility Scripts

- `calc_mbs.py` - Calculate memory requirements for models
- `calc_metrics.py` - Compute FID and other metrics
- `dataset_tool.py` - Prepare datasets for training
- `dataset_tool_seg.py` - Prepare datasets with segmentation
- `estimate_camera_params.py` - Estimate camera matrices from images
- `resave_model.py` - Convert model formats

---

## Implementation Details

### Key Files for Expression Editing

| Script | Purpose |
|--------|---------|
| `extract_expression_vectors.py` | Extract editing vectors from PTI embeddings |
| `apply_expression_vectors.py` | Apply vectors to generate interpolation videos |
| `compare_expressions.py` | Visualize expressions in grid/horizontal/vertical layouts |
| `combine_expression_vectors.py` | Combine vectors from multiple identities |
| `pipeline_expression_editing.sh` | Complete automated workflow |
| `projector_withseg.py` | Modified projector for expression datasets |

### Modifications from Original PanoHead

1. **projector_withseg.py** - Enhanced version of `projector.py` that:
   - Works with expression image folders
   - Loads camera parameters from `dataset.json`
   - Supports per-image PTI optimization with segmentation

2. **pipeline_expression_editing.sh** - Automated workflow that:
   - Runs PTI on expression set
   - Extracts vectors
   - Applies vectors to target identity
   - Generates interpolation videos

3. **Camera Parameter Handling**:
   - Automatic loading from `dataset.json`
   - Support for default front-facing camera
   - Integration with 3DDFA_V2 for estimation

---

## Troubleshooting

### Camera Matrix Issues

**Error**: `camera matrix: torch.Size([1, 0])` - Empty camera matrix
- **Solution**: Ensure `dataset.json` is in the correct location and has proper format

**Error**: Camera not found for image
- **Solution**: Check that image filenames in `dataset.json` match actual files (case-sensitive)

### Memory Issues

**Error**: Out of GPU memory
- **Solution**: Reduce batch size or use smaller model
- Check `calc_mbs.py` for memory requirements

### PTI Optimization Issues

**Error**: Poor projection quality
- **Solution**: Increase `--num-steps` and `--num-steps-pti`
- Ensure input image is well-lit and frontal
- Check camera parameters are reasonable

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

The above code requires `torch_utils` and `dnnlib` to be accessible via `PYTHONPATH`. It does not need source code for the networks themselves &mdash; their class definitions are loaded from the pickle via `torch_utils.persistence`.

The pickle contains three networks. `'G'` and `'D'` are instantaneous snapshots taken during training, and `'G_ema'` represents a moving average of the generator weights over several training steps. The networks are regular instances of `torch.nn.Module`, with all of their parameters and buffers placed on the CPU at import and gradient computation disabled by default.



## Datasets

FFHQ-F(ullhead) consists of [Flickr-Faces-HQ dataset](https://github.com/NVlabs/ffhq-dataset), [K-Hairstyle dataset](https://psh01087.github.io/K-Hairstyle/), and an in-house human head dataset. For head pose estimation, we use [WHENet](https://arxiv.org/abs/2005.10353).

Due to the license issue, we are not able to release FFHQ-F dataset that we used to train the model. [test_data_img](./dataset/testdata_img/) and [test_data_seg](./dataset/testdata_seg/) are just an example for showing the dataset struture. For the camera pose convention, please refer to [EG3D](https://github.com/NVlabs/eg3d). 


## Datasets format
For training purpose, we can use either zip files or normal folder for image dataset and segmentation dataset. For PTI, we need to use folder.

To compress dataset folder to zip file, we can use [dataset_tool_seg](./dataset_tool_seg.py). 

For example:
```.bash
python dataset_tool_seg.py --img_source dataset/testdata_img --seg_source  dataset/testdata_seg --img_dest dataset/testdata_img.zip --seg_dest dataset/testdata_seg.zip --resolution 512x512
```

## Obtaining camera pose and cropping the images
Please follow the [guide](3DDFA_V2_cropping/cropping_guide.md)

## Obtaining segmentation masks
You can try using deeplabv3 or other off-the-shelf tool to generate the masks. For example, using deeplabv3: [misc/segmentation_example.py](misc/segmentation_example.py)




## Training

Examples of training using `train.py`:

```
# Train with StyleGAN2 backbone from scratch with raw neural rendering resolution=64, using 8 GPUs.
# with segmentation mask, trigrid_depth@3, self-adaptive camera pose loss regularizer@10

python train.py --outdir training-runs  --img_data dataset/testdata_img.zip --seg_data dataset/testdata_seg.zip --cfg=ffhq --batch=32 --gpus 8\\
--gamma=1 --gamma_seg=1 --gen_pose_cond=True --mirror=1 --use_torgb_raw=1 --decoder_activation="none" --disc_module MaskDualDiscriminatorV2\\
--bcg_reg_prob 0.2 --triplane_depth 3 --density_noise_fade_kimg 200 --density_reg 0 --min_yaw 0 --max_yaw 180 --back_repeat 4 --trans_reg 10 --gpc_reg_prob 0.7


# Second stage finetuning to 128 neural rendering resolution (optional).

python train.py --outdir results --img_data dataset/testdata_img.zip --seg_data dataset/testdata_seg.zip --cfg=ffhq --batch=32 --gpus 8\\
--resume=~/training-runs/experiment_dir/network-snapshot-025000.pkl\\
--gamma=1 --gamma_seg=1 --gen_pose_cond=True --mirror=1 --use_torgb_raw=1 --decoder_activation="none" --disc_module MaskDualDiscriminatorV2\\
--bcg_reg_prob 0.2 --triplane_depth 3 --density_noise_fade_kimg 200 --density_reg 0 --min_yaw 0 --max_yaw 180 --back_repeat 4 --trans_reg 10 --gpc_reg_prob 0.7\\
--neural_rendering_resolution_final=128 --resume_kimg 1000
```

## Metrics



```.bash
./get_metrics.sh
```
There are three evaluation modes: all, front, and back as we mentioned in the paper. Please refer to [cal_metrics.py](./calc_metrics.py) for the implementation.


## Citation

If you find our repo helpful, please cite our paper using the following bib:

```
@InProceedings{An_2023_CVPR,
    author    = {An, Sizhe and Xu, Hongyi and Shi, Yichun and Song, Guoxian and Ogras, Umit Y. and Luo, Linjie},
    title     = {PanoHead: Geometry-Aware 3D Full-Head Synthesis in 360deg},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {20950-20959}
}
```

## Development

This is a research reference implementation and is treated as a one-time code drop. As such, we do not accept outside code contributions in the form of pull requests.

## Acknowledgements

We thank Shuhong Chen for the discussion during Sizhe's internship.

This repo is heavily based off the [NVlabs/eg3d](https://github.com/NVlabs/eg3d) repo; Huge thanks to the EG3D authors for releasing their code!
#!/usr/bin/env bash
#SBATCH --job-name=pano_expression_pipeline
#SBATCH --account=digital_human
#SBATCH --partition=gpu
#SBATCH --gpus=2080ti:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/pipeline_%j.log
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=udemirbas@student.ethz.ch

# Complete pipeline for expression editing:
# 1) PTI on expressions set (for vector extraction)
# 2) PTI on target set (crop_img)
# 3) Extract vectors from expressions
# 4) Apply vectors to target set
# 
# IMPORTANT: Your dataset.json should be in EXPRESSION_DIR and contain labels like:
# {
#   "labels": [
#     ["neutral_image.jpg", [camera_matrix_25_values]],
#     ["smile_image.jpg", [camera_matrix_25_values]],
#     ...
#   ]
# }
# 
# Usage:
# sbatch pipeline_expression_editing.sh [network] [expr_dir] [expr_pti_out] [vectors_dir] [target_dir] [target_pti_out] [interp_out]

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

NETWORK="${1:-models/easy-khair-180-gpc0.8-trans10-025000.pkl}"
EXPRESSION_DIR="${2:-dataset/expressions}"  # Should contain dataset.json with labels
OUTPUT_DIR="${3:-fin_resultsv5/pti_out_expressions}"
VECTORS_DIR="${4:-fin_resultsv5/expression_vectors}"
TARGET_DIR="${5:-dataset/final_dataset/}"
TARGET_OUTPUT_DIR="${6:-fin_resultsv5/pti_out_crop}"
INTERP_OUTPUT_DIR="${7:-fin_resultsv5/interpolations_crop_img}"

NUM_STEPS_W=500              # Initial w optimization steps
NUM_STEPS_PTI=500            # PTI fine-tuning steps
STRENGTH=1.0                 # Maximum strength for interpolation
INTERP_STEPS=50              # Number of interpolation frames

# Create logs directory
mkdir -p logs

# Conservative default for TORCH_CUDA_ARCH_LIST (set before activating conda)
# so that nvcc invoked by torch's cpp_extension won't attempt to compile
# unsupported architectures (e.g. compute_120) on this cluster.
if [ -z "$TORCH_CUDA_ARCH_LIST" ]; then
    echo "Exporting TORCH_CUDA_ARCH_LIST=8.6;8.0;7.5"
    export TORCH_CUDA_ARCH_LIST="8.6;8.0;7.5"
fi

# Initialize and activate conda environment
eval "$(conda shell.bash hook)"
conda activate panohead

echo "Starting expression editing pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Python: $(python --version)"
echo "Python path: $(which python)"

# Clear any previously cached torch extension builds to force recompilation
# with the current TORCH_CUDA_ARCH_LIST. This avoids nvcc trying stale
# object files compiled for unsupported archs.
echo "Clearing torch extensions cache (~/.cache/torch_extensions)"
rm -rf ~/.cache/torch_extensions/* || true

run_pti_for_dataset() {
    local DATA_DIR="$1"
    local PTI_OUT="$2"
    local LABEL="$3"

    if [ ! -f "$DATA_DIR/dataset.json" ]; then
        echo "Error: Missing dataset.json in $DATA_DIR ($LABEL)"
        exit 1
    fi

    mkdir -p "$PTI_OUT"

    local ITEMS
    ITEMS=$(python - <<PY
import os
path = '$DATA_DIR'
exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
fnames = []
for root, _dirs, files in os.walk(path):
    for f in files:
        if os.path.splitext(f)[1].lower() in exts:
            rel = os.path.relpath(os.path.join(root, f), start=path).replace('\\\\', '/')
            fnames.append(rel)
fnames = sorted(fnames)
for i, rel in enumerate(fnames):
    stem = os.path.splitext(os.path.basename(rel))[0]
    print(f"{i}:{stem}")
PY
 2>>"$PTI_OUT/debug.log")

    if [ -z "$ITEMS" ]; then
        echo "Error: Could not parse dataset.json in $DATA_DIR"
        exit 1
    fi

    echo "Found $(echo \"$ITEMS\" | wc -l) images in $LABEL"

    for item in $ITEMS; do
        idx=$(echo "$item" | cut -d: -f1)
        name=$(echo "$item" | cut -d: -f2)

        echo ""
        echo "[$LABEL:$((idx+1))] Processing image: $name (index $idx)"

        python projector_withseg.py \
            --outdir="$PTI_OUT" \
            --target_img="$DATA_DIR" \
            --network="$NETWORK" \
            --idx "$idx" \
            --num-steps "$NUM_STEPS_W" \
            --num-steps-pti "$NUM_STEPS_PTI" \
            --save-video true
    done
}

# ============================================================================
# STEP 1: Run PTI on each expression image in dataset.json
# ============================================================================

echo "========================================="
echo "STEP 1: Running PTI on expression images"
echo "========================================="

run_pti_for_dataset "$EXPRESSION_DIR" "$OUTPUT_DIR" "expressions"

echo ""
echo "✓ Expression PTI optimization complete"

# ============================================================================
# STEP 2: Run PTI on target images (crop_img)
# ============================================================================

echo ""
echo "========================================="
echo "STEP 2: Running PTI on target crop images"
echo "========================================="

run_pti_for_dataset "$TARGET_DIR" "$TARGET_OUTPUT_DIR" "crop_img"

echo ""
echo "✓ Target PTI optimization complete"

# ============================================================================
# STEP 3: Extract expression editing vectors
# ============================================================================

echo ""
echo "========================================="
echo "STEP 3: Extracting expression vectors"
echo "========================================="

mkdir -p "$VECTORS_DIR"

python extract_expression_vectors.py \
    --pti-dir="$OUTPUT_DIR" \
    --base-idx=1 \
    --outdir="$VECTORS_DIR" \
    --visualize=true

echo ""
echo "✓ Expression vectors extracted"

# ============================================================================
# STEP 4: Apply vectors to target crop_img identities
# ============================================================================

echo ""
echo "========================================="
echo "STEP 4: Generating interpolation videos on crop_img"
echo "========================================="

# Find the vectors file dynamically
VECTORS_FILE=$(find "$VECTORS_DIR" -name "expression_vectors_*.pkl" | head -1)

if [ ! -f "$VECTORS_FILE" ]; then
    echo "Warning: No vectors file found in $VECTORS_DIR"
    echo "Skipping interpolation generation"
else
    echo "Found vectors file: $VECTORS_FILE"

    # Extract available expressions from vectors file
    EXPRESSIONS=$(python -c "
import pickle
with open('$VECTORS_FILE', 'rb') as f:
    vectors = pickle.load(f)
for expr in sorted(vectors.keys()):
    print(expr)
" 2>>"$VECTORS_DIR/debug.log")

    TARGET_ITEMS=$(python - <<PY
import os
path = '$TARGET_DIR'
exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
fnames = []
for root, _dirs, files in os.walk(path):
    for f in files:
        if os.path.splitext(f)[1].lower() in exts:
            rel = os.path.relpath(os.path.join(root, f), start=path).replace('\\\\', '/')
            fnames.append(rel)
fnames = sorted(fnames)
for i, rel in enumerate(fnames):
    stem = os.path.splitext(os.path.basename(rel))[0]
    print(f"{i}:{stem}")
PY
 2>>"$TARGET_OUTPUT_DIR/debug.log")

    if [ -z "$EXPRESSIONS" ]; then
        echo "Warning: Could not extract expressions from vectors file"
    elif [ -z "$TARGET_ITEMS" ]; then
        echo "Warning: Could not parse target dataset.json from $TARGET_DIR"
    else
        mkdir -p "$INTERP_OUTPUT_DIR"
        echo "Found expressions: $EXPRESSIONS"

        for target_item in $TARGET_ITEMS; do
            target_idx=$(echo "$target_item" | cut -d: -f1)
            target_name=$(echo "$target_item" | cut -d: -f2)

            BASE_W_PATH="$TARGET_OUTPUT_DIR"/*"/$target_idx/projected_w.npz"
            if ! compgen -G "$BASE_W_PATH" > /dev/null; then
                echo "Warning: Missing projected_w.npz for target index $target_idx ($target_name), skipping"
                continue
            fi
            BASE_W_PATH=$(compgen -G "$BASE_W_PATH" | head -1)

            for expr in $EXPRESSIONS; do
                echo ""
                echo "Generating interpolation for target=$target_name expression=$expr"

                python apply_expression_vectors.py \
                    --network="$NETWORK" \
                    --base-w="$BASE_W_PATH" \
                    --dataset-json="$TARGET_DIR/dataset.json" \
                    --base-idx="$target_idx" \
                    --vectors="$VECTORS_FILE" \
                    --expression="$expr" \
                    --strength="$STRENGTH" \
                    --steps="$INTERP_STEPS" \
                    --outdir="$INTERP_OUTPUT_DIR/$target_name/$expr" \
                    --freeze-layers=4 \
                    --fps=30 \
                    --save-frames=false || echo "  Warning: Failed target=$target_name expression=$expr"
            done
        done

        echo ""
        echo "✓ Interpolation videos generated for crop_img targets"
    fi
fi

# ============================================================================
# STEP 5: Summary and next steps
# ============================================================================

echo ""
echo "========================================="
echo "PIPELINE COMPLETE"
echo "========================================="
echo ""
echo "Results:"
echo "  • Expression PTI embeddings: $OUTPUT_DIR"
echo "  • Target PTI embeddings: $TARGET_OUTPUT_DIR"
echo "  • Expression vectors: $VECTORS_DIR"
if [ -d "$INTERP_OUTPUT_DIR" ]; then
    echo "  • Interpolation videos on crop_img: $INTERP_OUTPUT_DIR"
fi
echo ""
echo "Next steps:"
echo "  1. Review vector analysis in: $VECTORS_DIR/vector_magnitudes.png"
echo "  2. Watch interpolation videos in: $INTERP_OUTPUT_DIR/"
echo "  3. Compare expressions: python compare_expressions.py \\" 
echo "       --network=$NETWORK \\" 
echo "       --base-w=<target_projected_w.npz> \\" 
echo "       --vectors=$VECTORS_FILE \\" 
echo "       --outdir=comparisons"
echo ""

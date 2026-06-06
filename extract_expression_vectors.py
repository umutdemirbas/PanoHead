"""Extract and apply expression editing vectors from PTI embeddings."""

import os
import numpy as np
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.6;8.0;7.5')
import torch
import click
import pickle
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import json

import dnnlib
import legacy


def load_w_vector(npz_path: str) -> np.ndarray:
    """Load w vector from projected_w.npz file."""
    data = np.load(npz_path)
    w = data['w']
    return w.squeeze()  # Remove batch dimension if present


def load_dataset_labels(pti_output_dir: str) -> Dict[int, str]:
    """
    Load dataset.json to map indices to expression names.
    """
    labels_dict = {}
    
    # Load from dataset.json
    search_paths = [
        Path('dataset/expressions/dataset.json'),  # From PanoHead root
        Path('../../dataset/expressions/dataset.json'),  # From pti_out_expressions
        Path(pti_output_dir).parent.parent / 'dataset' / 'expressions' / 'dataset.json',
        Path(pti_output_dir).parent / 'dataset.json',  # In pti output dir
    ]
    
    for dataset_json_path in search_paths:
        if dataset_json_path.exists():
            try:
                with open(dataset_json_path, 'r') as f:
                    data = json.load(f)
                    for idx, (img_path, _) in enumerate(data.get('labels', [])):
                        # Extract expression name: "laugh/umut.jpg" → "laugh", or "laugh_umut.jpg" → "laugh_umut"
                        img_path = img_path.replace('\\', '/')
                        expr_name = img_path.split('/')[0] if '/' in img_path else img_path.replace('.jpg', '').replace('.png', '')
                        labels_dict[idx] = expr_name
                    print(f"Loaded dataset labels from: {dataset_json_path}")
                    return labels_dict
            except Exception as e:
                print(f"Error reading {dataset_json_path}: {e}")
                pass
    
    print("Warning: Could not find mapping file or dataset.json, using numeric indices")
    return labels_dict


def compute_expression_vector(neutral_w_path: str, expression_w_path: str) -> np.ndarray:
    """Compute editing vector as difference between two w embeddings."""
    neutral_w = load_w_vector(neutral_w_path)
    expression_w = load_w_vector(expression_w_path)
    
    # Ensure same shape
    if neutral_w.shape != expression_w.shape:
        print(f"Warning: shape mismatch {neutral_w.shape} vs {expression_w.shape}")
        min_len = min(neutral_w.shape[0], expression_w.shape[0])
        neutral_w = neutral_w[:min_len]
        expression_w = expression_w[:min_len]
    
    editing_vector = expression_w - neutral_w
    return editing_vector


def extract_all_vectors(pti_output_dir: str, base_idx: int = 2) -> Dict[str, np.ndarray]:
    """
    Extract editing vectors for all expressions relative to base expression.
    Uses dataset.json indexing as the canonical index throughout.
    
    Args:
        pti_output_dir: Path to PTI output directory
        base_idx: Index of base expression in dataset.json (default 2 for 'neutral')
    """
    vectors = {}
    pti_path = Path(pti_output_dir)
    
    # Attempt to locate dataset.json and build mappings between dataset.json order
    # and the on-disk sorted image ordering (ImageFolderDataset uses sorted filenames).
    idx_to_name = {}
    dataset_to_sorted = {}
    dataset_json_path = None
    # search common locations
    possible = [
        Path('dataset/expressions/dataset.json'),
        Path('../../dataset/expressions/dataset.json'),
        Path(pti_output_dir).parent.parent / 'dataset' / 'expressions' / 'dataset.json',
        Path(pti_output_dir).parent / 'dataset.json',
    ]
    for p in possible:
        if p.exists():
            dataset_json_path = p
            break

    sorted_fnames = []
    relpath_to_sorted_idx = {}
    if dataset_json_path is not None:
        try:
            with open(dataset_json_path, 'r') as f:
                data = json.load(f)
                labels = data.get('labels', [])
                # build idx_to_name from dataset.json ordering
                for idx, (img_path, _) in enumerate(labels):
                    img_path = img_path.replace('\\', '/')
                    expr_name = img_path.split('/')[0] if '/' in img_path else os.path.splitext(img_path)[0]
                    idx_to_name[idx] = expr_name

            # Build sorted filename list from the dataset directory on disk
            dataset_dir = dataset_json_path.parent
            exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
            for root, _dirs, files in os.walk(dataset_dir):
                for f in files:
                    if os.path.splitext(f)[1].lower() in exts:
                        rel = os.path.relpath(os.path.join(root, f), start=dataset_dir).replace('\\', '/')
                        sorted_fnames.append(rel)
            sorted_fnames = sorted(sorted_fnames)
            for i, rel in enumerate(sorted_fnames):
                relpath_to_sorted_idx[rel] = i

            # Build mapping dataset_idx -> sorted_idx using full relative paths.
            # Basenames are not safe here because many datasets reuse the same
            # filename under different expression folders (e.g. laugh/umut.jpg).
            for dataset_idx, (img_path, _) in enumerate(json.load(open(dataset_json_path))['labels']):
                rel_path = img_path.replace('\\', '/')
                if rel_path in relpath_to_sorted_idx:
                    dataset_to_sorted[dataset_idx] = relpath_to_sorted_idx[rel_path]
                else:
                    dataset_to_sorted[dataset_idx] = dataset_idx

            print(f"Loaded dataset.json from: {dataset_json_path}; found {len(sorted_fnames)} image files")
        except Exception as e:
            print(f"Warning: could not parse dataset.json: {e}")

    if not idx_to_name:
        print("Warning: Could not load dataset labels, using numeric indices")

    # Find base w vector using the sorted (on-disk) PTI index mapping
    base_w_path = None
    base_pti_idx = dataset_to_sorted.get(base_idx, base_idx)

    for model_dir in pti_path.iterdir():
        if model_dir.is_dir():
            idx_dir = model_dir / str(base_pti_idx)
            if idx_dir.is_dir():
                candidate = idx_dir / 'projected_w.npz'
                if candidate.exists():
                    base_w_path = candidate
                    break

    if not base_w_path or not base_w_path.exists():
        print(f"Could not find base expression w vector at PTI index {base_pti_idx} (dataset index {base_idx})")
        return vectors

    base_name = idx_to_name.get(base_idx, f"index_{base_idx}")
    print(f"Base expression: {base_name} (dataset index {base_idx}, PTI index {base_pti_idx})")
    print(f"Base w loaded from: {base_w_path}")

    # Compute vectors for all other expressions using dataset->sorted mapping
    for dataset_idx in range(max(1, max(idx_to_name.keys())+1) if idx_to_name else 0):
        if dataset_idx != base_idx:
            expr_name = idx_to_name.get(dataset_idx, f"index_{dataset_idx}")
            pti_idx = dataset_to_sorted.get(dataset_idx, dataset_idx)
            
            # Find w vector for this expression at its PTI index
            for model_dir in pti_path.iterdir():
                if model_dir.is_dir():
                    idx_dir = model_dir / str(pti_idx)
                    if idx_dir.is_dir():
                        w_path = idx_dir / 'projected_w.npz'
                        if w_path.exists():
                            try:
                                editing_vec = compute_expression_vector(str(base_w_path), str(w_path))
                                vectors[expr_name] = editing_vec
                                print(f"Extracted vector for: {expr_name} (dataset idx {dataset_idx}, PTI idx {pti_idx}, shape: {editing_vec.shape})")
                            except Exception as e:
                                print(f"Error processing {expr_name}: {e}")
                            break
    
    return vectors


def apply_expression_vector(base_w: np.ndarray, editing_vector: np.ndarray, 
                           strength: float = 1.0) -> np.ndarray:
    """Apply expression editing vector to a base w embedding."""
    return base_w + strength * editing_vector


def generate_interpolation(base_w: np.ndarray, editing_vector: np.ndarray, 
                          num_steps: int = 10) -> np.ndarray:
    """Generate interpolation between base expression and edited expression."""
    strengths = np.linspace(0, 1, num_steps)
    interpolations = []
    
    for strength in strengths:
        w = apply_expression_vector(base_w, editing_vector, strength)
        interpolations.append(w)
    
    return np.array(interpolations)


@click.command()
@click.option('--pti-dir', 'pti_output_dir', help='PTI output directory', 
              required=True, metavar='DIR')
@click.option('--base-idx', 'base_idx', help='Index of base expression in dataset.json', 
              type=int, default=2, show_default=True)
@click.option('--outdir', help='Output directory for vectors and analysis', 
              required=True, metavar='DIR')
@click.option('--visualize', type=bool, help='Generate visualization plots', 
              default=True, show_default=True)
def main(pti_output_dir: str, base_idx: int, outdir: str, visualize: bool):
    """Extract expression vectors from PTI embeddings and save for reuse."""
    
    os.makedirs(outdir, exist_ok=True)
    
    print(f"Extracting expression vectors from: {pti_output_dir}")
    print(f"Base expression index: {base_idx}")
    
    # Extract all vectors
    expression_vectors = extract_all_vectors(pti_output_dir, base_idx)
    
    if not expression_vectors:
        print("No expression vectors found!")
        return
    
    # Load labels for output naming
    idx_to_name = load_dataset_labels(pti_output_dir)
    base_name = idx_to_name.get(base_idx, f"index_{base_idx}")
    
    # Save vectors
    vectors_file = os.path.join(outdir, f'expression_vectors_{base_name}.pkl')
    with open(vectors_file, 'wb') as f:
        pickle.dump(expression_vectors, f)
    print(f"\nSaved {len(expression_vectors)} vectors to: {vectors_file}")
    
    # Save individual vectors as numpy files for easy inspection
    vectors_dir = os.path.join(outdir, 'vectors')
    os.makedirs(vectors_dir, exist_ok=True)
    
    for expr_name, vector in expression_vectors.items():
        vec_file = os.path.join(vectors_dir, f'{expr_name}_vector.npy')
        np.save(vec_file, vector)
        print(f"  - {expr_name}: shape {vector.shape}, norm {np.linalg.norm(vector):.4f}")
    
    # Generate analysis
    print("\n" + "="*60)
    print("EXPRESSION VECTOR ANALYSIS")
    print("="*60)
    
    vector_norms = {name: np.linalg.norm(vec) for name, vec in expression_vectors.items()}
    vector_norms_sorted = sorted(vector_norms.items(), key=lambda x: x[1], reverse=True)
    
    print("\nVector magnitudes (sorted by strength):")
    for expr, norm in vector_norms_sorted:
        print(f"  {expr:20s}: {norm:10.4f}")
    
    # Compute pairwise distances between expression vectors
    print("\nPairwise distances between expression vectors:")
    expr_names = list(expression_vectors.keys())
    for i, expr1 in enumerate(expr_names):
        for expr2 in expr_names[i+1:]:
            dist = np.linalg.norm(expression_vectors[expr1] - expression_vectors[expr2])
            print(f"  {expr1:20s} <-> {expr2:20s}: {dist:10.4f}")
    
    # Visualization
    if visualize:
        print("\nGenerating visualizations...")
        
        # Plot 1: Vector magnitudes
        fig, ax = plt.subplots(figsize=(10, 6))
        names = [name for name, _ in vector_norms_sorted]
        norms = [norm for _, norm in vector_norms_sorted]
        ax.barh(names, norms, color='steelblue')
        ax.set_xlabel('Vector Magnitude (L2 norm)')
        ax.set_title(f'Expression Vector Strengths (base: {base_name})')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'vector_magnitudes.png'), dpi=150)
        print(f"  Saved: vector_magnitudes.png")
        
        # Plot 2: Distance heatmap
        if len(expr_names) > 1:
            fig, ax = plt.subplots(figsize=(8, 8))
            distances = np.zeros((len(expr_names), len(expr_names)))
            for i, expr1 in enumerate(expr_names):
                for j, expr2 in enumerate(expr_names):
                    distances[i, j] = np.linalg.norm(
                        expression_vectors[expr1] - expression_vectors[expr2]
                    )
            
            im = ax.imshow(distances, cmap='viridis')
            ax.set_xticks(range(len(expr_names)))
            ax.set_yticks(range(len(expr_names)))
            ax.set_xticklabels(expr_names, rotation=45, ha='right')
            ax.set_yticklabels(expr_names)
            ax.set_title(f'Expression Vector Distance Matrix')
            plt.colorbar(im, ax=ax, label='L2 Distance')
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, 'vector_distances.png'), dpi=150)
            print(f"  Saved: vector_distances.png")
        
        plt.close('all')
    
    print(f"\nAll results saved to: {outdir}")
    print(f"Vectors pickle file: {vectors_file}")


if __name__ == "__main__":
    main()

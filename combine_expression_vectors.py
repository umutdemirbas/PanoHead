"""Combine and average expression vectors from multiple faces for robustness."""

import os
import numpy as np
import click
import pickle
from pathlib import Path
from typing import Dict, List
import json


def load_w_vector(npz_path: str) -> np.ndarray:
    """Load w vector from projected_w.npz file."""
    data = np.load(npz_path)
    w = data['w']
    return w.squeeze()


def load_vectors_from_dir(vectors_dir: str) -> Dict[str, List[np.ndarray]]:
    """Load all vectors from a directory containing multiple vector files."""
    vectors_by_expr = {}
    
    vectors_path = Path(vectors_dir)
    if not vectors_path.exists():
        return vectors_by_expr
    
    for vec_file in sorted(vectors_path.glob('*_vector.npy')):
        expr_name = vec_file.stem.replace('_vector', '')
        vec = np.load(str(vec_file))
        
        if expr_name not in vectors_by_expr:
            vectors_by_expr[expr_name] = []
        vectors_by_expr[expr_name].append(vec)
    
    return vectors_by_expr


def compute_robust_vectors(vectors_by_expr: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    """Compute robust (averaged) vectors from multiple instances."""
    robust_vectors = {}
    vector_stats = {}
    
    for expr_name, vec_list in vectors_by_expr.items():
        if not vec_list:
            continue
        
        # Stack and average
        vec_stack = np.array(vec_list)
        mean_vec = np.mean(vec_stack, axis=0)
        std_vec = np.std(vec_stack, axis=0)
        
        robust_vectors[expr_name] = mean_vec
        vector_stats[expr_name] = {
            'num_instances': len(vec_list),
            'mean_norm': float(np.linalg.norm(mean_vec)),
            'std_norm': float(np.std([np.linalg.norm(v) for v in vec_list])),
            'mean_std': float(np.mean(std_vec)),
        }
    
    return robust_vectors, vector_stats


def compute_vector_reliability(vectors_by_expr: Dict[str, List[np.ndarray]]) -> Dict[str, float]:
    """Compute reliability score based on consistency across instances."""
    reliability = {}
    
    for expr_name, vec_list in vectors_by_expr.items():
        if len(vec_list) < 2:
            reliability[expr_name] = 0.5  # Not enough samples
            continue
        
        # Compute pairwise cosine similarities
        norms = [np.linalg.norm(v) for v in vec_list]
        max_norm = max(norms)
        
        similarities = []
        for i in range(len(vec_list)):
            for j in range(i + 1, len(vec_list)):
                cos_sim = np.dot(vec_list[i], vec_list[j]) / (norms[i] * norms[j] + 1e-8)
                similarities.append(cos_sim)
        
        if similarities:
            mean_similarity = np.mean(similarities)
            reliability[expr_name] = (mean_similarity + 1) / 2  # Convert [-1, 1] to [0, 1]
        else:
            reliability[expr_name] = 0.5
    
    return reliability


@click.command()
@click.option('--input-dirs', 'input_dirs', multiple=True, required=True,
              help='Input directories with expression_vectors (can specify multiple)')
@click.option('--vectors-subdir', default='vectors',
              help='Subdirectory name containing vector files', show_default=True)
@click.option('--outdir', help='Output directory', required=True, metavar='DIR')
@click.option('--min-instances', type=int, default=1,
              help='Minimum number of instances required to compute average', show_default=True)

def main(input_dirs: tuple, vectors_subdir: str, outdir: str, min_instances: int):
    """Combine expression vectors from multiple faces for robustness."""
    
    os.makedirs(outdir, exist_ok=True)
    
    print(f"Loading expression vectors from {len(input_dirs)} directories...")
    
    # Collect all vectors by expression
    all_vectors_by_expr = {}
    
    for input_dir in input_dirs:
        vectors_dir = os.path.join(input_dir, vectors_subdir)
        print(f"  Reading from: {vectors_dir}")
        
        vectors_by_expr = load_vectors_from_dir(vectors_dir)
        
        for expr_name, vec_list in vectors_by_expr.items():
            if expr_name not in all_vectors_by_expr:
                all_vectors_by_expr[expr_name] = []
            all_vectors_by_expr[expr_name].extend(vec_list)
    
    # Filter by minimum instances
    filtered_vectors = {
        expr: vecs for expr, vecs in all_vectors_by_expr.items()
        if len(vecs) >= min_instances
    }
    
    print(f"\nCollected {len(filtered_vectors)} expressions:")
    for expr, vecs in sorted(filtered_vectors.items()):
        print(f"  {expr:20s}: {len(vecs)} instances")
    
    if not filtered_vectors:
        print("No expressions met minimum instance requirement!")
        return
    
    # Compute robust vectors
    print("\nComputing robust (averaged) vectors...")
    robust_vectors, vector_stats = compute_robust_vectors(filtered_vectors)
    
    # Compute reliability scores
    print("Computing reliability scores...")
    reliability = compute_vector_reliability(filtered_vectors)
    
    # Display analysis
    print("\n" + "="*70)
    print("ROBUST VECTOR ANALYSIS")
    print("="*70)
    print(f"{'Expression':<20} {'Instances':<12} {'Norm':<12} {'Std Norm':<12} {'Reliability':<12}")
    print("-"*70)
    
    for expr_name in sorted(vector_stats.keys()):
        stats = vector_stats[expr_name]
        rel = reliability[expr_name]
        print(f"{expr_name:<20} {stats['num_instances']:<12} "
              f"{stats['mean_norm']:<12.4f} {stats['std_norm']:<12.4f} {rel:<12.2f}")
    
    print("\nReliability interpretation:")
    print("  > 0.9: Highly reliable (consistent across instances)")
    print("  > 0.7: Reliable (mostly consistent)")
    print("  > 0.5: Moderate (some variation)")
    print("  < 0.5: Low confidence")
    
    # Save robust vectors
    output_file = os.path.join(outdir, 'robust_expression_vectors.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(robust_vectors, f)
    print(f"\nSaved robust vectors to: {output_file}")
    
    # Save individual vectors
    vectors_out_dir = os.path.join(outdir, 'robust_vectors')
    os.makedirs(vectors_out_dir, exist_ok=True)
    
    for expr_name, vector in robust_vectors.items():
        vec_file = os.path.join(vectors_out_dir, f'{expr_name}_vector.npy')
        np.save(vec_file, vector)
    
    print(f"Saved individual vectors to: {vectors_out_dir}")
    
    # Save metadata
    metadata = {
        'num_expressions': len(robust_vectors),
        'total_instances': sum(len(v) for v in filtered_vectors.values()),
        'expressions': {}
    }
    
    for expr_name in sorted(vector_stats.keys()):
        metadata['expressions'][expr_name] = {
            'vector_norm': vector_stats[expr_name]['mean_norm'],
            'std_norm': vector_stats[expr_name]['std_norm'],
            'num_instances': vector_stats[expr_name]['num_instances'],
            'reliability': float(reliability[expr_name]),
        }
    
    metadata_file = os.path.join(outdir, 'robust_vectors_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata to: {metadata_file}")
    
    # Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    high_rel = [e for e, r in reliability.items() if r > 0.8]
    low_rel = [e for e, r in reliability.items() if r < 0.6]
    
    if high_rel:
        print(f"\nHighly reliable expressions (safe to use): {', '.join(high_rel)}")
    
    if low_rel:
        print(f"\nLow reliability expressions (use with caution): {', '.join(low_rel)}")
        print("  Consider collecting more expression instances for these.")
    
    if len(filtered_vectors) < 3:
        print("\nWarning: Only a few expressions collected.")
        print("Consider adding more expression instances for better results.")
    
    print(f"\nNext step: Use robust vectors with apply_expression_vectors.py")
    print(f"  python apply_expression_vectors.py \\")
    print(f"    --vectors {output_file} \\")
    print(f"    --expression smile \\")
    print(f"    --strength 1.0 \\")
    print(f"    ... (other options)")


if __name__ == "__main__":
    main()

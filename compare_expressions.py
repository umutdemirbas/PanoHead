"""Compare multiple expression vectors applied to the same base image."""

import os
import numpy as np
import torch
import torch.nn.functional as F
import click
import pickle
from pathlib import Path
import imageio
from tqdm import tqdm

import dnnlib
import legacy


def load_w_vector(npz_path: str) -> np.ndarray:
    """Load w vector from projected_w.npz file."""
    data = np.load(npz_path)
    w = data['w']
    return w.squeeze()


def load_vectors(vectors_file: str) -> dict:
    """Load expression vectors from pickle file."""
    with open(vectors_file, 'rb') as f:
        vectors = pickle.load(f)
    return vectors


def apply_expression_vector(base_w: np.ndarray, editing_vector: np.ndarray, 
                           strength: float = 1.0) -> np.ndarray:
    """Apply expression editing vector to a base w embedding."""
    return base_w + strength * editing_vector


def render_w_to_image(G, w: np.ndarray, c: np.ndarray, 
                     device: torch.device) -> np.ndarray:
    """Render a single w latent code to an image."""
    w = torch.from_numpy(w).unsqueeze(0).to(device).float()
    c = torch.from_numpy(c).unsqueeze(0).to(device).float()
    
    with torch.no_grad():
        synth = G.synthesis(w, c=c, noise_mode='const')['image']
        synth = (synth + 1) * (255 / 2)
        synth = synth.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    
    return synth


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--base-w', 'base_w_path', help='Base w vector (projected_w.npz)', required=True)
@click.option('--camera', 'camera_path', help='Camera matrix (from dataset label)', required=False)
@click.option('--vectors', 'vectors_file', help='Expression vectors pickle file', required=True)
@click.option('--strength', type=float, help='Strength for all expressions', 
              default=1.0, show_default=True)
@click.option('--outdir', help='Output directory', required=True, metavar='DIR')
@click.option('--layout', type=click.Choice(['grid', 'horizontal', 'vertical']),
              help='Layout for comparison', default='grid', show_default=True)
@click.option('--cols', type=int, help='Number of columns for grid layout',
              default=3, show_default=True)

def main(network_pkl: str, base_w_path: str, camera_path: str, vectors_file: str,
         strength: float, outdir: str, layout: str, cols: int):
    """Generate comparison images of multiple expressions applied to same base."""
    
    os.makedirs(outdir, exist_ok=True)
    device = torch.device('cuda')
    
    print(f"Loading network from: {network_pkl}")
    with dnnlib.util.open_url(network_pkl) as fp:
        network_data = legacy.load_network_pkl(fp)
        G = network_data['G_ema'].requires_grad_(False).to(device)
    
    print(f"Loading base w vector from: {base_w_path}")
    base_w = load_w_vector(base_w_path)
    
    print(f"Loading expression vectors from: {vectors_file}")
    expression_vectors = load_vectors(vectors_file)
    
    # Load or create camera matrix
    if camera_path and os.path.exists(camera_path):
        print(f"Loading camera matrix from: {camera_path}")
        c = np.load(camera_path)
    else:
        print("Using default front-facing camera")
        cam2world = np.eye(4)
        cam2world[0, 3] = 0
        cam2world[1, 3] = 0
        cam2world[2, 3] = 2.7
        intrinsics = np.array([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]])
        c = np.concatenate([cam2world.reshape(-1), intrinsics.reshape(-1)]).astype(np.float32)
    
    if c.ndim == 1:
        c = c.reshape(1, -1)
    
    # Also render the base image (neutral)
    print(f"\nRendering {len(expression_vectors) + 1} expression variations...")
    
    images = {}
    
    # Neutral (base)
    base_image = render_w_to_image(G, base_w, c[0], device)
    images['neutral'] = base_image
    print(f"  ✓ neutral (base)")
    
    # All expressions
    for expr_name in tqdm(sorted(expression_vectors.keys()), desc="Rendering expressions"):
        editing_vector = expression_vectors[expr_name]
        w_edited = apply_expression_vector(base_w, editing_vector, strength)
        image = render_w_to_image(G, w_edited, c[0], device)
        images[expr_name] = image
    
    # Create comparison image based on layout
    print(f"\nCreating {layout} layout comparison...")
    
    if layout == 'grid':
        # Grid layout
        num_images = len(images)
        rows = (num_images + cols - 1) // cols
        
        img_h, img_w = base_image.shape[:2]
        grid_h = rows * img_h + (rows + 1) * 5
        grid_w = cols * img_w + (cols + 1) * 5
        
        grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
        
        try:
            import cv2
            font = cv2.FONT_HERSHEY_SIMPLEX
            has_cv2 = True
        except ImportError:
            has_cv2 = False
        
        for idx, (expr_name, image) in enumerate(sorted(images.items())):
            row = idx // cols
            col = idx % cols
            y = row * img_h + (row + 1) * 5
            x = col * img_w + (col + 1) * 5
            
            grid[y:y+img_h, x:x+img_w] = image
            
            if has_cv2:
                text = expr_name.replace('_', ' ').title()
                text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
                text_x = x + (img_w - text_size[0]) // 2
                text_y = y - 5
                cv2.putText(grid, text, (text_x, text_y), font, 0.7, (0, 0, 0), 2)
        
        output_path = os.path.join(outdir, f'comparison_grid_strength{strength:.1f}.png')
        import PIL.Image
        PIL.Image.fromarray(grid).save(output_path)
        print(f"Saved grid comparison: {output_path}")
    
    elif layout == 'horizontal':
        # Horizontal layout (all in one row)
        images_list = [images[k] for k in sorted(images.keys())]
        combined = np.concatenate(images_list, axis=1)
        output_path = os.path.join(outdir, f'comparison_horizontal_strength{strength:.1f}.png')
        import PIL.Image
        PIL.Image.fromarray(combined).save(output_path)
        print(f"Saved horizontal comparison: {output_path}")
    
    elif layout == 'vertical':
        # Vertical layout (all in one column)
        images_list = [images[k] for k in sorted(images.keys())]
        combined = np.concatenate(images_list, axis=0)
        output_path = os.path.join(outdir, f'comparison_vertical_strength{strength:.1f}.png')
        import PIL.Image
        PIL.Image.fromarray(combined).save(output_path)
        print(f"Saved vertical comparison: {output_path}")
    
    # Save individual images
    images_dir = os.path.join(outdir, 'individual_expressions')
    os.makedirs(images_dir, exist_ok=True)
    
    import PIL.Image
    for expr_name, image in images.items():
        img_path = os.path.join(images_dir, f'{expr_name}_strength{strength:.1f}.png')
        PIL.Image.fromarray(image).save(img_path)
    
    print(f"Saved individual expressions to: {images_dir}")
    print("\nExpression comparison complete!")


if __name__ == "__main__":
    main()

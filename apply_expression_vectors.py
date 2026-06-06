"""Apply expression editing vectors to generate interpolation videos."""

import os
import numpy as np
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.6;8.0;7.5')
import torch
import click
import pickle
import imageio
import PIL.Image
from pathlib import Path
from tqdm import tqdm
import json

import dnnlib
import legacy


def load_w_vector(npz_path: str) -> np.ndarray:
    """Load w vector from projected_w.npz file."""
    data = np.load(npz_path)
    return np.asarray(data['w'], dtype=np.float32).squeeze()


def load_vectors(vectors_file: str) -> dict:
    """Load expression vectors from pickle file."""
    with open(vectors_file, 'rb') as f:
        vectors = pickle.load(f)
    return {name: np.asarray(vector, dtype=np.float32) for name, vector in vectors.items()}


def load_camera_from_dataset(dataset_json_path: str, image_idx: int) -> np.ndarray:
    """Load camera matrix from dataset.json for a specific image index."""
    try:
        with open(dataset_json_path, 'r') as f:
            data = json.load(f)
            labels = data.get('labels', [])
            if image_idx < len(labels):
                camera_values = labels[image_idx][1]
                # Convert string values to float
                c = np.array([float(v) for v in camera_values], dtype=np.float32)
                return c
    except Exception as e:
        print(f"Warning: Could not load camera from dataset: {e}")
    return None


def get_sorted_filenames(dataset_dir: str):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    fnames = []
    for root, _dirs, files in os.walk(dataset_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                rel = os.path.relpath(os.path.join(root, f), start=dataset_dir).replace('\\', '/')
                fnames.append(rel)
    return sorted(fnames)


def load_camera_by_sorted_index(dataset_json_path: str, dataset_dir: str, sorted_index: int) -> np.ndarray:
    """Load camera matrix by using the sorted on-disk filename ordering and matching to dataset.json labels."""
    try:
        sorted_fnames = get_sorted_filenames(dataset_dir)
        if sorted_index < 0 or sorted_index >= len(sorted_fnames):
            return None
        target_basename = os.path.basename(sorted_fnames[sorted_index])
        with open(dataset_json_path, 'r') as f:
            data = json.load(f)
            labels = data.get('labels', [])
            for img_path, cam_vals in labels:
                if os.path.basename(img_path.replace('\\', '/')) == target_basename:
                    return np.array([float(v) for v in cam_vals], dtype=np.float32)
    except Exception as e:
        print(f"Warning: could not load camera by sorted index: {e}")
    return None


def apply_expression_vector(base_w: np.ndarray, editing_vector: np.ndarray,
                           strength: float = 1.0, freeze_layers: int = 6) -> np.ndarray:
    """Apply expression editing vector while preserving the coarse identity layers."""
    edited_w = np.array(base_w, copy=True)

    if edited_w.ndim == 2 and editing_vector.ndim == 2:
        start_layer = min(max(freeze_layers, 0), edited_w.shape[0])
        edited_w[start_layer:] = edited_w[start_layer:] + strength * editing_vector[start_layer:]
        return edited_w

    return edited_w + strength * editing_vector


def render_w_to_image(G, w: np.ndarray, c: np.ndarray, 
                     device: torch.device, debug: bool = False) -> np.ndarray:
    """Render a single w latent code to an image."""
    # Ensure w is properly shaped
    if w.ndim > 1:
        w = w.squeeze()
    
    # Ensure c is 1D before conversion
    if c.ndim == 2:
        c = c[0]
    
    w_tensor = torch.from_numpy(w).unsqueeze(0).to(device).float()
    c_tensor = torch.from_numpy(c).unsqueeze(0).to(device).float()
    
    with torch.no_grad():
        synth = G.synthesis(w_tensor, c=c_tensor, noise_mode='const')['image']
        synth = (synth + 1) * (255 / 2)
        synth = synth.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    
    return synth


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--base-w', 'base_w_path', help='Base w vector (projected_w.npz)', required=True)
@click.option('--camera', 'camera_path', help='Camera matrix (from dataset label)', required=False)
@click.option('--dataset-json', 'dataset_json_path', help='Path to dataset.json used to load camera by base index', required=False)
@click.option('--vectors', 'vectors_file', help='Expression vectors pickle file', required=True)
@click.option('--expression', 'expression_name', help='Expression to apply', required=True)
@click.option('--strength', type=float, help='Maximum strength for interpolation', 
              default=1.0, show_default=True)
@click.option('--steps', type=int, help='Number of interpolation steps', 
              default=20, show_default=True)
@click.option('--freeze-layers', type=int, help='Number of coarse W+ layers to keep from the base latent',
              default=5, show_default=True)
@click.option('--outdir', help='Output directory', required=True, metavar='DIR')
@click.option('--fps', type=int, help='Video FPS', default=30, show_default=True)
@click.option('--save-frames', type=bool, help='Save individual frames', 
              default=False, show_default=True)
@click.option('--base-idx', type=int, help='Index of base image in dataset.json', 
              default=2, show_default=True)

def main(network_pkl: str, base_w_path: str, camera_path: str, dataset_json_path: str,
         vectors_file: str,
         expression_name: str, strength: float, steps: int, freeze_layers: int, outdir: str, fps: int, 
         save_frames: bool, base_idx: int):
    """Apply expression editing vector to generate interpolation video."""
    
    os.makedirs(outdir, exist_ok=True)
    device = torch.device('cuda')
    
    print(f"Loading network from: {network_pkl}")
    with dnnlib.util.open_url(network_pkl) as fp:
        network_data = legacy.load_network_pkl(fp)
        G = network_data['G_ema'].requires_grad_(False).eval().to(device)
    
    print(f"Loading base w vector from: {base_w_path}")
    base_w = load_w_vector(base_w_path)
    
    print(f"Loading expression vectors from: {vectors_file}")
    expression_vectors = load_vectors(vectors_file)
    
    if expression_name not in expression_vectors:
        print(f"Expression '{expression_name}' not found in vectors!")
        print(f"Available expressions: {list(expression_vectors.keys())}")
        return
    
    editing_vector = np.asarray(expression_vectors[expression_name], dtype=np.float32)
    print(f"Using expression vector: {expression_name}")
    print(f"  Base W shape: {base_w.shape}, dtype: {base_w.dtype}")
    print(f"  Editing vector shape: {editing_vector.shape}, dtype: {editing_vector.dtype}")
    print(f"  Editing vector norm: {np.linalg.norm(editing_vector):.4f}")
    print(f"  Editing vector min/max: {editing_vector.min():.4f} / {editing_vector.max():.4f}")
    print(f"  Base W min/max: {base_w.min():.4f} / {base_w.max():.4f}")
    
    # Load or create camera matrix
    c = None
    
    # First, try a user-provided dataset.json.
    dataset_json_paths = []
    if dataset_json_path:
        dataset_json_paths.append(Path(dataset_json_path))

    # Then try common defaults.
    dataset_json_paths.extend([
        Path('dataset/expressions/dataset.json'),
        Path('dataset/crop_img/dataset.json'),
        Path('../../dataset/expressions/dataset.json'),
        Path('../../dataset/crop_img/dataset.json'),
    ])
    
    for dataset_json_path in dataset_json_paths:
        if dataset_json_path.exists():
            print(f"Found dataset.json at: {dataset_json_path}")
            # Try matching by sorted on-disk ordering first (ImageFolderDataset uses sorted filenames)
            dataset_dir = dataset_json_path.parent
            c = load_camera_by_sorted_index(str(dataset_json_path), str(dataset_dir), base_idx)
            if c is not None:
                print(f"Loaded camera for sorted index {base_idx} from dataset.json at {dataset_json_path}")
                break
            # Fall back to direct index lookup in dataset.json
            c = load_camera_from_dataset(str(dataset_json_path), base_idx)
            if c is not None:
                print(f"Loaded camera for dataset.json index {base_idx} from {dataset_json_path}")
                break
    
    # Fall back to provided camera path
    if c is None and camera_path and os.path.exists(camera_path):
        print(f"Loading camera matrix from: {camera_path}")
        c = np.load(camera_path)
    
    # Fall back to default camera
    if c is None:
        print("Using default front-facing camera")
        # Default front-facing camera for PanoHead
        cam2world = np.eye(4)
        cam2world[0, 3] = 0
        cam2world[1, 3] = 0
        cam2world[2, 3] = 2.7
        intrinsics = np.array([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]])
        c = np.concatenate([cam2world.reshape(-1), intrinsics.reshape(-1)]).astype(np.float32)
    
    # Ensure camera shape is correct
    if c.ndim == 1:
        c = c.reshape(1, -1)
    
    print(f"  Camera shape: {c.shape}, dtype: {c.dtype}")
    
    # Generate interpolation
    print(f"\nGenerating {steps} interpolation frames...")
    strengths = np.linspace(0, strength, steps)
    
    video_path = os.path.join(outdir, f'{expression_name}_interpolation.mp4')
    video = imageio.get_writer(video_path, mode='I', fps=fps, codec='libx264', bitrate='16M')
    
    frames_dir = os.path.join(outdir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    # Render and save base image for reference
    print("Rendering base (neutral) image...")
    base_image = render_w_to_image(G, base_w, c[0], device)
    base_frame_path = os.path.join(frames_dir, 'base_neutral.png')
    PIL.Image.fromarray(base_image).save(base_frame_path)
    print(f"Saved base image to: {base_frame_path}")
    
    for i, s in enumerate(tqdm(strengths, desc="Rendering frames")):
        # Apply expression vector
        w_edited = apply_expression_vector(base_w, editing_vector, s, freeze_layers)
        
        # Debug: check edited w vector
        if i == 0 or i == len(strengths) - 1:
            delta = w_edited - base_w
            print(f"\n  Step {i}: strength={s:.2f}")
            print(f"    Delta norm: {np.linalg.norm(delta):.4f}")
            print(f"    Edited W min/max: {w_edited.min():.4f} / {w_edited.max():.4f}")
        
        # Render image
        image = render_w_to_image(G, w_edited, c[0], device)
        
        # Debug: check rendered image
        if i == 0 or i == len(strengths) - 1:
            print(f"    Image shape: {image.shape}, dtype: {image.dtype}, min/max: {image.min()}/{image.max()}")

        
        # Add text overlay
        text = f"{expression_name.replace('_', ' ').title()} - Strength: {s:.2f}"
        try:
            import cv2
            # Ensure image is C-contiguous for cv2.putText compatibility
            image_cv = np.ascontiguousarray(image)
            cv2.putText(image_cv, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (0, 255, 0), 2)
            image = image_cv
        except ImportError:
            pass  # cv2 not available, skip text overlay
        
        video.append_data(image)
        
        # Always save frames for debugging/inspection
        frame_path = os.path.join(frames_dir, f'frame_{i:03d}.png')
        PIL.Image.fromarray(image).save(frame_path)

    final_latent_path = os.path.join(outdir, f'{expression_name}_edited_projected_w.npz')
    np.savez(final_latent_path, w=np.asarray(w_edited, dtype=np.float32)[np.newaxis, ...])
    print(f"Saved final edited latent to: {final_latent_path}")
    
    video.close()
    print(f"\nSaved video to: {video_path}")
    print(f"Saved frames to: {frames_dir}")
    
    # Save metadata
    metadata = {
        'expression': expression_name,
        'max_strength': strength,
        'num_steps': steps,
        'base_w_shape': base_w.shape,
        'vector_norm': float(np.linalg.norm(editing_vector)),
    }
    
    import json
    metadata_file = os.path.join(outdir, f'{expression_name}_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_file}")


if __name__ == "__main__":
    main()

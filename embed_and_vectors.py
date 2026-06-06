
"""Embed images into latent space and compute editing vectors."""

import os
import sys
import glob
from time import perf_counter
import copy

import click
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

# Add paths
sys.path.append('torch_utils')

import dnnlib
import legacy

from camera_utils import LookAtPoseSampler

def project(
    G,
    target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    c: torch.Tensor,
    *,
    num_steps=1000,
    w_avg_samples=10000,
    initial_learning_rate=0.1,
    initial_noise_factor=0.05,
    lr_rampdown_length=0.25,
    lr_rampup_length=0.05,
    noise_ramp_length=0.75,
    regularize_noise_weight=1e5,
    optimize_noise=False,
    verbose=False,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    camera_lookat_point = torch.tensor([0, 0, 0.0], device=device)
    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=2.7, device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c_samples = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), c_samples.repeat(w_avg_samples, 1))
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)
    w_avg = np.mean(w_samples, axis=0, keepdims=True)
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Fix delta_c
    delta_c = G.t_mapping(torch.from_numpy(np.mean(z_samples, axis=0, keepdims=True)).to(device), c[:1], truncation_psi=1.0, truncation_cutoff=None, update_emas=False)
    delta_c = torch.squeeze(delta_c, 1)
    c[:, 3] += delta_c[:, 0]
    c[:, 7] += delta_c[:, 1]
    c[:, 11] += delta_c[:, 2]

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32) / 255.0 * 2 - 1
    target_images_perc = (target_images + 1) * (255/2)
    if target_images_perc.shape[2] > 256:
        target_images_perc = F.interpolate(target_images_perc, size=(256, 256), mode='area')
    target_features = vgg16(target_images_perc, resize_images=False, return_lpips=True)

    w_avg = torch.tensor(w_avg, dtype=torch.float32, device=device).repeat(1, G.backbone.mapping.num_ws, 1)
    w_opt = w_avg.detach().clone()
    w_opt.requires_grad = True

    if optimize_noise:
        optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
    else:
        optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    if optimize_noise:
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = w_opt + w_noise
        synth_images = G.synthesis(ws, c=c, noise_mode='const')['image']

        # Downsample image to 256x256 if it's larger than that.
        synth_images_perc = (synth_images + 1) * (255/2)
        if synth_images_perc.shape[2] > 256:
            synth_images_perc = torch.nn.functional.interpolate(synth_images_perc, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images_perc, resize_images=False, return_lpips=True)
        perc_loss = (target_features - synth_features).square().sum(1).mean()

        mse_loss = (target_images - synth_images).square().mean()

        w_norm_loss = (w_opt - w_avg).square().mean()

        # Noise regularization.
        reg_loss = 0.0
        if optimize_noise:
            for v in noise_bufs.values():
                noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                    if noise.shape[2] <= 8:
                        break
                    noise = torch.nn.functional.avg_pool2d(noise, kernel_size=2)
        loss = 0.1 * mse_loss + perc_loss + 1.0 * w_norm_loss + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1}/{num_steps} mse: {mse_loss:.2f} perc: {perc_loss:.2f} w_norm: {w_norm_loss:.2f} noise: {reg_loss:.2f}')

        # Normalize noise.
        if optimize_noise:
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

    if w_opt.shape[1] == 1:
        w_opt = w_opt.repeat([1, G.mapping.num_ws, 1])

    return w_opt.detach().cpu()


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--input-dir', help='Directory containing input images', required=True)
@click.option('--output-dir', help='Directory to save outputs', required=True)
@click.option('--num-steps', help='Number of optimization steps', type=int, default=500, show_default=True)
@click.option('--device', help='Device to use', default='cuda', show_default=True)
def main(network_pkl, input_dir, output_dir, num_steps, device):
    device = torch.device(device)

    # Load network
    print(f'Loading network from {network_pkl}...')
    with dnnlib.util.open_url(network_pkl) as fp:
        network_data = legacy.load_network_pkl(fp)
        G = network_data['G_ema'].requires_grad_(False).to(device)
    G.rendering_kwargs["ray_start"] = 2.35

    # Get image files
    image_files = sorted(glob.glob(os.path.join(input_dir, '*.png')) + glob.glob(os.path.join(input_dir, '*.jpg')) + glob.glob(os.path.join(input_dir, '*.jpeg')))
    if not image_files:
        print(f'No images found in {input_dir}')
        return

    latents = {}
    os.makedirs(output_dir, exist_ok=True)

    # Default camera parameters
    camera_lookat_point = torch.tensor([0, 0, 0.0], device=device)
    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=2.7, device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    for img_path in image_files:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f'Embedding {img_name}...')

        # Load and preprocess image
        target_pil = PIL.Image.open(img_path).convert('RGB')
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
        target_uint8 = np.array(target_pil, dtype=np.uint8)

        # Project
        start_time = perf_counter()
        projected_w = project(
            G,
            target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device),
            c=c.clone(),
            num_steps=num_steps,
            device=device,
            verbose=True
        )
        print(f'Elapsed: {perf_counter() - start_time:.1f} s')

        # Save latent
        latent_path = os.path.join(output_dir, f'{img_name}_latent.npz')
        np.savez(latent_path, w=projected_w.numpy())
        latents[img_name] = projected_w.numpy()

    # Compute editing vectors between all pairs
    vectors = {}
    img_names = list(latents.keys())
    for i in range(len(img_names)):
        for j in range(i+1, len(img_names)):
            name1 = img_names[i]
            name2 = img_names[j]
            vector = latents[name2] - latents[name1]  # from name1 to name2
            vectors[f'{name1}_to_{name2}'] = vector

    # Save vectors
    vectors_path = os.path.join(output_dir, 'editing_vectors.npz')
    np.savez(vectors_path, **vectors)
    print(f'Saved latents and vectors to {output_dir}')


if __name__ == '__main__':
    main()
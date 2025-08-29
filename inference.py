import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from dit import DiT
from model import MeanFlow
import os
import argparse
import numpy as np
from PIL import Image


class DiTInference:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract config
        config = checkpoint.get('config', {})
        self.image_size = config.get('image_size', 29)
        self.image_channels = config.get('image_channels', 1)
        self.sigma_min = config.get('sigma_min', 1e-6)
        
        # Initialize DiT model
        print("Initializing DiT model...")
        self.model = DiT(
            input_size=self.image_size,
            patch_size=4,
            in_channels=self.image_channels,
            dim=256,
            depth=8,
            num_heads=4,
            num_classes=10,
            learn_sigma=False,
            class_dropout_prob=0.1,
        ).to(device)
        
        # Load weights (prefer EMA if available)
        if 'ema' in checkpoint:
            print("Loading EMA weights...")
            self.model.load_state_dict(checkpoint['ema'])
        else:
            print("Loading model weights...")
            self.model.load_state_dict(checkpoint['model'])
        
        self.model.eval()
        
        # Initialize sampler with scheduler parameters from checkpoint
        self.sampler = MeanFlow(
            device=device,
            channels=self.image_channels,
            image_size=self.image_size,
            num_classes=10,
            cfg_drop_prob=0.1
        )
        
        print("Model loaded successfully!")
    
    @torch.no_grad()
    def sample(self, num_samples=16, class_labels=None, seed=None):
        """
        Sample images from the model.
        
        Args:
            num_samples: Number of samples to generate
            class_labels: List/tensor of class labels (0-9), or None for random
            cfg_scale: Classifier-free guidance scale
            num_steps: Number of sampling steps
            seed: Random seed for reproducibility
        
        Returns:
            Generated images tensor in [0, 1]
        """
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        
        if class_labels is not None:
            # Handle class labels
            if isinstance(class_labels, int):
                class_labels = torch.full((num_samples,), class_labels, device=self.device)
            elif isinstance(class_labels, list):
                class_labels = torch.tensor(class_labels, device=self.device)
            else:
                class_labels = class_labels.to(self.device)
            
            # Ensure correct number of labels
            if len(class_labels) < num_samples:
                class_labels = class_labels.repeat(num_samples // len(class_labels) + 1)[:num_samples]
        
        # Use the sampler's method
        images = self.sampler.sample_each_class(
            self.model,
            10,
            sample_steps=4
        )
        
        return images



def main():
    parser = argparse.ArgumentParser(description='DiT Direct Image Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Directory to save outputs')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    # Sampling parameters
    parser.add_argument('--num_samples', type=int, default=64,
                       help='Number of samples to generate')
    parser.add_argument('--class_label', type=int, default=None,
                       help='Specific class to sample (0-9), or None for random')
    
    
    args = parser.parse_args()
    
    # Initialize model
    model = DiTInference(args.checkpoint, device=args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
   
    images = model.sample(
        num_samples=args.num_samples,
        class_labels=args.class_label,
        seed=args.seed
    )
    
    # Save individual images
    for i, img in enumerate(images):
        print('img shape: ', img.shape)
        # change 1 channel to 3 channel
        if img.shape[0] == 1: # then repeat
            img = img.repeat(3, 1, 1)
        print('max', img.max(), 'min ', img.min())
        img = img.clamp(-1, 1)
        # img = img.repeat
        img_np = (((img.permute(1, 2, 0).cpu().numpy() + 1) / 2) * 255).astype(np.uint8)
        img_np = np.clip(img_np, 0, 255)
        Image.fromarray(img_np).save(
            os.path.join(args.output_dir, f'sample_{i:04d}.png')
        )
    
    # Save grid
    grid = make_grid(images, nrow=int(np.sqrt(args.num_samples)), normalize=False)
    save_image(grid, os.path.join(args.output_dir, 'samples_grid.png'))
        
    
    print(f"\nGenerated images saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
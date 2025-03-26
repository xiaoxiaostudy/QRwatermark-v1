"""
Distortion attack evaluation script for watermarked images.
This script evaluates the robustness of watermarked images against various types of attacks.
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import kornia
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from network.Network import Network
from util.MyDataloader import MyDataset
from util.distorationattack import *

# Constants
DEFAULT_ATTACKS = [
    {'attack': 'none'},
    *[{'attack': 'erasing', 'scale': jj} for jj in np.linspace(0, 0.35, 10)],
    *[{'attack': 'crop', 'ratio': jj} for jj in np.linspace(0.8, 1, 10)],
    *[{'attack': 'rotation', 'angle': jj} for jj in np.linspace(0, 2.5, 10)],
    *[{'attack': 'brightness', 'min_brightness': 0.5 * jj, 'max_brightness': 1.5 * jj} 
      for jj in range(1, 5)],
    *[{'attack': 'contrast', 'contrast': jj} for jj in [1, 0.75, 0.5, 0.25, 0.15]],
    *[{'attack': 'saturation', 'rnd_sat': jj} for jj in np.linspace(0, 0.8, 10)],
    *[{'attack': 'blurring', 'N_blur': 1 + 2 * jj} for jj in range(1, 10)],
    *[{'attack': 'gnoise', 'rnd_noise': 0.02 * jj} for jj in np.linspace(0.1, 0.8, 10)],
    *[{'attack': 'jpeg', 'quality': 10 * jj} for jj in range(1, 11)]
]

ATTACKS_DICT = {
    "none": Identity(),
    "erasing": Erasing(scale=0.5),
    "crop": Crop(ratio=0.5),
    "rotation": Rotation(),
    "brightness": Brightness(min_brightness=0.8, max_brightness=1.2),
    "contrast": Contrast(contrast=0.5),
    "saturation": Saturation(rnd_sat=1.0),
    "blurring": Blurring(N_blur=7),
    "gnoise": Gnoise(rnd_noise=0.02),
    "jpeg": JPEG(quality=75),   
}

def get_parser() -> argparse.ArgumentParser:
    """Create and return the command line argument parser."""
    parser = argparse.ArgumentParser(description="Watermark Distortion Attack Evaluation")
    group = parser.add_argument_group('Experiments parameters')
    
    group.add_argument("--result_folder", type=str, default="results/DFPW256m64/",
                      help="Result folder path")
    group.add_argument("--model_epoch", type=int, default=59,
                      help="Model epoch to load")
    group.add_argument("--ckpt_path", type=str, default="ckpt/",
                      help="Checkpoint path")
    group.add_argument("--vqvae", type=str, default="vq-f4",
                      help="VQ-VAE model type (vq-f4, vq-f4-noattn, vq-f8, vq-f8-n256, vq-f16)")
    group.add_argument("--strength_factor", type=float, default=1,
                      help="Strength factor for encoding")
    group.add_argument("--dataset_path", type=str, default="datasets/test",
                      help="Path to the dataset")
    group.add_argument("--batch_size", type=int, default=8,
                      help="Batch size for testing")
    group.add_argument("--H", type=int, default=256,
                      help="Height of the images")
    group.add_argument("--W", type=int, default=256,
                      help="Width of the images")
    group.add_argument("--message_length", type=int, default=64,
                      help="Length of the messages")
    group.add_argument("--save_images_number", type=int, default=1,
                      help="Number of images to save")
    group.add_argument("--encoder_weight", type=float, default=10,
                      help="Encoder weight")
    group.add_argument("--encoder_percepweight", type=float, default=0.1,
                      help="Encoder perceptual weight")
    group.add_argument("--discriminator_weight", type=float, default=0.001,
                      help="Discriminator weight")
    group.add_argument("--decoder_weight", type=float, default=10,
                      help="Decoder weight")
    group.add_argument("--lr", type=float, default=4e-5,
                      help="Learning rate")
    
    return parser

def generate_attacks(img_tensor: torch.Tensor, 
                    attacks: List[Dict], 
                    attacks_dict: Dict) -> List[torch.Tensor]:
    """
    Generate a list of attacked images from a tensor image.
    
    Args:
        img_tensor: Input image tensor
        attacks: List of attack configurations
        attacks_dict: Dictionary of attack implementations
        
    Returns:
        List of attacked image tensors
    """
    attacked_imgs = []
    for attack in attacks:
        attack = attack.copy()
        attack_name = attack.pop('attack')
        if attack_name in attacks_dict:
            attacked_img = attacks_dict[attack_name](img_tensor, **attack)
            attacked_imgs.append(attacked_img)
        else:
            print(f"Warning: Unknown attack: {attack_name}")
    return attacked_imgs

def compute_tf(messages: torch.Tensor, 
              decoded_messages: torch.Tensor) -> Tuple[float, float]:
    """
    Compute true-false metrics for decoded messages.
    
    Args:
        messages: Original messages
        decoded_messages: Decoded messages after attack
        
    Returns:
        Tuple of (accuracy, low confidence score)
    """
    batch_size = len(messages)
    acc = 0
    low = 0

    for i in range(batch_size):
        sample_messages = messages[i].cpu().detach().numpy()
        sample_decoded_messages = decoded_messages[i].cpu().detach().numpy()

        fpr, tpr, thresholds = metrics.roc_curve(
            sample_messages, sample_decoded_messages, pos_label=1)
        acc += np.max(1 - (fpr + (1 - tpr)) / 2)
        low += tpr[np.where(fpr < .01)[0][-1]]

    return acc / batch_size, low / batch_size

def evaluate_attacks(network: Network, 
                    dataloader: DataLoader, 
                    args: argparse.Namespace) -> List[Dict]:
    """
    Evaluate different attacks on the watermarked images.
    
    Args:
        network: The watermarking network
        dataloader: Test data loader
        args: Command line arguments
        
    Returns:
        List of evaluation results
    """
    logs = []
    device = network.device
    
    for i, images in enumerate(tqdm(dataloader, desc="Evaluating attacks")):
        image = images.to(device)
        message = torch.Tensor(
            np.random.choice([0, 1], (image.shape[0], args.message_length))
        ).to(device)

        network.vqmodel.eval_mode()

        with torch.no_grad():
            images, messages = images.to(device), message.to(device)
            encoded_images = network.vqmodel.encoder(images, messages)
            encoded_images = images + (encoded_images - image) * args.strength_factor

            noised_images = generate_attacks(encoded_images, DEFAULT_ATTACKS, ATTACKS_DICT)

            for jj, attack in enumerate(DEFAULT_ATTACKS):
                attack = attack.copy()
                attack_name = attack.pop('attack')
                param_names = [f'param{kk}' for kk in range(len(attack.keys()))]
                attack_params = dict(zip(param_names, list(attack.values())))
                
                decoded_messages = network.vqmodel.decoder(noised_images[jj])

                psnr = kornia.losses.psnr_loss(
                    noised_images[jj].detach(), images, 2).item()
                ssim = 1 - kornia.losses.ssim_loss(
                    noised_images[jj].detach(), images, 
                    window_size=5, reduction="mean"
                ).item()

                error_rate = network.decoded_message_error_rate_batch(
                    messages, decoded_messages)
                acc, low = compute_tf(messages, decoded_messages)
                
                log = {
                    "keyword": "evaluation",
                    "img": i,
                    "attack": attack_name,
                    **attack_params,
                    "err": error_rate,
                    "bit_acc": acc,
                    "psnr": psnr,
                    "ssim": ssim,
                    "low": low,
                }
                logs.append(log)
                
    return logs

def save_results(logs: List[Dict], args: argparse.Namespace) -> None:
    """
    Save evaluation results to CSV file.
    
    Args:
        logs: List of evaluation results
        args: Command line arguments
    """
    df = pd.DataFrame(logs)
    df['param0'] = df['param0'].fillna(-1)
    df_new = df[["img", "attack", "param0", "bit_acc", "psnr", "ssim", "err", "low"]]
    df_group = df_new.groupby(['attack', 'param0'], as_index=False).mean().drop(columns='img')
    
    csv_name = f"DFP{args.H}m{args.message_length}distoration.csv"
    csv_save_path = os.path.join("distoration_result", csv_name)
    os.makedirs("distoration_result", exist_ok=True)
    df_group.to_csv(csv_save_path, index=False)
    print(f"Results saved to {csv_save_path}")

def main(args: argparse.Namespace) -> None:
    """Main function to run the evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize network
    network = Network(
        H=args.H, 
        W=args.W, 
        message_length=args.message_length,
        device=device,
        batch_size=args.batch_size, 
        lr=args.lr,
        encoder_weight=args.encoder_weight,
        encoder_percepweight=args.encoder_percepweight,
        discriminator_weight=args.discriminator_weight,
        decoder_weight=args.decoder_weight,
        ckpt_path=args.ckpt_path,
        vqvae=args.vqvae
    )

    # Load model
    EC_path = os.path.join(args.result_folder, "models", f"EC_epoch_{args.model_epoch}.pth")
    network.load_model_ed(EC_path)

    # Prepare dataset
    val_dataset = MyDataset(args.dataset_path, args.H, args.W)
    test_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True
    )

    # Run evaluation
    logs = evaluate_attacks(network, test_dataloader, args)
    
    # Save results
    save_results(logs, args)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

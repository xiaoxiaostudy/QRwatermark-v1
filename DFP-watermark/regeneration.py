import os
import torch
import kornia
import pandas as pd
import numpy as np
from PIL import Image
import argparse
from typing import List, Dict, Any
from sklearn import metrics
from torch.utils.data import DataLoader
from torchvision import transforms

from network.Network import Network
from wmattacker import VAEWMAttacker, KLVAEWMAttacker, VQVAEWMAttacker, DiffWMAttacker
from util.MyDataloader import MyDataset
#before import ReSDPipeline,download https://github.com/XuandongZhao/WatermarkAttacker and put it in the same folder as this script
from WatermarkAttacker.src.diffusers import ReSDPipeline

def get_parser() -> argparse.ArgumentParser:
    """Configure and return the argument parser."""
    parser = argparse.ArgumentParser(description='Watermark regeneration testing script')
    group = parser.add_argument_group('Experiments parameters')
    
    group.add_argument("--result_folder", type=str, default="results/DFPW256m64/")
    group.add_argument("--model_epoch", type=int, default=59)
    group.add_argument("--strength_factor", type=float, default=1)
    group.add_argument("--dataset_path", type=str, default="datasets/test")
    group.add_argument("--batch_size", type=int, default=1)
    group.add_argument("--H", type=int, default=256)
    group.add_argument("--W", type=int, default=256)
    group.add_argument("--message_length", type=int, default=64)
    group.add_argument("--save_images_number", type=int, default=1)
    group.add_argument("--encoder_weight", type=float, default=10)
    group.add_argument("--encoder_percepweight", type=float, default=0.1)
    group.add_argument("--discriminator_weight", type=float, default=0.001)
    group.add_argument("--decoder_weight", type=float, default=10)
    group.add_argument("--lr", type=float, default=4e-5)
    
    return parser

def setup_pipeline(device: torch.device) -> ReSDPipeline:
    """Initialize and return the Stable Diffusion pipeline."""
    pipe = ReSDPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2",
        revision="fp16",
        torch_dtype=torch.float16
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)
    print('Finished loading diffusion model')
    return pipe

def compute_tf(messages: torch.Tensor, decoded_messages: torch.Tensor) -> tuple[float, float]:
    """Compute true-false metrics for decoded messages."""
    batch_size = len(messages)
    acc = low = 0.0

    for i in range(batch_size):
        sample_messages = messages[i].cpu().detach().numpy()
        sample_decoded_messages = decoded_messages[i].cpu().detach().numpy()
        
        fpr, tpr, _ = metrics.roc_curve(sample_messages, sample_decoded_messages, pos_label=1)
        
        acc += np.max(1 - (fpr + (1 - tpr))/2)
        low += tpr[np.where(fpr < 0.01)[0][-1]]

    return acc / batch_size, low / batch_size

def get_attacker(regen_type: str, param: int, device: torch.device, captions: List[str] = None) -> Any:
    """Initialize and return the appropriate attacker based on regeneration type."""
    if regen_type == "regen_vae":
        return VAEWMAttacker(model_name="bmshj2018-factorized", quality=param, metric="mse", device=device)
    elif regen_type == "regen_klvae":
        return KLVAEWMAttacker(f=param, device=device)
    elif regen_type == "regen_vqvae":
        return VQVAEWMAttacker(f=param, device=device)
    elif regen_type == "regen_diffusion":
        pipe = setup_pipeline(device)
        return DiffWMAttacker(pipe, noise_step=param, captions=captions)
    else:
        raise ValueError(f"Unknown regeneration attack type: {regen_type}")

def process_single_image(image: torch.Tensor, attacker: Any, device: torch.device) -> torch.Tensor:
    """Process a single image through the attacker."""
    temp_input = 'temp/input.jpg'
    temp_output = 'temp/output.jpg'
    
    os.makedirs('temp', exist_ok=True)
    
    # Save input image
    transforms.ToPILImage()(image.cpu()).save(temp_input)
    
    # Apply attack
    attacker.attack([temp_input], [temp_output])
    
    # Load result
    attacked_image = transforms.ToTensor()(Image.open(temp_output)).unsqueeze(0).to(device)
    
    # Cleanup
    os.remove(temp_input)
    os.remove(temp_output)
    
    return attacked_image

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize network
    network = Network(
        H=args.H, W=args.W, message_length=args.message_length,
        device=device, batch_size=args.batch_size, lr=args.lr,
        encoder_weight=args.encoder_weight,
        encoder_percepweight=args.encoder_percepweight,
        discriminator_weight=args.discriminator_weight,
        decoder_weight=args.decoder_weight,
        ckpt_path=args.ckpt_path,
        vqvae=args.vqvae
    )
    
    # Load model
    EC_path = os.path.join(args.result_folder, "models", f"EC_{args.model_epoch}.pth")
    network.load_model_ed(EC_path)
    
    # Setup data
    val_dataset = MyDataset(args.dataset_path, args.H, args.W)
    test_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                               shuffle=False, num_workers=8, pin_memory=True)
    
    # Define regeneration attacks
    regenattacks = [
        {'regen_type': 'regen_vae', 'param': p} for p in [4, 5, 6, 7, 8]
    ] + [
        {'regen_type': 'regen_diffusion', 'param': p} for p in [10, 20, 30, 40]
    ] + [
        {'regen_type': 'regen_klvae', 'param': p} for p in [4, 8, 16]
    ] + [
        {'regen_type': 'regen_vqvae', 'param': p} for p in [4, 8, 16, 32]
    ]
    
    logs = []
    
    # Run attacks
    for attack in regenattacks:
        attacker = get_attacker(attack['regen_type'], attack['param'], device)
        print(f"\nTesting {attack['regen_type']} with param {attack['param']}")
        
        test_results = {
            "acc": [], "psnr": [], "ssim": [],"TPR@0.1FPR": []
        }
        
        for images in test_dataloader:
            images = images.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (images.shape[0], args.message_length))).to(device)
            
            network.vqmodel.eval_mode()
            encoded_images = network.vqmodel.encoder(images, message)
            encoded_images = images + (encoded_images - images) * args.strength_factor
            
            # Process each image in batch
            regen_images = torch.cat([
                process_single_image(img, attacker, device)
                for img in encoded_images
            ], dim=0)
            
            decoded_messages = network.vqmodel.decoder(regen_images)
            
            # Compute metrics
            psnr = kornia.losses.psnr_loss(regen_images.detach(), images, 2).item()
            ssim = 1 - kornia.losses.ssim_loss(regen_images.detach(), images, window_size=5, reduction="mean").item()
            acc, low = compute_tf(message, decoded_messages)
            
            # Store results
            for key, value in zip(test_results.keys(), [acc,psnr, ssim, low]):
                test_results[key].append(value)
        
        # Compute and store mean results
        mean_results = {key: np.mean(value) for key, value in test_results.items()}
        mean_results.update({"regen_type": attack['regen_type'], "param": attack['param']})
        logs.append(mean_results)
    
    # Save results
    df = pd.DataFrame(logs)
    csv_name = f"DFP{args.H}m{args.message_length}regeneration.csv"
    os.makedirs("regeneration_result", exist_ok=True)
    csv_save_path = os.path.join("regeneration_result", csv_name)
    df.to_csv(csv_save_path, index=False)
    print(f"Results saved to {csv_name}")

if __name__ == "__main__":
    main()
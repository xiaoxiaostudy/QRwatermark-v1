import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torchvision.transforms import functional as F
import kornia
from torch.utils.data import DataLoader
from utils import *
from network.Network import *
from noise import *
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore", message="The default value of the antialias parameter of all the resizing transforms")

def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Experiments parameters')
    aa("--result_folder", type=str, default="results/vqvae-4.21_m64__2024_04_21__02_30_25/", help="Result folder")
    aa("--model_epoch", type=int, default=241, help="Model epoch to load")
    aa("--strength_factor", type=float, default=1, help="Strength factor for encoding")
    aa("--dataset_path", type=str, default="datasets/", help="Path to the dataset")
    aa("--batch_size",type=int,default=4)
    aa("--H", type=int, default=128, help="Height of the images")
    aa("--W", type=int, default=128, help="Width of the images")
    aa("--message_length", type=int, default=64, help="Length of the messages")
    aa("--save_images_number", type=int, default=1, help="Number of images to save")
    aa("--encoder_weight", type=float, default=10, help="Encoder weight")
    aa("--encoder_percepweight", type=float, default=0.01, help="Encoder perceptual weight")
    aa("--discriminator_weight", type=float, default=0.001, help="Discriminator weight")
    aa("--decoder_weight", type=float, default=10, help="Decoder weight")
    aa("--lr",type=float,default=1e-3)


    return parser

def generate_attacks(img_tensor, attacks, attacks_dict):
    """ Generate a list of attacked images from a tensor image. """
    attacked_imgs = []
    for attack in attacks:
        attack = attack.copy()
        attack_name = attack.pop('attack')
        if attack_name in attacks_dict:
            attacked_img = attacks_dict[attack_name](img_tensor, **attack)
            attacked_imgs.append(attacked_img)
        else:
            print(f"Unknown attack: {attack_name}")
    return attacked_imgs

def compute_tf(messages, decoded_messages):
    batch_size = len(messages)
    acc = 0
    low = 0

    for i in range(batch_size):
        sample_messages = messages[i].cpu().detach().numpy()
        sample_decoded_messages = decoded_messages[i].cpu().detach().numpy()

        fpr, tpr, thresholds = metrics.roc_curve(sample_messages, sample_decoded_messages, pos_label=1)
        acc += np.max(1 - (fpr + (1 - tpr)) / 2)
        low += tpr[np.where(fpr < .01)[0][-1]]

    acc /= batch_size
    low /= batch_size
    return acc, low

def main(args):
    # 设备设置
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

    # 加载网络
    network = Network(args.H, args.W, args.message_length, device, args.batch_size, args.lr,
                      args.encoder_weight, args.encoder_percepweight, args.discriminator_weight, args.decoder_weight)
    EC_path = os.path.join(args.result_folder, "models", f"EC_{args.model_epoch}.pth")
    network.load_model_ed(EC_path)

    # 加载数据集
    test_dataset = MBRSDataset(os.path.join(args.dataset_path, "111"), args.H, args.W)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 定义攻击字典和列表
    attacks_dict = {
        "none": Identity(),
        "rotation": Rotation(),
        "crop": Crop(ratio=0.5),
        "resizedcrop": Resizedcrop(scale=0.5),
        "erasing": Erasing(scale=0.5),
        "brightness": Brightness(min_brightness=0.8, max_brightness=1.2),
        "contrast": Contrast(contrast=0.5),
        "saturation": Saturation(rnd_sat=1.0),
        "blurring": Blurring(N_blur=7),
        "gnoise": Gnoise(rnd_noise=0.02),
        "jpeg": JPEG(quality=75),
    }

    attacks = [{'attack': 'none'}] \
        + [{'attack': 'rotation', 'angle': jj} for jj in range(5, 45, 5)] \
        + [{'attack': 'crop', 'ratio': jj} for jj in np.linspace(0.4, 0.9, 10)] \
        + [{'attack': 'resizedcrop', 'scale': 0.1 * jj} for jj in range(1, 10)] \
        + [{'attack': 'erasing', 'scale': 0.1 * jj} for jj in range(1, 10)] \
        + [{'attack': 'brightness', 'min_brightness': 0.5 * jj, 'max_brightness': 1.5 * jj} for jj in range(1, 5)] \
        + [{'attack': 'contrast', 'contrast': jj} for jj in [1, 0.75, 0.5, 0.25, 0.15]] \
        + [{'attack': 'saturation', 'rnd_sat': jj} for jj in [1, 0.75, 0.5, 0.25, 0.15]] \
        + [{'attack': 'blurring', 'N_blur': 1 + 2 * jj} for jj in range(1, 10)] \
        + [{'attack': 'gnoise', 'rnd_noise': 0.02 * jj} for jj in range(1, 10)] \
        + [{'attack': 'jpeg', 'quality': 10 * jj} for jj in range(1, 11)]



    logs = []

    # 测试
    for i, images in enumerate(test_dataloader):
        image = images.to(device)
        message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], args.message_length))).to(device)

        network.vqmodel.eval()

        with torch.no_grad():
            images, messages = images.to(network.device), message.to(network.device)
            encoded_images = network.vqmodel.encoder(images, messages)
            encoded_images = images + (encoded_images - image) * args.strength_factor

            noised_images = generate_attacks(encoded_images, attacks, attacks_dict)

            for jj in range(len(attacks)):
                attack = attacks[jj].copy()
                attack_name = attack.pop('attack')
                param_names = ['param%i' % kk for kk in range(len(attack.keys()))]
                attack_params = dict(zip(param_names, list(attack.values())))
                decoded_messages = network.vqmodel.decoder(noised_images[jj])

                psnr = kornia.losses.psnr_loss(noised_images[jj].detach(), images, 2).item()
                ssim = 1 - kornia.losses.ssim_loss(noised_images[jj].detach(), images, window_size=5, reduction="mean").item()

                error_rate = network.decoded_message_error_rate_batch(messages, decoded_messages)
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

    # 保存结果到数据框
    df = pd.DataFrame(logs)
    df['param0'] = df['param0'].fillna(-1)
    df_new = df[["img", "attack", "param0", "bit_acc", "psnr", "ssim", "err", "low"]]
    df_group = df_new.groupby(['attack', 'param0'], as_index=False).mean().drop(columns='img')
    df_group.to_csv("test_results111.csv", index=False)
    print("ok!")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

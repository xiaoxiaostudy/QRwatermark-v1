# Enhancing Digital Watermark Robustness through Discrete Feature Purification

# Usage
First, clone the repository locally and move inside the folder:

```sh
git clone https://github.com/xiaoxiaostudy/DFP-watermark.git
cd DFP-watermark
```

Then, install the dependencies:

```sh
conda create -n watermark python=3.8
conda activate watermark
pip install -r requirements.txt
```

# Dataset Preparation
Both the 10k training set and the 1k test set images are random choosed from the coco dataset. Make sure to prepare your dataset in the following structure:
```sh
DFP/
├── datasets/
│   ├── test
        ├── .png
        └── ...
│   └── train
└── ...
```

#Train the model
Download VQVAE-f4 (https://ommer-lab.com/files/latent-diffusion/vq-f4.zip) and put it in ckpt/ ,and then run:
```sh
python train.py --image_H 256 --image_W 256 --message_length 64 --batch_size 4
```

#Evalation the model
For distoration attack, just run:
```sh
python distoration.py --image_H 256 --image_W 256 --message_length 64 --batch_size 8
```



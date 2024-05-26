# QRwatermark-v1
Currently this is a simple version. The model is complete, but it lacks more experiments. This part will be improved soon.The training process is faithfully recorded in [Results](URL "[title](https://github.com/xiaoxiaoprincess/QRwatermark-v1/tree/main/results/QRwatermark)")
# Usage
First, clone the repository locally and move inside the folder:

```sh
git clone https://github.com/xiaoxiaoprincess/QRwatermark-v1.git
cd QRwatermark-v1
```

Then, install the dependencies:

```sh
pip install -r requirements.txt
```

This codebase has been developed with Python version 3.8, PyTorch version 1.10.2, CUDA 10.2, and torchvision 0.11.3. The following considers QRwatermark-v1/ as the root folder, all paths are relative to it.

# Running the model
Now I just provide the code to test the passive attack effect of the model. I will update here once I have completed all the experiments and confirmed the final version of the paper, I promise it will be before June 30th.
```sh
python distoration.py
```
# Dataset Preparation
Both the training set and the test set images are from the coco dataset.Make sure to prepare your dataset in the following structure:
```sh
QRwatermark-v1/
├── datasets/
│   ├── test
        ├── testimage1.png
        └── ...
│   └── train
└── ...
```

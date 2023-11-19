from dataset import UltrasonicDataset
from engine import trainer
from utils import plot_results, load_model
from models.get_model import get_model

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torchvision
torchvision.disable_beta_transforms_warning()

from monai.losses import DiceCELoss

import matplotlib.pyplot as plt
import numpy as np
import random

import yaml
import json
import time
import os


config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)

LEARNING_RATE = float(config["LEARNING_RATE"])
BATCH_SIZE = int(config["BATCH_SIZE"])
NUM_EPOCHS = int(config["NUM_EPOCHS"])

LOSS = config["LOSS"]
DATASET = config["DATASET"]

IMAGE_SIZE = int(config["IMAGE_SIZE"])
THRESHOLD = float(config["THRESHOLD"])
MODEL = config["MODEL"]
PRETRAINED = config["PRETRAINED"]
PRETRAINED_RUN = config["PRETRAINED_RUN"]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")


def START_seed():
    seed = 9
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 


def main():
    START_seed()

    #run id is date and time of the run
    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")

    #create folder for this run in runs folder
    os.mkdir("./runs/" + run_id)

    save_dir = "./runs/" + run_id
    

    #load data
    transforms_train = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=(-30, 30)),
        v2.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-10, 10, -10, 10)),
        v2.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0), antialias=True),
    ])

    transforms_test = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    ])

    train_path = "datasets/" + DATASET + "/train"
    test_path = "datasets/" + DATASET + "/test"
    train_dataset = UltrasonicDataset(root=train_path, transforms=transforms_train)
    test_dataset = UltrasonicDataset(root=test_path, transforms=transforms_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    #load model
    model = get_model(MODEL, IMAGE_SIZE)

    if PRETRAINED:
        weights_path = "./runs/" + PRETRAINED_RUN + "/best_model.pth"
        model = load_model(model, weights_path)

    model.to(DEVICE)
    
    #load optimizer
    if LOSS == "DiceCELoss":
        loss = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)
    elif LOSS == "BCEWithLogitsLoss":
        loss = torch.nn.BCEWithLogitsLoss()
    else:
        raise Exception("Loss not implemented")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)


    #train model
    results = trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        loss_fn=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=DEVICE,
        epochs=NUM_EPOCHS,
        save_dir=save_dir,
    )

    train_summary = {
        "config": config,
        "results": results,
    }

    with open(save_dir + "/train_summary.json", "w") as f:
        json.dump(train_summary, f, indent=4)

    plot_results(results["train_loss"], results["val_loss"], "Loss", save_dir)
    plot_results(results["train_dice"], results["val_dice"], "Dice", save_dir)


if __name__ == "__main__":
    main()



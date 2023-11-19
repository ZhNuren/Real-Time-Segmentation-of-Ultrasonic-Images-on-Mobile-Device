import torch
from monai.metrics import DiceMetric
from monai.transforms import Compose, AsDiscrete
from monai.data import decollate_batch

from tqdm import tqdm

from utils import save_model

def train_step(
        model: torch.nn.Module,
        train_loader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
):
    """
    Train model for one epoch.

    Args:
        model: PyTorch model to train.
        train_loader: PyTorch dataloader for training data.
        loss_fn: PyTorch loss function.
        optimizer: PyTorch optimizer.
        device: PyTorch device to use for training.

    Returns:
        Average loss for the epoch.
    """

    model.train()
    train_loss = 0.0
    train_dice = 0.0

    post_label = Compose([AsDiscrete(to_onehot=2)])
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    dice_acc = DiceMetric(include_background=True, get_not_nans=True)

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        if model._get_name() == "LRASPP":
            output = output["out"]

        if model._get_name() == "MobileViTV2ForSemanticSegmentation" or\
           model._get_name() == "SegformerForSemanticSegmentation":
            output = output[0]
            output = torch.nn.functional.interpolate(output, size=(target.shape[2], target.shape[3]), mode='bilinear', align_corners=False)

        if loss_fn._get_name() == "BCEWithLogitsLoss":

            loss = loss_fn(output[:, 1, :, :].unsqueeze(1), target)
        else:
            loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # threshold output and target
        val_labels_list = decollate_batch(target)
        val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]

        val_outputs_list = decollate_batch(output)
        val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

        dice_acc(y_pred=val_output_convert, y=val_labels_convert)

        dice = dice_acc.aggregate()[0]
        
        train_dice += dice.item()

    train_loss /= len(train_loader)
    train_dice /= len(train_loader)

    return train_loss, train_dice


def val_step(
        model: torch.nn.Module,
        val_loader,
        loss_fn: torch.nn.Module,
        device: torch.device,
):
    """
    Evaluate model on val data.

    Args:
        model: PyTorch model to evaluate.
        val_loader: PyTorch dataloader for val data.
        loss_fn: PyTorch loss function.
        device: PyTorch device to use for evaluation.

    Returns:
        Average loss and accuracy for the val set.
    """

    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    
    post_label = Compose([AsDiscrete(to_onehot=2)])
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    dice_acc = DiceMetric(include_background=True, get_not_nans=True)

    with torch.no_grad():
        for data, target in tqdm(val_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            if model._get_name() == "LRASPP":
                output = output["out"]

            if model._get_name() == "MobileViTV2ForSemanticSegmentation" or\
               model._get_name() == "SegformerForSemanticSegmentation":
                output = output[0]
                output = torch.nn.functional.interpolate(output, size=(target.shape[2], target.shape[3]), mode='bilinear', align_corners=False)


            if loss_fn._get_name() == "BCEWithLogitsLoss":
                loss = loss_fn(output[:, 1, :, :].unsqueeze(1), target)
            else:
                loss = loss_fn(output, target)

            val_loss += loss.item()

            # threshold output and target
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]

            val_outputs_list = decollate_batch(output)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

            dice_acc(y_pred=val_output_convert, y=val_labels_convert)

            dice = dice_acc.aggregate()[0]

            val_dice += dice.item()

    val_loss /= len(val_loader)
    val_dice /= len(val_loader)

    return val_loss, val_dice


def trainer(
        model: torch.nn.Module,
        train_loader,
        val_loader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        device: torch.device,
        epochs: int,
        save_dir: str,
):
    """
    Train and evaluate model.

    Args:
        model: PyTorch model to train.
        train_loader: PyTorch dataloader for training data.
        val_loader: PyTorch dataloader for val data.
        loss_fn: PyTorch loss function.
        optimizer: PyTorch optimizer.
        lr_scheduler: PyTorch learning rate scheduler.
        device: PyTorch device to use for training.
        epochs: Number of epochs to train the model for.

    Returns:
        Average loss and accuracy for the val set.
    """

    results = {
        "train_loss": [],
        "train_dice": [],
        "val_loss": [],
        "val_dice": [],
        "learning_rate": [],
    }
    best_val_loss = 1e10

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}:")
        train_loss, train_dice = train_step(model, train_loader, loss_fn, optimizer,  device)
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.5f}")
        results["learning_rate"].append(optimizer.param_groups[0]["lr"])
        lr_scheduler.step()

        results["train_loss"].append(train_loss)
        results["train_dice"].append(train_dice)

        val_loss, val_dice = val_step(model, val_loader, loss_fn, device)
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        print()
        
        results["val_loss"].append(val_loss)
        results["val_dice"].append(val_dice)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, save_dir + "/best_model.pth")

        save_model(model, save_dir + "/last_model.pth")


    return results
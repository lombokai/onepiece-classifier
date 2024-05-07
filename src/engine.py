import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_step(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    optim: torch.optim.Optimizer
):
    # put model in train mode
    model.train()

    # var to track loss and acc
    train_loss, train_acc = 0, 0

    for batch_id, (X, y) in enumerate(loader):

        X, y = X.to(device), y.to(device)

        # forward pass
        logits = model(X)

        # calculate loss
        loss = loss_fn(logits, y)
        train_loss += loss.item()

        optim.zero_grad()
        loss.backward()
        optim.step()

        # track accuracy
        y_class = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        train_acc += (y_class == y).sum().item()/len(y_class)

    # average loss and acc
    train_loss = train_loss/len(loader)
    train_acc = train_acc/len(loader)

    return train_loss, train_acc  


def val_step(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
):

    # var to track loss and acc
    val_loss, val_acc = 0, 0

    with torch.no_grad():
        
        # put model in eval mode
        model.eval()
        
        for batch_id, (X, y) in enumerate(loader):

            X, y = X.to(device), y.to(device)

            # forward pass
            logits = model(X)

            # calculate loss
            loss = loss_fn(logits, y)
            val_loss += loss.item()

            # calculate acc
            y_class = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            # y_class = logits.argmax(dim=1)
            val_acc += (y_class == y).sum().item()/len(y_class)
            

    # average loss
    val_loss = val_loss/len(loader)
    val_acc = val_acc/len(loader)

    return val_loss, val_acc


def train(
    model: nn.Module,
    loss_fn: nn.Module,
    optim: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int   
):
    # track results
    results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    prev_train_loss = np.Inf
    prev_val_loss = np.Inf

    for e in tqdm(range(epochs)):

        train_loss, train_acc = train_step(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optim=optim,
            device=device
        )
        val_loss, val_acc = val_step(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device
        )

        # Print out what's happening
        print(
            f"Epoch: {e+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        # save model only decrease 
        if train_loss < prev_train_loss and val_loss < prev_val_loss:

            # save model
            torch.save(model.state_dict(), "src/checkpoint/checkpoint.pth")

            # update prev loss
            prev_train_loss = train_loss
            prev_val_loss = val_loss

            print("[INFO]: Model Saved")

    return results

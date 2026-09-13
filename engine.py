"""
Train and test loops for a PyTorch classification model.
"""
from typing import Dict, List, Tuple

import torch
from tqdm.auto import tqdm


def train_step(model, dataloader, loss_fn, optimizer, device) -> Tuple[float, float]:
    """Runs one epoch of training. Returns (train_loss, train_acc)."""
    model.train()
    train_loss, train_acc = 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    return train_loss / len(dataloader), train_acc / len(dataloader)


def test_step(model, dataloader, loss_fn, device) -> Tuple[float, float]:
    """Runs one epoch of evaluation. Returns (test_loss, test_acc)."""
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            test_pred = model(X)
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()

            test_pred_labels = test_pred.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    return test_loss / len(dataloader), test_acc / len(dataloader)


def train(model, train_dataloader, test_dataloader, optimizer, loss_fn,
          epochs: int, device) -> Dict[str, List[float]]:
    """Trains + tests a model for a number of epochs, printing progress
    and returning a results dict of loss/acc per epoch."""
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

        print(
            f"Epoch: {epoch + 1} | train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results

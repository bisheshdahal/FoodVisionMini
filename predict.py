"""
Makes a prediction on a single custom image with a trained model and
plots the image + prediction.
"""
from typing import List

import matplotlib.pyplot as plt
import torch
import torchvision


def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform=None,
    device: torch.device = "cpu",
) -> None:
    # load image as float32 tensor, scaled to [0, 1]
    img = torchvision.io.read_image(str(image_path)).type(torch.float32) / 255.0

    if transform:
        img = transform(img)

    model.to(device)
    model.eval()
    with torch.inference_mode():
        img = img.unsqueeze(dim=0)  # add batch dimension
        pred_logits = model(img.to(device))

    pred_probs = torch.softmax(pred_logits, dim=1)
    pred_label = torch.argmax(pred_probs, dim=1)

    plt.imshow(img.squeeze().permute(1, 2, 0))
    label = class_names[pred_label] if class_names else pred_label
    plt.title(f"Pred: {label} | Prob: {pred_probs.max():.3f}")
    plt.axis(False)

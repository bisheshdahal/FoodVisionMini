# FoodVision Mini 🍕🥩🍣

A tiny CNN that looks at a photo and tells you if it's pizza, steak, or sushi. Built with PyTorch, trained on a small slice of Food101, and mostly built as a way to actually understand the image classification pipeline instead of just copy-pasting a tutorial.

Nothing fancy here — this isn't meant to beat any benchmarks. It's a from-scratch TinyVGG model, small enough to train on a laptop in a couple of minutes, and simple enough that you can actually trace through what every line is doing.

## Why this exists

Most "learn PyTorch" tutorials have you build a model once and never touch it again. This project goes a step further — the same code is organized into reusable `.py` scripts (not just notebook cells), so it mirrors how an actual small project would be structured: data loading, model, training loop, and prediction all live in their own files and get imported into the notebook.

## What's in here

```
foodvision_mini/
├── foodvision_mini.ipynb   # main notebook — start here
├── data_setup.py           # downloads the dataset + builds DataLoaders
├── model_builder.py        # TinyVGG, the CNN itself
├── engine.py                # train_step / test_step / the full train loop
├── predict.py                # run the model on one custom image + plot it
└── utils.py                  # folder walker + loss/accuracy curve plots
```

## The model

`TinyVGG` — based on the architecture from [CNN Explainer](https://poloclub.github.io/cnn-explainer/). It's about as simple as a CNN gets:

- Conv block 1: conv → ReLU → conv → ReLU → maxpool
- Conv block 2: same thing again
- Flatten → single linear layer → 3 class scores

That's it. No batch norm, no dropout, no residual connections. It's small on purpose — the point was learning how the pieces fit together, not squeezing out accuracy.

## The data

A small subset of [Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/): just 3 classes (pizza, steak, sushi), roughly 75 training images and 25 test images per class. Small enough that the whole thing trains fast, which matters when you're iterating and re-running cells constantly.

The dataset downloads and unzips itself automatically the first time you run the notebook — no manual setup needed.

## Running it

```bash
pip install torch torchvision matplotlib requests tqdm
jupyter notebook foodvision_mini.ipynb
```

Run the cells top to bottom. It'll pull the data, build the DataLoaders, train for a few epochs, and let you throw a custom image at the trained model to see what it predicts.

## A gotcha worth knowing about

If you're testing the model on your own image (not from the dataset) using `predict.py`, watch the `transform` you pass in. `torchvision.io.read_image()` already returns a tensor, so if your transform pipeline includes `transforms.ToTensor()`, it'll throw a `TypeError` — that transform expects a PIL Image or numpy array, not a tensor that's already been converted. Either drop `ToTensor()` from the pipeline, or load the image with PIL instead of `read_image()`. `transforms.Resize()` on its own works fine either way since it doesn't care about the input type.

## What I actually got out of this

Mostly just a clearer picture of the full pipeline end to end — how raw images on disk turn into batched tensors, how a training loop actually updates weights epoch by epoch, and where things quietly break (looking at you, `ToTensor()`) when a function further down the chain assumes a different input type than what it's actually getting.

## Credit

Built while working through the custom datasets section of [Daniel Bourke's PyTorch for Deep Learning course](https://www.learnpytorch.io/) — the TinyVGG architecture and the overall project structure are adapted from there.

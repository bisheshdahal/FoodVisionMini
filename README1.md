# FoodVision Mini 🍕🥩🍣

A small CNN (TinyVGG) built with PyTorch to classify images as **pizza**,
**steak**, or **sushi**. Built while learning PyTorch's custom-datasets
workflow — mainly to understand the full pipeline end to end:
data -> model -> train -> predict.

## What's in here

```
foodvision_mini/
├── foodvision_mini.ipynb     # main notebook - run this
└── going_modular/
    ├── data_setup.py         # download data + build DataLoaders
    ├── model_builder.py      # TinyVGG - the CNN itself
    ├── engine.py              # train_step / test_step / train loop
    ├── predict.py             # predict on a single custom image
    └── utils.py                # directory walker + loss curve plotting
```

## The model

`TinyVGG`: 2 conv blocks (conv -> relu -> conv -> relu -> maxpool),
then a linear layer. Deliberately small so it trains fast and is easy
to reason about.

## Data

Subset of [Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/):
3 classes, ~75 training / 25 test images per class. Downloaded
automatically the first time the notebook runs.

## Running it

```
pip install torch torchvision matplotlib requests tqdm
jupyter notebook foodvision_mini.ipynb
```

## Credit

Based on the custom datasets section of [Daniel Bourke's PyTorch for Deep
Learning course](https://www.learnpytorch.io/).

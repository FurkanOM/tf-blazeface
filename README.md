# BlazeFace

This is **unofficial** tensorflow blazeface implementation from scratch.
This repo includes the entire training pipeline of blazeface.
However, since the dataset used in the training process is a modified version of some datasets, it is not shared at this stage.
Anchor / prior box hyperparameters were taken from the [MediaPipe](https://github.com/google/mediapipe) implementation.
Loss calculation and augmentation methods were implemented as in [SSD](https://github.com/FurkanOM/tf-ssd).

It's implemented and tested with **tensorflow 2.0, 2.1, and 2.2**

## Usage

Project models created in virtual environment using [miniconda](https://docs.conda.io/en/latest/miniconda.html).
You can also create required virtual environment with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

To create virtual environment (tensorflow-2 gpu environment):

```sh
conda env create -f environment.yml
```

To train and test BlazeFace model:

```sh
python trainer.py
python predictor.py
```

If you have GPU issues you can use **-handle-gpu** flag with these commands:

```sh
python trainer.py -handle-gpu
```

### References

* BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs [[paper]](https://arxiv.org/abs/1907.05047)
* SSD: Single Shot MultiBox Detector [[paper]](https://arxiv.org/abs/1512.02325)
* MediaPipe [[code]](https://github.com/google/mediapipe)
* BlazeFace-PyTorch [[code]](https://github.com/hollance/BlazeFace-PyTorch)

# SRT: Scene Representation Transformer

This is an independent PyTorch implementation of OSRT, as presented in the paper
["Object Scene Representation Transformer"](https://osrt-paper.github.io/) by Sajjadi et al.
All credit for the model goes to the original authors.


## Setup
After cloning the repository and creating a new conda environment, the following steps will get you started:

### Data
The code currently supports the following datasets. Simply download and place (or symlink) them in the data directory.

- The 3D datasets introduced by [ObSuRF](https://stelzner.github.io/obsurf/).
- SRT's [MultiShapeNet (MSN)](https://srt-paper.github.io/#dataset) dataset, specifically version 2.3.3. It may be downloaded via gsutil:
 ```
 pip install gsutil
 mkdir -p data/msn/multi_shapenet_frames/
 gsutil -m cp -r gs://kubric-public/tfds/multi_shapenet_frames/2.3.3/ data/msn/multi_shapenet_frames/
 ```

### Dependencies
This code requires at least Python 3.9 and [PyTorch 1.11](https://pytorch.org/get-started/locally/). Additional dependencies may be installed via `pip -r requirements.txt`.
Note that Tensorflow is required to load SRT's MultiShapeNet data, though the CPU version suffices.

Rendering videos additionally depends on `ffmpeg>=4.3` being available in your `$PATH`.

## Running Experiments
Each run's config, checkpoints, and visualization are stored in a dedicated directory. Recommended configs can be found under `runs/[dataset]/[model]`.

### Training
To train a model on a single GPU, simply run e.g.:
```
python train.py runs/clevr3d/osrt/config.yaml
```
To train on multiple GPUs on a single machine, launch multiple processes via [Torchrun](https://pytorch.org/docs/stable/elastic/run.html), where $NUM_GPUS is the number of GPUs to use:
```
torchrun --standalone --nnodes 1 --nproc_per_node $NUM_GPUS train.py runs/clevr3d/osrt/config.yaml
```
Checkpoints are automatically stored in and (if available) loaded from the run directory. Visualizations and evaluations are produced periodically.
Check the args of `train.py` for additional options. Importantly, to log training progress, use the `--wandb` flag to enable [Weights & Biases](https://wandb.ai).

### Rendering videos
Videos may be rendered using `render.py`, e.g.
```
python render.py runs/clevr3d/osrt/config.yaml --sceneid 1 --motion rotate_and_closeup --fade
```
Rendered frames and videos are placed in the run directory. Check the args of `render.py` for various camera movements,
and `compile_video.py` for different ways of compiling videos.

## Citation

```
@article{sajjadi2022osrt,
  author = {Sajjadi, Mehdi S. M.
			and Duckworth, Daniel
			and Mahendran, Aravindh
			and van Steenkiste, Sjoerd
			and Paveti{\'c}, Filip
			and Lu{\v{c}}i{\'c}, Mario
			and Guibas, Leonidas J.
			and Greff, Klaus
			and Kipf, Thomas
			},
  title = {{Object Scene Representation Transformer}},
  journal = {arXiv preprint arXiv:2206.06922},
  year  = {2022}
}
```


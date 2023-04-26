# OSRT: Object Scene Representation Transformer

This is an independent PyTorch implementation of OSRT, as presented in the paper
["Object Scene Representation Transformer"](https://osrt-paper.github.io/) by Sajjadi et al.
All credit for the model goes to the original authors.

<img src="https://drive.google.com/uc?id=1Gsoxlab6c3wOL0Bdj6SEV8L1RsI-mhWF" alt="MSN Example" width="900"/>

## Setup
After cloning the repository and creating a new conda environment, the following steps will get you started:

### Data
The code currently supports the following datasets. Simply download and place (or symlink) them in the data directory.

- The 3D datasets introduced by [ObSuRF](https://stelzner.github.io/obsurf/).
- OSRT's [MultiShapeNet (MSN-hard)](https://osrt-paper.github.io/#dataset) dataset. It may be downloaded via gsutil:
 ```
 pip install gsutil
 mkdir -p data/osrt/multi_shapenet_frames/
 gsutil -m cp -r gs://kubric-public/tfds/kubric-frames/multi_shapenet_conditional/2.8.0/ data/osrt/multi_shapenet_frames/
 ```

### Dependencies
This code requires at least Python 3.9 and [PyTorch 1.11](https://pytorch.org/get-started/locally/).
Additional dependencies may be installed via `pip -r requirements.txt`. Note that Tensorflow is
required to load OSRT's MultiShapeNet data, though the CPU version suffices.

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
Checkpoints are automatically stored in and (if available) loaded from the run directory.
Visualizations and evaluations are produced periodically.  Check the args of `train.py` for
additional options. Importantly, to log training progress, use the `--wandb` flag to enable [Weights
& Biases](https://wandb.ai).

### Rendering videos
Videos may be rendered using `render.py`, e.g.
```
python render.py runs/msn/osrt/config.yaml --sceneid 1 --motion rotate_and_closeup --fade
```
Rendered frames and videos are placed in the run directory. Check the args of `render.py` for various camera movements,
and `compile_video.py` for different ways of compiling videos.

## Results
We have found OSRT's object segmentation performance to be strongly dependent on the batch sizes
used during training. Due to memory constraints, we were unable to match OSRT's settings on MSN-hard.
We conducted our largest and most successful run thus far on 8 A100 GPUs with 80GB VRAM each,
utilizing 2304 target rays per scene as opposed to the 8192 specified in the paper.
It reached a foreground ARI of around 0.73 and a PSNR of 22.8 after
750k iterations. For download, we provide both the
[checkpoint](https://drive.google.com/file/d/1EAxajGk0guvKtj0FLjza24pMbdV0p7br/view?usp=sharing)
and a
[sample video](https://drive.google.com/file/d/1m0H4Sk2DjldCdJ_O3k3siXehuk_dyd_M/view).

To match the memory availability of your hardware, consider adjusting `data/num_points` or
`training/batch_size` in `config.yaml`. However, setting these too low can make the model prone to
getting stuck in local optima, especially early in training.

We also provide a [checkpoint for CLEVR3D](https://drive.google.com/file/d/1HCwrVPWHWErGF5K_Oud5xKCWqdP-pxkB/view), with a Fg-ARI of over 0.97.
Note that this number isn't reached on every run though, as there are some other optima the model can fall into.

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
  journal = {NeurIPS},
  year  = {2022}
}
```


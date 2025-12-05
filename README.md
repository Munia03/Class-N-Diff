
## Setup

First, download and set up the repo:


We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment. 

```bash
conda env create -f environment.yml
conda activate DiT
```


## Training model


```bash
torchrun --nnodes=1 --nproc_per_node=N train_label.py --model DiT-L/4 --data-path /path/to/ISIC/train
```


## Generating synthetic data 

```bash
python generation.py --model DiT-L/4 --image-size 256 --ckpt /path/to/model.pt
```

## BibTeX

```bibtex
@article{munia2025class,
  title={Class-N-Diff: Classification-Induced Diffusion Model Can Make Fair Skin Cancer Diagnosis},
  author={Munia, Nusrat and Imran, Abdullah},
  journal={arXiv preprint arXiv:2510.16887},
  year={2025}
}
```







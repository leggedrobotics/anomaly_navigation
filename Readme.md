# Anomaly Navigation - ANNA

## Overview 

This is code to train and evaluate anomaly detection algorithms using multi-modal sensor information.

Based in part on [Deep-SVDD-PyTorch](https://github.com/lukasruff/Deep-SVDD-PyTorch).

**Author:** Lorenz Wellhausen, [lorenwel@ethz.ch](mailto:lorenwel@ethz.ch)

**Affiliation:** [Robotic Systems Lab](https://rsl.ethz.ch/), ETH Zürich

## Publications

If you use this work in an academic context, please cite the following publication:

> L. Wellhausen, R. Ranftl and M. Hutter,
> **"Safe Robot Navigation via Multi-Modal Anomaly Detection"**,
> in IEEE Robotics and Automation Letters (RA-L), 2020

## Installation

Tested on Ubuntu 18.04 using Python 3 and Pytorch 1.2.0/1.3.1.

Install dependencies:

`sudo apt install git-lfs virtualenv unzip`

Create and activate virtual environment with Python 3 as default version:

```
virtualenv --system-site-packages -p python3 ~/venv
source ~/venv/bin/activate
```

[Install Pytorch using Pip](https://pytorch.org/get-started/locally/) (Follow instructions for Ubuntu and your CUDA version).

Install additional Python dependencies:

`pip install click numpy matplotlib opencv-python sklearn tensorboard`

## Usage

All of the commands below assume that you have your terminal open in the base directory of this repo.

### ANNA Dataset

We use our anomaly navigation (ANNA) dataset to evaluate the performance of different methods and sensor configurations.
You can automatically download and extract the ANNA dataset into the appropriate folders by calling the appropriate script.

`./get_dataset.sh`

You can also download the ANNA dataset yourself from the [ETH Research Collection](https://www.research-collection.ethz.ch/handle/20.500.11850/389950)

### Single Network

To train a network using Real-NVP, with RGB, gravity-aligned depth and surface normal information (the highest performing method evaluated in the publication), simply call the provided script:

`./train_anomaly_detection.sh`

### All Combinations

To train all possible combinations (reproducing Table I from the publication), call

`python train_all_combinations.py`

This will take multiple days to train if you want to average over 10 runs, as we did for the publication. 
You can also adjust the number of iterations in the code, if you want to train more quickly.

We also provide our script to generate the pretty table we use in our publication.

`python anomaly_detection/utils/generate_overview_table.py`

### Incremental Data

To train with incrementally more data from different environmental conditions (reproducing Fig. 5 from the publication) , call 

`python train_incrementally.py`

### Track Progress

Network training logs progress via Tensorboard so that you can track AUC and loss performance during training.
To launch tensorboard run

`tensorboard --logdir log`

Then, navigate to `http://127.0.0.1:6006/` in your browser.

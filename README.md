# Implementation of A Theoretical Framework for Target Propagation
(https://arxiv.org/abs/2006.14331). Largely based on official repository: (https://github.com/meulemansalex/theoretical_framework_for_target_propagation)

## Install Python packages
All the needed Python libraries can be installed with conda by running:
```
$ conda env create -f environment.yml
```

## Running the methods
To train a model with target_prop call use main.py from terminal:

Fully connected(MLP) layers only:
```
    py main.py --num_hidden=3 --size_hidden=[256,128,128] --epochs=100 --network_type=DMLPDTP2 --save_loss_plot
```
Conv and fully connected layers:
```
    py main.py --epochs=100 --network_type=DDTPConv --save_loss_plot --initialization=xavier_normal
```

You will also need to add a --dataset flag:
    for training on standard mnist:
        add: --dataset=mnist
    for PVR:
        add: --dataset=PVR

If training without GPU:
    add: --NO_CUDA=True
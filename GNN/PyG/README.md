## PyG
This part of code is running under the PyG.
PyTorch Geometric (PyG) is a geometric deep learning extension library for PyTorch.
It consists of various methods for deep learning on graphs and other irregular structures.

## Github
https://github.com/rusty1s/pytorch_geometric

## Installation
$ pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html

$ pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html

$ pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html

$ pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html

$ pip install torch-geometric

where ${CUDA} should be replaced by either cpu, cu92, cu101 or cu102 depending on your PyTorch installation.

## Run the code

- gcn.py
    - Cora Dataset
    - try different parameters such as hidden units
    - try different optimizers
    
- drop.py
    - PubMed Dataset
    - try DropEdge method with different number of layers

$ python gcn.py
$ python gcn.py --use_gdc

$ python drop.py
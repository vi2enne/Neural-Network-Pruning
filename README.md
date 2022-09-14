## Simultaneously Learning Architectures and Features of Deep Neural Networks

This repository contains a reimplementation for paper "Simultaneously Learning Architectures and Features of Deep Neural Networks, ICANN 2019". 

## Usage


##### Clone this repo
```bash
git clone https://github.com/vi2enne/Neural-Network-Pruning
cd Neural-Network-Pruning
 ```

#### Python requirements 

Currently, the code supports Python 3
* numpy 
* PyTorch (>=1.1.0)
* pandas 
* thop
* sklearn

#### Download

Download audio spectrograms data from ISMIR2018 Tutorial from https://github.com/slychief/ismir2018_tutorial

```bash
python train.py -sr --filter-percent 0.5
```

If you use this code, please cite:

```
@inproceedings{wang2019simultaneously,
  title={Simultaneously learning architectures and features of deep neural networks},
  author={Wang, Tinghuai and Fan, Lixin and Wang, Huiling},
  booktitle={International Conference on Artificial Neural Networks},
  pages={275--287},
  year={2019},
  organization={Springer}
}
```

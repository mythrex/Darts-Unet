# Differential Architecture Search with 4 Normal Cells

We implemented a Neural Architecture Search technique inspired By [DARTS](https://arxiv.org/abs/1806.09055) for Semantic Segmentation of Images.
Author:
BabyGotHack

---

The algorithm is based on continuous relaxation and gradient descent in the architecture space. It is able to efficiently design high-performance convolutional architectures for Semantic Segmentation (on ISRO's UrbanDataset and Orchards3(junagarh)). Only a single GPU is required.

## Requirements

```
python >=3.5.5
Pytorch == 0.3.1
torchvision == 0.2.0
```

## Important files and their Usage

- operations.py : Contains all the inter cell operations.
- genotypes.py : Contains all intra cell operations.
- train_search.py : It searchs for genotype(architecture of each cell).
- Model_search.py : it gets imported in train_search.py to find alphas.
- utils.py : It contains all utility methods like mIoU, Accuracy, etc.
- train.py : When a optimal architecture is found out by network, user runs Train.py to train network on that architecture.
- model.py : when alphas are cell architecture are obtained by train_search.py, model.py imports genotype searched and constructs model based on that genotype on every epoch.
- Inference.ipynb : For inferencing the best mIoU model obtained by the network.

## Execution

To run the application

```
./run
```

## Datasets

As of now we have train and tested on UrbanDataset of ISRO.

## Citation

If you use any part of this code in your research, please cite or just remember the Name [BabyGotHack](https://www.github.com/orgs/babygothacked/people):

```
@Members{
  Shivam Kaushik(rs00188)
  Ankush Malik(rs00190)
  Rishab Sharma(rs00189)
  Harshit Singhal(rs00235)
}
```

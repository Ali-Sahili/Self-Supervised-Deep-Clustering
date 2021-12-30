# Deep Clustering with Convolutional Autoencoders
Pytorch implementation of Deep Convolutional Embedded Clustering (DCEC) method proposed by [Guo et al.](https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf).


<p float="center">
  <img src="network.png" width="540"/>
</p>


## Introduction
Deep Convolutional Embedded Clustering (DCEC) extends DEC algorithm by replacing Stacked Auto-encoder (SAE) with convolutional Auto-encoder (CAE). In addition, to avoid features space distortion, they add the reconstruction loss to the objective and optimize it simultaneously along with the clustering loss. To preserve the local structure of data, they propose to keep the decoder untouched by directly attaching the clustering loss to embedded layer.


## Requirements
- [numpy](http://www.numpy.org/)
- [torch](https://pytorch.org/)
- [torchvision](https://pypi.org/project/torchvision/)
- [scikit-learn](https://pypi.org/project/scikit-learn/)


## Project Structure

```
.
├─ network/
│  ├─ layers.py             <- Clustering layer
│  └─ model.py              <- build Deep Convolutional Embedded Clustering (DCEC) model 
│
├─ utils.py                 <- Utility functions
├─ train.py                 <- training function
├─ main.py                  <- main file
├─ architecture.png          
└─ README.md
```


## Usage
As mentioned in the paper, to proof the concept, we used a simple common dataset: "MNIST" dataset.
To train and evaluate DCEC model on this dataset, set up your configuration (parameters) and run the script as follows:

```
python3 main.py [--data_path DATA_PATH]
                [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS]
                [--num_epochs NUM_EPOCHS]
```


## Acknowledgments
Thanks to DCEC implementation introduced [here](https://github.com/michaal94/torch_DCEC).

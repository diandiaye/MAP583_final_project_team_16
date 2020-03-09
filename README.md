# MAP583_project_16

In this project we propose to make a deep learning model to build real images from segmented masks.
We proceed with a deterministic approach. So we won't use GANs

# Data

we used images from the [Cityscapes dataset](https://www.cityscapes-dataset.com/dataset-overview/). These images are taken from roads of 50 cities in Germany and during different seasons of the year. This database consists of 5000 images, each of which has a mask. Each mask can have up to 30 different colors (30 classes).

To train our model, we chose NN images with their segmentation. 
These images and their masks are in `.png` format. 


# Project structure

The project is structured as following:

```bash
.
├── loader
|   └── load_cityscapes_dataset.py # loading data from https://github.com/diandiaye/MAP583.git
|   └── dataloader.py # data loader
├── models
|   └── architecture
|   └── unet.py # classical CNN
├── toolbox
|   └── loss.py # diversity loss : percetrual losses VGG-19 
|   └── utils.py   # various utility functions
├── trainer.py # pipelines for training, validation and testing
```


# Architecture

![alt text](https://raw.githubusercontent.com/diandiaye/MAP583_final_project_team_16/master/images/Architecture.png) 

# Output

On the left, we have a real image, and on the right we have what our model predicts.

![alt text](https://raw.githubusercontent.com/diandiaye/MAP583_final_project_team_16/master/images/Result%20.png) 


### Tensorboard
In order the visualize metrics and results in tensorboard you need to launch it separately: `tensorboard --logdir = logs`. You can then access tensorboard in our browser at [localhost:6006](localhost:6006)
If you have performed multiple experiments, tensorboard will aggregate them in the same dashboard.


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
|   └── load_cityscapes_dataset.py # loading and pre-processing gtsrb data
├── models
|   └── architecture
|   └── .py # classical CNN
├── toolbox
|   └── optimizer and losses selectors
|   └── logger.py  # keeping track of most results during training and storage to static .html file
|   └── metrics.py # code snippets for computing scores and main values to track
|   └── plotter.py # snippets for plotting and saving plots to disk
|   └── utils.py   # various utility functions
├── commander.py # main file from the project serving for calling all necessary functions for training and testing
├── args.py # parsing all command line arguments for experiments
├── trainer.py # pipelines for training, validation and testing
```


# Launching

# Output

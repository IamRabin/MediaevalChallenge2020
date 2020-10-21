# Meta-Learning: Finetuning of Pretrained Network with Attention

This repo contains the code submitted to Medival challenge on GI tract's poly segmentaion. The pre-trained
weights of Unet model trained on brain MRI images were fine tuned ; which is a open source available at  https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/,thanks to "mateuszbuda" .Attention mechanism was incorporated while finetuning weights transfered from Unet trained on MRI dataset.



## Docker

```
docker build -f Dockerfile -t polypseg .
```



## Data
Dataset were made publicly available on  [/multimediaeval.github.io/editions/2020/tasks/medico/](https://multimediaeval.github.io/editions/2020/tasks/medico/). It consists of  images of  polyps in GI tract and its masks. Data folders should be  arranged as in the tree structure below to load data in different mode.
![Data Folder Structure](./readme_fig/folder_tree.png)


## Model

The pretrained model has been finetuned  with the polyp  dataset.Unet model is guided with attention mechanism.The dataset consists of 1000 images which was then splitted into 80:20 ratio of train and validation set. The test data consisted of 160 images .


## Training steps

1. Download the dataset .
2. Run docker container.
3. Run `train.py` script.Root path is set to `./medico2020`. For  help run: `python3 train.py --help`.



## Test prediction

1. Download the test data
2. Run docker container.
3. Run `test.py`  which will load the provided  trained weights from `./weights/unet.pt` file  
. Test predicted images will be saved in "./predictions". For  help run: `python3 test.py --help`.




# Results

For viewing the graph,images and events; "./log" can be loaded and viewed in tensorboard.


Mean IOU on validation= 89.11 %

## Plot
<<<<<<< HEAD
![Validation DSC VS Iterations ](./readme_fig/val_dsc.png)
=======
![Validation DSC VS Iterations](./readme_fig/val_dsc.png)
>>>>>>> 1fae896bc0847e9d7f536946e36015444b29b5d3

## Sample predictions(Red) over groundtruth(Green)

![Green- GT, Red- Pred](./readme_fig/img.gif)

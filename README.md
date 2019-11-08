# A demo of object detection --SSD(pytorch)
Including visualization made by visdom and MAP evaluation 

This notebook is written to introduce the whole process of deep learning. Here I choose the pytorch version of SSD as an example to show how to train, evaluate and test a model.

This notebook requires **python version of 3.7 and pytorch version of 1.3 with cuda, this notebook cannot run without cuda**!

pytorch1.3 can get from https://pytorch.org/get-started/locally/

Before getting started, make sure you have installed necessary packages listed below: if not, type the command behind them in your terminal to install them                 

 - opencv: `pip install opencv-python`
 - visdom: `pip install visdom`
 - numpy: `pip install numpy`
 - matplotlib:`pip install matplotlib`

The dataset I use in this process is VOC2007, so you need to download it at [http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)

Unzip it to VOCdevkit directory and the path should look like this
```
├─VOCdevkit
│  ├─results
│  │  └─VOC2007
│  │      └─Main
│  └─VOC2007
│      ├─Annotations
│      ├─ImageSets
│      │  ├─Layout
│      │  ├─Main
│      │  └─Segmentation
│      ├─JPEGImages
│      ├─SegmentationClass
│      └─SegmentationObject
```
In the original directory there is a txt file named val1.txt, it's made by myself because the original val.txt in `./VOCdevkit/VOC2007/ImageSets/Main/` is so large that if we do evaluation with this txt it will cost a lot of time, so I extract some contents from val.txt and make val1.txt. Thus remember to
**move the `val1.txt` to `./VOCdevkit/VOC2007/ImageSets/Main/`** after you download VOC2007 

Also,our model need a pre-parameter of VGG16, so you need to download the vgg16-parameters to the **weights** directory from [https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth](https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth)

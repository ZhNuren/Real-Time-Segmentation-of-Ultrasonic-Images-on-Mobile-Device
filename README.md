# Segmentation of ultrasonic images on Mobile devices

AI701 Course Project

by Rikhat Akizhanov, Nuren Zhaksylyk, Aidar Myrzakhan

## Desciption

This project endeavors to innovate within the field of medical imaging by focusing on the development of a lightweight and efficient AI model that performs segmentation of images to identify health problems using just a mobile device. The core objective of this project is to combine ultrasonic technology, mobile devices, and neural networks to develop revolutionary diagnostic tools that is readily available, user-friendly, and reliable. We aim to design and validate lightweight neural network models capable of performing effective real-time instance segmentation on ultrasonic images.

## Folders

1. deep_learning - folder consist of python scripts and jupyter notebooks to train all the needed deep learning models and creating mobile version of the models to be used in mobile application
2. mobile_app - folder consist of Android application which needs to be opened using Anroid Studio


## Datasets

1. [Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/)

2. [CT2US for Kidney Segmwntation](https://www.kaggle.com/datasets/siatsyx/ct2usforkidneyseg)

3. [Open Kidney Dataset](http://rsingla.ca/kidneyUS/)

### Prepare Datasets

Download datasets from links above, unzip and put them to datasets folder. Rename the unzipped folders as "BUSI", "OpenKidney" and "CT2US" respectively. 

Use `split_BUSI.ipynb`, `split_OpenKindey.ipynb` and `split_CT2US.ipynb` to divide the dataset to train and test 

## Learning and Configuration

Before start learning specify hyperparameter, model and dataset. Here is options and types of parameters. File to be changed is `deep_learning/config.yaml`:

1. `LEARNING_RATE`: float
2. `BATCH_SIZE`: int
3. `NUM_EPOCHS` : int
4. `IMAGE_SIZE` : int (256 recommended)
5. `MODEL`: [SegResNet, MobileNetV3, ResUNEt, MobileViTV2, Segformer]
6. `PRETRAINED`: bool
7. `PRETRAINED_RUN`: string (folder name of run log)
8. `DATASET`: [BUSI, CT2US, OpenKidney]
9. `LOSS`: [DiceCELoss, BCEWithLogitsLoss]

To start learning:

`$ cd deep_learning`

`$ python train.py`


## Test and Mobile Conversion

`deep_learning/mobole.ipynb` and `deep_learning/test.ipynb` provide examples on how to test models and convert weights for mobile application. On the top of each notebook it is required to enter the name of run log which is located in `runs` folder after training is finished.


### Use Android Studio

Open the ImageSegmentation project using Android Studio. Note the app's `build.gradle` file has the following lines:

```
    def pytorch_version = "1.12.1" // Use the same version for both
    implementation "org.pytorch:pytorch_android_lite:$pytorch_version"
    implementation "org.pytorch:pytorch_android_torchvision_lite:$pytorch_version"
```

and in the MainActivity.java, the code below is used to load the model:

```
mModule1 = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "segresnet.ptl"));
```

### Run the app
Select an Android emulator or device and build and run the app. The example image and its segmented result are as follows:

![](screenshot1.png)
![](screenshot2.png)

Note that the example image used in the repo is pretty large (400x400) so the segmentation process may take about 10 seconds. You may use an image of smaller size but the segmentation result may be less accurate.


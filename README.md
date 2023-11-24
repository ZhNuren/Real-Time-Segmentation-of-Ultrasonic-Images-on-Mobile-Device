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
Make sure you put generated .ptl file under `Real-Time-Segmentation-of-Ultrasonic-Images-on-Mobile-Device/mobile_app/ImageSegmentation/app/src/main/assets`.


## Use Android Studio
### Server side
Open `serverapp` project inside `mobile_app` folder using Android Studio. Note that the server device will be the the phone that has application with API to cast scans from ultrasonic devices directly to your phone screen.
You will have to specify `desiredFrameRate` in the `MediaProjectionService.java` which is dependent on what kind of mobile device you want to do a segmentation. It should be close to FPS of the device on this task. You can check it by pressing `Test Video` button on the main app.

### Client side
Open the `ImageSegmentation` project using Android Studio. Note the app's `build.gradle` file has the following lines:

```
    def pytorch_version = "1.12.1" // Use the same version for both
    implementation "org.pytorch:pytorch_android_lite:$pytorch_version"
    implementation "org.pytorch:pytorch_android_torchvision_lite:$pytorch_version"
```

and in the `MainActivity.java`, the code below is used to load the model:

```
mModule1 = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "segresnet.ptl"));
```
Make sure that the name of your .ptl file is similar with the name of the model on this line.

## Run the app
### APK file
Download APK files from [this Google Drive folder](https://drive.google.com/drive/folders/1jClM-F2I6sUogWWJwIP8T89h1ThcM36y?usp=sharing).

Install `aragorn server` and `aragorn client` applications on corresponding devices explained above. Once installed, make sure both devices are under same network. Press `start server` from server side and allow screen recording.

Get IP address of the server device, click `Live` button and paste it as follows and press `Connect` button:

<p align="center">
<img src="https://github.com/ZhNuren/Real-Time-Segmentation-of-Ultrasonic-Images-on-Mobile-Device/assets/43644508/6f6213bf-f6c2-47e8-af2d-4d631edca60b" width="300"/> <img src="https://github.com/ZhNuren/Real-Time-Segmentation-of-Ultrasonic-Images-on-Mobile-Device/assets/43644508/9479df66-8021-4c38-a39d-324b765371bd" width="300"/>
</p>

Please note that we are using SegResNet model and `desiredFrameRate` is set to 10 from server side as we are getting FPS around that number on client side.

### Android Studio
Open corresponding projects `ImageSegmentation`, `serverapp` for client and server sides. Connect devices, build and run corresponding projects on you devices.


# Segmentation of ultrasonic images on Mobile devices

AI701 Course Project

by Rikhat Akizhanov, Nuren Zhaksylyk, Aidar Myrzakhan

## Desciption

This project endeavors to innovate within the field of medical imaging by focusing on the development of a lightweight and efficient AI model that performs segmentation of images to identify health problems using just a mobile device. The core objective of this project is to combine ultrasonic technology, mobile devices, and neural networks to develop revolutionary diagnostic tools that is readily available, user-friendly, and reliable. We aim to design and validate lightweight neural network models capable of performing effective real-time instance segmentation on ultrasonic images.

## Datasets

1. [Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/)

2. [CT2US for Kidney Segmwntation](https://www.kaggle.com/datasets/siatsyx/ct2usforkidneyseg)

### Prepare Datasets

Download datasets from Kaggle, unzip and put them to datasets folder. Rename the unzipped folders as "BUSI" and "CT2US" respectively. 

Use `split_BUSI.ipynb` and `split_CT2US.ipynb` to divide the dataset to train and test 
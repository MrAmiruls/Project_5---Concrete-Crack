# Project_5 - Concrete-Crack (Identifying Cracks on Concrete with Image Classification using Convolutional Neural Network)

### About
Crack detection is important for the inspection and evaluation during the maintenance of concrete structures. This project is to create a convolutional neural network model that can identify cracks on conrete with high accuracy.The model is trained with a dataset of 40000 images(20000 images of concrete in good condition and 20000 images of concrete with cracks. This is the dataset source being used. https://www.kaggle.com/competitions/crack-detection-image-classification-sparc/data

### IDE and Framework
The project is completed with Spyder as the main IDE. The main frameworks used in this project are Numpy, Matplotlib and Tensorflow Keras.

### Methodology
The dataset consists of 2 folders: 'negative' and 'positive'. Both contains 20000 of images. Firstly, the data was split into train-validation set with ratio of 70:30. Next the validation data split into two part to obtain some test data with ratio of 80:20.

The input layer receive coloured images with a dimension of 160x160. The full shape will be (160,160,3). Next, Transfer learning is applied for building the deep learning model. Firstly, a preprocessing layer is created that will change the pixel values of input images to a range of -1 to 1. This layer serves as the feature scaler and it is also a requirement for the transfer learning model to output the correct signals.

For feature extractor, a pretrained model of MobileNet v2 is used. The model is readily available within TensorFlow Keras package, with ImageNet pretrained parameters. It is also frozen hence will not update during model training.

A global average pooling and dense layer are used as the classifier to output softmax signals. The softmax signals are used to identify the predicted class.


![accuracy graph](https://user-images.githubusercontent.com/85603599/166097756-7f3ed58a-d810-41ac-94b7-3410b13167e5.jpg)

![loss graph](https://user-images.githubusercontent.com/85603599/166097734-de4386c9-fc81-4842-a27d-23429626f245.jpg)

### Result
![results](https://user-images.githubusercontent.com/85603599/166097826-a223e467-5dc5-475a-b260-26e5e67ddd26.jpg)

![result](https://user-images.githubusercontent.com/85603599/166097856-67914e34-3708-4d67-b6a2-ad18e0f539e0.png)



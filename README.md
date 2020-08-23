![](https://raw.githubusercontent.com/jeffluppes/birdclassification/master/examples/example2.png)
# Bird Detection and Classification with Web Scraped Data
Involving Deep Learning, Google Cloud, and a humble Raspberry Pi with an Edge TPU accelerator. 

![](https://raw.githubusercontent.com/jeffluppes/birdclassification/master/examples/example1.png)
# Introduction
I like photographing birds, but I don't know a whole lot about them, nor how to tell closely-related species apart. Being a tech-minded data scientist, I decided to train a model on images on birds to do this for me. The bird classification project grew out of this idea as well as just wanting to do something to monitor birds in my neighbourhood. It has since grown to include a raspberry pi for deployment that also runs a generic object detection model. In order to speed processing up a little the models are converted to TensorFlow Lite, and the pi has access to a Google Coral Edge TPU Accelerator. 

I went through a couple of steps in the project that are documented more exhaustively in separate blog posts:

* Data Scraping: http://jeffluppes.github.io/2020/01/21/Building-a-bird-detector-from-scratch/
* Model training: <todo>
* Object detection: <todo> 
* The hardware parts: <todo>

# Project Summary:
### Web Scraping
For training data, I scraped roughly 200.000 images from Waarneming.nl, a well-known observation registration website. Many users submit photographs of sightings, and these are made in the same country that I am located in = great for accounting for regional variation. I spaced out requests to query only once per second, and only collected photos of which the licence allowed usage. 

### (Classification) Model Training
Initially I used a pre-trained version of VGG16 that I fine-tuned for birds. Surprisingly, I was unable to get a higher performance than what I got from using plain old CNNs. I just rolled with the CNNs as a result as the models are subject to change and can be easily improved. I offloaded training to Google Cloud Platform for access to a K80 GPU. The script I supply here uses TensorFlow 2 and was tested with 2.2. 

![](https://raw.githubusercontent.com/jeffluppes/birdclassification/master/examples/example3.png)

### Object Detection with MobileNet
I use a pre-trained version of MobileNetV2 on COCO for single shot detection. Using this has some benefits (mainly prediction time and model size) but also some drawbacks: technically the model is trained for detection of far more than birds, and the performance is not spectacular. It fails to catch many smaller birds and it often predicts a bounding box that is too small. It seems to deal badly with flocks of birds, often predicting just a single one. I intend to finetune and convert my own bird detection model for this later on. 

### Converting to tflite
To improve performance on the Raspberry Pi I wanted to convert the models to a smaller, pi-friendlier size. A smaller tflite version of the MobileNetV2 model was freely available, but I converted the classification model myself. The models were converted using the Python TFlite converter package ([docs here](https://www.tensorflow.org/lite/convert/python_api#converting_a_keras_model_)). This reduced the model size from roughly 320 MB to 106 mb. Quantization can further reduce the size down to 26 MB. Performance does suffer as a result, but anecdotely not as much as I had expected. Will update this space with a performance review at some point in the future.

# The hardware section
I use a Raspberry Pi 4 B, a picamera (8 mpx) and a Coral USB accelerator. This set-up is available for about 150 USD. I've mounted the pi on an old aluminium camera tripod, where it received its own case. 


![](https://raw.githubusercontent.com/jeffluppes/birdclassification/master/examples/example4.png)
![](https://raw.githubusercontent.com/jeffluppes/birdclassification/master/examples/example5.png)

# Installation / troubleshooting notes

On a Raspberry Pi, ou can run the .py file as if you would do with a normal python script after installing all the usual suspects in terms of libraries.

In order to use the Edge TPU capabilities, issue the following commands to fetch the library:

`echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-std`

I will add info here on installing TF 2 on a pi at some point. 

# A special thanks to
A special thanks goes out to:

* Linde Koeweiden, Jobien Veninga, Johan van den Burg for subject matter expertise.
* Nick Moesker for various hints on raspberry pi stuff.
* Dylan Verheul, Laurens Hogeweg, and Wilfred Gerritsen for inviting me over to demonstrate my work.

Furthermore, the pi part of this project benefited a lot from youtube videos and code by [@EdjeElectronics](https://github.com/EdjeElectronics).



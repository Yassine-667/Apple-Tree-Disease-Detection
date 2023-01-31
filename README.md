# Plant Disease Detection API
## Introduction :

   This is an apple tree disease detection API developed using Python and Tensorflow. It uses machine learning algorithms to identify plant diseases based on the images of leaves uploaded by the user.

## Requirements :
  python 3.10 
  tensorflow 2.11
  visual studio 2022
  visual studio 2019
  tensorflow-gpu
  cuda 11.2
  cudnn 8.1
  
  
## Installation :

Clone this repository to your local machine.
Create a virtual environment and activate it.
Install the required packages using the command :

    pip install -r requirements.txt.

## Usage :

Start the API by running python main.py
The API will be running on http://localhost:3000/.

You can now make predictions by sending a POST request to http://localhost:5000/predict with an image of a plant leaf attached.

The response will be a JSON object with the prediction result.


## Model Training :

If you want to train your own model, you can use the Training-Algo.py script. You will need to provide your own dataset for training and validation.
i personnally took my dataset from Kaggle . and i just worked with a part of it ( apple tree diseases ) .

i used my own GPU to train my model , if you want to do the same you'll need an nvidia gpu as well as having cuda and cudnn installed and working proprely , the python file test-gpu.py will let you know if the tensorflow recognizes your gpu by having 1 or plus as an output .
## Conclusion :

This API can be used for quick and accurate detection of apple tree diseases. It can be integrated with mobile and web applications for easy and convenient usage.

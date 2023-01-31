# Plant Disease Detection API
## Introduction
    This is a plant disease detection API developed using Python and Tensorflow. It uses machine learning algorithms to identify plant diseases based on the images of leaves uploaded by the user.

Requirements
Python 3.x
Tensorflow 2.x
Flask
OpenCV
Installation
Clone this repository to your local machine.
Create a virtual environment and activate it.
Install the required packages using the command pip install -r requirements.txt.
Usage
Start the API by running python app.py.
The API will be running on http://localhost:5000/.
You can now make predictions by sending a POST request to http://localhost:5000/predict with an image of a plant leaf attached.
The response will be a JSON object with the prediction result.
Model Training
If you want to train your own model, you can use the train.py script. You will need to provide your own dataset for training and validation.

Conclusion
This API can be used for quick and accurate detection of plant diseases. It can be integrated with mobile and web applications for easy and convenient usage.

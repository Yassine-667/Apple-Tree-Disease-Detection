from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

model=tf.keras.models.load_model(r"C:\Users\lazra\OneDrive\Bureau\Plant Disease detection\model")

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





class_names = ["Apple Scab", "Black rot", "Apple Rust","Healthy"]



@app.get("/ping")
async def ping(): 
    return 'Hello Everyone' 

def read_file_as_image(data)->np.ndarray:
    image =np.array(Image.open(BytesIO(data)))
    return image



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
   image =read_file_as_image(await file.read())
   img_batch=np.expand_dims(image, 0)
   predictions=model.predict(img_batch)
   predicted_class=class_names[np.argmax(predictions[0])]
   confidence=np.max(predictions[0])
   return {
    'class':predicted_class,
    'confidence':float(confidence)
   }
   

if __name__=="__main__":
    uvicorn.run(app, host='localhost', port=8000)



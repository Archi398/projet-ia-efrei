from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import os
from sklearn.model_selection import train_test_split

app = FastAPI(
    title="Image Classification API",
    description="API for training and predicting image classification models",
    version="1.0.0"
)

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        label = 'chien' if 'chien' in filename else 'chat'
        img_path = os.path.join(folder, filename)
        image = Image.open(img_path)
        images.append(np.array(image.resize((128, 128))))
        labels.append(label)
    return np.array(images), np.array(labels)

@app.post("/training")
async def train_model():
    try:
        images, labels = load_images_from_folder('img')
        
        label_to_index = {'chien': 0, 'chat': 1}
        labels = np.array([label_to_index[label] for label in labels])
        
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(128, 128, 3)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
        model.save('model/model.h5')
        return {"message": "Model trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(file.file.read()))
        image = np.array(image.resize((128, 128)))[np.newaxis, ...]
        
        model = tf.keras.models.load_model('model/model.h5')
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        index_to_label = {0: 'chien', 1: 'chat'}
        predicted_label = index_to_label[predicted_class]
        
        return {"prediction": predicted_label}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model")
async def get_model_info():
    try:
        # Example: fetching models from HuggingFace
        response = requests.get("https://huggingface.co/api/models")
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    if not os.path.exists('model'):
        os.makedirs('model')
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

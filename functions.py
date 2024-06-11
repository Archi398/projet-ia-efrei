import tensorflow as tf
from sklearn.model_selection import train_test_split

def train_model(df, target):
    try:
        X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
        
        return model
    except Exception as e:
        raise ValueError(f"Erreur lors de l'entraînement du modèle: {e}")

def make_prediction(model, df):
    try:
        predictions = model.predict(df)
        return predictions
    except Exception as e:
        raise ValueError(f"Erreur lors de la prédiction: {e}")

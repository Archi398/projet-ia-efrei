from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def train_model(df, target):
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def make_prediction(model, df):
    predictions = model.predict(df)
    return predictions

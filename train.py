import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocess import create_pipeline

def train_and_log(df):
    model_pipeline, X, y = create_pipeline(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    mlflow.start_run()

    
    model_pipeline.fit(X_train, y_train)

    
    y_pred = model_pipeline.predict(X_test)

    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("accuracy", model_pipeline.score(X_test, y_test))

    
    mlflow.sklearn.log_model(model_pipeline, "model")

    print(classification_report(y_test, y_pred))

    mlflow.end_run()
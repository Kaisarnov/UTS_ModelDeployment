import os
import pickle
from preprocess import create_pipeline  
from ingest_data import load_data       
from train import train_and_log         


model_folder = 'model'

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

def save_model():
    
    df = load_data() 

    
    model_pipeline, X, y = create_pipeline(df)

    
    model_pipeline.fit(X, y)

    
    model_path = os.path.join(model_folder, 'model.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(model_pipeline, f)

    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    save_model()
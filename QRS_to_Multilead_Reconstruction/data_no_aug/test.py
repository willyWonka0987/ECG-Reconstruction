import joblib
import numpy as np

# Load the file
file_path = "ecg_train_clean.pkl"
data = joblib.load(file_path)

# Top-level info
print(f"Type of data: {type(data)}")
print(f"Data shape: {data.shape}")
print(f"Data dtype: {data.dtype}")

# If it's an array of objects (e.g., list-like items), examine the first one
if data.dtype == object:
    print("\nFirst item type:", type(data[0]))

    if isinstance(data[0], list):
        print("Length of first item:", len(data[0]))
        if isinstance(data[0][0], list):
            print("Shape of one lead:", np.array(data[0][0]).shape)
            print("First few values of first lead:\n", np.array(data[0][0]))
        else:
            print("First item content (abbreviated):", data[0])
    elif isinstance(data[0], np.ndarray):
        print("Shape of first item:", data[0].shape)
        print("First item data:\n", data[0])
    elif isinstance(data[0], dict):
        print("First item keys:", list(data[0].keys()))
        print("First item example:\n", data[0])
else:
    print("Data sample:\n", data[0])

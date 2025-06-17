import joblib
import numpy as np

# Paths to your saved QRS segments
train_path = "qrs_train_segments.pkl"
test_path = "qrs_test_segments.pkl"

# Load datasets
train_segments = joblib.load(train_path)
test_segments = joblib.load(test_path)

# Print dataset info
print("🔎 Train Segments:")
print(f"  ➤ Type: {type(train_segments)}")
print(f"  ➤ Number of segments: {len(train_segments)}")
if len(train_segments) > 0:
    print(f"  ➤ Shape of one segment: {np.array(train_segments[0]).shape}")

print("\n🔎 Test Segments:")
print(f"  ➤ Type: {type(test_segments)}")
print(f"  ➤ Number of segments: {len(test_segments)}")
if len(test_segments) > 0:
    print(f"  ➤ Shape of one segment: {np.array(test_segments[0]).shape}")


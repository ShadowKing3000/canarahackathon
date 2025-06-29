import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Load the scaler used during training
scaler = joblib.load("scaler.pkl")

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="autoencoder_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define samples
normal_sample = pd.DataFrame([{
    'typing_speed': 1.5,
    'tap_pressure': 1,
    'swipe_velocity': 360,
    'gesture_duration': 145,
    'orientation_variance': 0.018
}])

anomalous_sample = pd.DataFrame([{
    'typing_speed': 0.4,
    'tap_pressure': 0,
    'swipe_velocity': 800,
    'gesture_duration': 40,
    'orientation_variance': 0.09
}])

samples = pd.concat([normal_sample, anomalous_sample], ignore_index=True)

# Normalize using the same scaler
samples_scaled = scaler.transform(samples)

# Run inference on each sample
THRESHOLD = 0.01
for i, sample in enumerate(samples_scaled):
    # Reshape input to match model's expected shape: (1, input_dim)
    input_data = np.array([sample], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get reconstructed output
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Calculate reconstruction error (MSE)
    mse = np.mean((sample - output_data) ** 2)

    label = "Anomaly" if mse > THRESHOLD else "Normal"
    print(f"Sample {i+1}: Reconstruction Error = {mse:.5f} → {label}")

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
# --- STEP 5: Load Model and Scaler ---
autoencoder = load_model("autoencoder_model.h5", compile=False)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

scaler = joblib.load("scaler.pkl")

# --- STEP 6: Create Sample Inputs ---

# Normal behavior sample
normal_sample = pd.DataFrame([{
    'typing_speed': 1.5,
    'tap_pressure': 1,
    'swipe_velocity': 360,
    'gesture_duration': 145,
    'orientation_variance': 0.018
}])

# Anomalous behavior sample
anomalous_sample = pd.DataFrame([{
    'typing_speed': 0.4,
    'tap_pressure': 0,
    'swipe_velocity': 800,
    'gesture_duration': 40,
    'orientation_variance': 0.09
}])

# Combine both
input_data = pd.concat([normal_sample, anomalous_sample], ignore_index=True)

# Normalize using same scaler
scaled_input = scaler.transform(input_data)

# --- STEP 7: Inference ---
reconstructed = autoencoder.predict(scaled_input)

# Calculate reconstruction error
reconstruction_error = np.mean((scaled_input - reconstructed) ** 2, axis=1)

# Set threshold empirically (example)
THRESHOLD = 0.01

# --- STEP 8: Classify and Print Results ---
for i, error in enumerate(reconstruction_error):
    label = "Anomaly" if error > THRESHOLD else "Normal"
    print(f"Sample {i+1}: Reconstruction Error = {error:.5f} → {label}")

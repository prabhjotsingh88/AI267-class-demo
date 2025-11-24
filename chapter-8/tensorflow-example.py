# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from typing import List, Dict

# 2. DATA LOADING & CLEANING
df = pd.read_csv('us_tornado_dataset_1950_2021.csv')

# Count and remove bad data (-9 magnitude)
print(f"Original rows: {len(df)}")
df = df[df['mag'] >= 0]
print(f"Cleaned rows: {len(df)}")

# Select Features and Target
feature_cols = ['wid', 'len', 'slat', 'slon', 'mo']
X = df[feature_cols].values
y = df['mag'].values

# 3. PREPROCESSING
# Split 80/20
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SCALE THE DATA (Crucial for Neural Networks!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Note: Unlike PyTorch, Keras handles Numpy arrays directly. 
# We don't need to manually convert them to Tensors here.

# 4. DEFINE MODEL ARCHITECTURE (Sequential API)
model = Sequential([
    # Input Layer (5 features) -> Hidden Layer (64 neurons)
    # 'relu' is the standard activation function for hidden layers
    Dense(64, activation='relu', input_shape=(5,)),
    
    # Hidden Layer -> Output Layer (6 neurons for classes 0-5)
    # 'softmax' converts the 6 numbers into probabilities that add up to 100%
    Dense(6, activation='softmax')
])

# 5. COMPILE THE MODEL
# We tell TensorFlow how to learn:
# - Optimizer: 'adam' (adjusts learning rate automatically)
# - Loss: 'sparse_categorical_crossentropy' (standard for multi-class classification with integer labels)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. TRAIN THE MODEL
print("Starting training...")
# batch_size=32 means it updates weights after every 32 rows
# verbose=1 shows a progress bar
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# 7. EVALUATION
print("\nGenerating Report...")
# Predict returns probabilities for all 6 classes
y_pred_probs = model.predict(X_test)

# We convert probabilities to the single class with highest score (argmax)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n--- Tornado Magnitude Classification Report (TensorFlow) ---\n")
print(classification_report(y_test, y_pred, labels=[0,1,2,3,4,5], zero_division=0))

# ==========================================
# 8. TESTING LOGIC (INFERENCE)
# ==========================================

classes = ('EF0 (Light)', 'EF1 (Moderate)', 'EF2 (Significant)', 
           'EF3 (Severe)', 'EF4 (Devastating)', 'EF5 (Incredible)')

def predict_tornado_damage_tf(tornadoes: List[Dict], model, scaler):
    # 1. Convert to DataFrame
    input_df = pd.DataFrame(tornadoes)
    
    # 2. Safety: Force column order
    input_df = input_df[feature_cols]
    
    # 3. Scale Data (Must use the same scaler from training!)
    scaled_data = scaler.transform(input_df.values)
    
    # 4. Predict (Returns probabilities)
    # Keras handles the Numpy array directly
    probs = model.predict(scaled_data, verbose=0)
    
    # 5. Convert probabilities to class index (0-5)
    predictions = np.argmax(probs, axis=1)
    
    # 6. Map to text labels
    return [classes[p] for p in predictions]

# --- TEST CASES ---
severe_case = {
    "wid": 2000,    "len": 25.0,
    "slat": 35.0,   "slon": -97.0,  "mo": 5
}

weak_case = {
    "wid": 30,      "len": 0.2,
    "slat": 41.0,   "slon": -87.0,  "mo": 9
}

print("\n--- Live Testing Results (TensorFlow) ---")
results = predict_tornado_damage_tf([severe_case, weak_case], model, scaler)

for i, res in enumerate(results):
    print(f"Test Case #{i+1}: Predicted {res}")

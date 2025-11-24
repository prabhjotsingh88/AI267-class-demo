# 1. IMPORT LIBRARIES
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 2. DATA LOADING & CLEANING
df = pd.read_csv('us_tornado_dataset_1950_2021.csv')

# Filter out unknown magnitudes (-9)
print(f"Original rows: {len(df)}")
df = df[df['mag'] >= 0]
print(f"Cleaned rows: {len(df)}")

# Select Features and Target
# wid=Width, len=Length, slat/slon=Lat/Lon, mo=Month
feature_cols = ['wid', 'len', 'slat', 'slon', 'mo']
X = df[feature_cols].values
y = df['mag'].values

# 3. PREPROCESSING
# Split data first
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SCALE the data (Standardize to mean=0, std=1)
# CRITICAL: We keep this 'scaler' object to use later for testing!
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# CONVERT to PyTorch Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 4. DEFINE THE MODEL ARCHITECTURE
class TornadoNet(nn.Module):
    def __init__(self):
        super(TornadoNet, self).__init__()
        self.layer1 = nn.Linear(5, 64)
        self.layer2 = nn.Linear(64, 6)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Initialize model
model = TornadoNet()

# 5. DEFINE LOSS AND OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 6. MODEL TRAINING LOOP
epochs = 1000
print("Starting training...")

for epoch in range(epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 7. EVALUATION
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, y_pred_tensor = torch.max(test_outputs, 1)
    y_pred = y_pred_tensor.numpy()

print("\n--- Tornado Magnitude Classification Report (PyTorch) ---\n")
print(classification_report(y_test, y_pred, labels=[0,1,2,3,4,5], zero_division=0))

# ==========================================
# 8. NEW: TESTING LOGIC (INFERENCE)
# ==========================================

# A. Define the Human-Readable Labels
classes = ('EF0 (Light)', 'EF1 (Moderate)', 'EF2 (Significant)', 
           'EF3 (Severe)', 'EF4 (Devastating)', 'EF5 (Incredible)')

# B. Define the Prediction Function
def predict_tornado_damage_pytorch(tornadoes: List[Dict], model, scaler):
    """
    Takes raw dictionary data, scales it to match the training data,
    converts to Tensor, and runs the PyTorch model.
    """
    # 1. Convert to DataFrame
    input_df = pd.DataFrame(tornadoes)
    
    # 2. SAFETY: Force column order to match training exactly
    input_df = input_df[feature_cols]
    
    # 3. CRITICAL: Scale the data
    # We use the existing 'scaler' from Step 3. 
    # Use .transform(), NOT .fit() (we don't want to relearn the mean)
    scaled_data = scaler.transform(input_df.values)
    
    # 4. Convert to Tensor
    input_tensor = torch.tensor(scaled_data, dtype=torch.float32)
    
    # 5. Run Model
    model.eval() # Good practice: set model to eval mode
    with torch.no_grad():
        outputs = model(input_tensor)
        # Find the winner (highest score)
        _, predicted_indices = torch.max(outputs, 1)
        
    # 6. Convert indices to text labels
    return [classes[idx.item()] for idx in predicted_indices]

# C. Define Test Cases
severe_case = {
    "wid": 2000,    "len": 25.0,
    "slat": 35.0,   "slon": -97.0,  "mo": 5
}

weak_case = {
    "wid": 30,      "len": 0.2,
    "slat": 41.0,   "slon": -87.0,  "mo": 9
}

# D. Run Prediction
print("\n--- Live Testing Results (PyTorch) ---")
results = predict_tornado_damage_pytorch([severe_case, weak_case], model, scaler)

for i, res in enumerate(results):
    print(f"Test Case #{i+1}: Predicted {res}")

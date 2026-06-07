# =============================================================================
# PROJECT 2: Data Classification Using AI
# Industrial Training Kit | Batch 2026
# Algorithm: K-Nearest Neighbors (KNN) | Dataset: Iris Benchmark
# =============================================================================

# --- IMPORTS (All dependencies declared at the top) --------------------------
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score

# =============================================================================
# STEP 1: LOAD & EXPLORE THE DATASET (INPUT)
# =============================================================================
iris = load_iris()

# Separate features (measurements) from labels (species)
X = iris.data  # Shape: (150, 4)
y = iris.target # Shape: (150,)

print("=" * 50)
print("STAGE 1: INPUT - DATASET OVERVIEW")
print("=" * 50)
print(f"Total Samples: {X.shape[0]}")
print(f"Features     : {iris.feature_names}")
print(f"Target Names : {iris.target_names}")


# =============================================================================
# STEP 2: SPLIT DATA (STRUCTURAL INTEGRITY)
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.20, 
    shuffle=True, 
    random_state=42
)

print("\n" + "=" * 50)
print("STAGE 2: PROCESS - DATA SPLITTING")
print("=" * 50)
print(f"Training Set: {X_train.shape[0]} samples")
print(f"Testing Set : {X_test.shape[0]} samples")


# =============================================================================
# STEP 3: FEATURE SCALING (THE GATEKEEPER)
# =============================================================================
scaler = StandardScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the training parameters (No Leakage!)
X_test_scaled = scaler.transform(X_test)

print("\n" + "=" * 50)
print("STAGE 2: PROCESS - FEATURE SCALING")
print("=" * 50)
print(f"Mean of scaled X_train: {np.mean(X_train_scaled):.2f} (Target: 0)")
print(f"Std  of scaled X_train: {np.std(X_train_scaled):.2f} (Target: 1)")


# =============================================================================
# STEP 4: TRAIN THE KNN MODEL (THE BRAIN)
# =============================================================================
# Initialize with K=5
model = KNeighborsClassifier(n_neighbors=5)

# Train (Memorize) the scaled data
model.fit(X_train_scaled, y_train)

print("\n" + "=" * 50)
print("STAGE 2: PROCESS - MODEL TRAINING")
print("=" * 50)
print("KNN Model (K=5) trained successfully.")


# =============================================================================
# STEP 5: MAKE PREDICTIONS (THE EXAM)
# =============================================================================
y_pred = model.predict(X_test_scaled)

print("\n" + "=" * 50)
print("STAGE 3: OUTPUT - PREDICTIONS VS ACTUAL")
print("=" * 50)
print(f"Predicted Labels: {y_pred}")
print(f"Actual Labels:    {y_test}")


# =============================================================================
# STEP 6: EVALUATE PERFORMANCE (THE GRADE)
# =============================================================================
# Confusion Matrix shows where the model got mixed up
cm = confusion_matrix(y_test, y_pred)

# F1 Score is the balance between Precision and Recall
f1 = f1_score(y_test, y_pred, average='macro')

print("\n" + "=" * 50)
print("STAGE 3: OUTPUT - FINAL EVALUATION")
print("=" * 50)
print("Confusion Matrix:")
print(cm)
print(f"\nF1 Score (Macro): {f1:.4f}")
print("Note: An F1 Score of 1.0 means perfect classification!")
print("=" * 50)
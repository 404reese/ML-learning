import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Example feature data (X_new)
X_new = np.array([[56.8, 17.5], [24.4, 24.1], [50.1, 10.9]])

# Example target labels (y), assuming 3 samples with random labels
y = np.array([0, 1, 0])  # Replace with your actual target values

# Check the shape of X_new
print("Shape of X_new:", X_new.shape)

# Initialize the KNeighborsClassifier with n_neighbors=3 (or less, depending on your data)
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the model using both features (X_new) and target labels (y)
knn.fit(X_new, y)

# Make predictions on the same data
prediction = knn.predict(X_new)

# Print the predictions
print(f"Prediction: {prediction}")

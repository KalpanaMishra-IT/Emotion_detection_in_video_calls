import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

# Sample data
X_train = np.random.rand(100, 64)  # 100 samples, 64 features each
y_train = np.random.randint(0, 10, 100)  # 100 labels, 10 classes

# Save X_train.npy
np.save('X_train.npy', X_train)

# Save y_train.npy
np.save('y_train.npy', y_train)

# Create and save label encoder
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

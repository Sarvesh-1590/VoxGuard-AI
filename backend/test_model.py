from model import ModelHandler
import numpy as np

# Initialize the model
handler = ModelHandler()

# Create a dummy input (MFCC of shape 40 x 200)
dummy_input = np.random.rand(40, 200)

# Run prediction
result = handler.predict(dummy_input)

print("Test prediction output:")
print(result)

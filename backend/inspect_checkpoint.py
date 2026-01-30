import torch
import os

model_path = os.path.join(os.path.dirname(__file__), "weights", "best_model.pth")
if not os.path.exists(model_path):
    print(f"File not found: {model_path}")
else:
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        with open("model_layers.txt", "w") as f:
            for key, value in state_dict.items():
                f.write(f"{key}: {value.shape}\n")
    except Exception as e:
        print(f"Error: {e}")

import torch

# Use correct file path
model_path = r"C:\Users\pranj\Desktop\Project 1\Final tester\models\best_lstm_model_hourly.pth"

# Load the model
model = torch.load(model_path, map_location=torch.device('cpu'))

# Print model architecture or state_dict
print(model)

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
print(torch.__version__)


# Load the data
file_path_train = "C:/Users/user/OneDrive - McMaster University/MFM/MFM 703/Project/S&P500 Data/S&P500 2005-2022 Training Set.xlsx"
file_path_test = "C:/Users/user/OneDrive - McMaster University/MFM/MFM 703/Project/S&P500 Data/S&P500 2023 Testing Set.xlsx"

train_data = pd.read_excel(file_path_train)
test_data = pd.read_excel(file_path_test)

# Convert the date colume in excel to datetime format and sorting
train_data['Exchange Date'] = pd.to_datetime(train_data['Exchange Date'])
test_data['Exchange Date'] = pd.to_datetime(test_data['Exchange Date'])

train_data = train_data.sort_values('Exchange Date').reset_index(drop=True)
test_data = test_data.sort_values('Exchange Date').reset_index(drop=True)

# Extract the Close price column as a NumPy array for both datasets.
train_close = train_data['Close'].values
test_close = test_data['Close'].values

# Define a function to split time series data into windows of size window_size.
# X: Contains sliding windows of the past prices.
# y: Contains the price to predict (target).
def create_time_windows(data, window_size, horizon=1):
    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size + horizon - 1])
    return np.array(X), np.array(y)

# Create windows of size 10 for training and testing data.
window_size = 10
X_train, y_train = create_time_windows(train_close, window_size)
X_test, y_test = create_time_windows(test_close, window_size)

# Split the training data into training and validation sets (80% Training/20% Testing).
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Convert NumPy arrays to PyTorch tensors for use in training.
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

# Combine features and targets into Tensor datasets.
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Implements the noise scheduling logic for DDPM, managing the diffusion process's noise levels at different time steps.
class NoiseScheduler:
    def __init__(self, num_steps, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_steps = num_steps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)  
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_bars)  
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_bars) 

    # Calculates x_(t-1) during reverse diffusion using predicted noise.
    def sample_prev_timestep(self, xt, noise_pred, t):
        # Get sqrt_one_minus_alpha_cum_prod and sqrt_alpha_cum_prod
        sqrt_one_minus_alpha_cum_prod_t = self.sqrt_one_minus_alpha_cum_prod[t].unsqueeze(-1).to(xt.device)
        sqrt_alpha_cum_prod_t = self.sqrt_alpha_cum_prod[t].unsqueeze(-1).to(xt.device)
        
        # Calculated x0
        x0 = ((xt - (sqrt_one_minus_alpha_cum_prod_t * noise_pred)) / sqrt_alpha_cum_prod_t)
        x0 = torch.clamp(x0, -1., 1.)
        
        # calculate average mean
        mean = xt - ((self.betas[t].unsqueeze(-1).to(xt.device) * noise_pred) / sqrt_one_minus_alpha_cum_prod_t)
        mean = mean / torch.sqrt(self.alphas[t].unsqueeze(-1).to(xt.device))
        
        if t[0] == 0:
            return mean, x0
        else:
            variance = (1 - self.alpha_bars[t - 1].unsqueeze(-1).to(xt.device)) / (
                1.0 - self.alpha_bars[t].unsqueeze(-1).to(xt.device)
            )
            variance = variance * self.betas[t].unsqueeze(-1).to(xt.device)
            sigma = variance ** 0.5
            z = torch.randn_like(xt)
            return mean + sigma * z, x0




import torch.nn as nn
# Defines the neural network to predict the noise during reverse diffusion.
class DenoisingModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DenoisingModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
    
    def forward(self, x_t, t):
        t_embedding = t.unsqueeze(-1).float()
        x_t_with_t = torch.cat((x_t, t_embedding), dim=-1)
        return self.net(x_t_with_t)
    
# Implements the DDPM model forward diffusion and reverse diffusion processes.
class DDPM:
    def __init__(self, model, noise_scheduler, num_steps):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.num_steps = num_steps

    def forward_diffusion(self, x_0, t):
        alpha_bar_t = self.noise_scheduler.alpha_bars[t].unsqueeze(-1).to(x_0.device)
        epsilon = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * epsilon
        return x_t, epsilon

    def reverse_diffusion(self, x_t, t):
        noise_pred = self.model(x_t, t)
        return self.noise_scheduler.sample_prev_timestep(x_t, noise_pred, t)

import torch.optim as optim

# Trains the DDPM model using mean squared error (MSE) loss between predicted and true noise.
def train_ddpm(ddpm, data_loader, num_epochs, device):
    optimizer = optim.Adam(ddpm.model.parameters(), lr=1e-3)
    ddpm.model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for x_0, _ in data_loader:
            x_0 = x_0.to(device)
            t = torch.randint(0, ddpm.num_steps, (x_0.shape[0],), device=device)
            x_t, epsilon = ddpm.forward_diffusion(x_0, t)
            epsilon_pred = ddpm.model(x_t, t)
            loss = nn.MSELoss()(epsilon_pred, epsilon)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(data_loader):.6f}")

# Evaluates the model's performance on the validation set.
def evaluate_ddpm(ddpm, data_loader, device):
    ddpm.model.eval()
    losses = []
    with torch.no_grad():
        for x_0, _ in data_loader:
            x_0 = x_0.to(device)
            t = torch.randint(0, ddpm.num_steps, (x_0.shape[0],), device=device)
            x_t, epsilon = ddpm.forward_diffusion(x_0, t)
            epsilon_pred = ddpm.model(x_t, t)
            loss = nn.MSELoss()(epsilon_pred, epsilon)
            losses.append(loss.item())
    return np.mean(losses)

# Set up device, model parameters, and the number of time steps.
device = "cuda" if torch.cuda.is_available() else "cpu"
input_size = window_size
hidden_size = 128
num_steps = 1

# Initialize noise scheduler, denoising model, and DDPM class.
noise_scheduler = NoiseScheduler(num_steps, device=device)
model = DenoisingModel(input_size, hidden_size).to(device)
ddpm = DDPM(model, noise_scheduler, num_steps)

# Train the model for 300 epochs and evaluate its performance on the validation set.
train_ddpm(ddpm, train_loader, num_epochs=1000, device=device)

val_loss = evaluate_ddpm(ddpm, val_loader, device)
print(f"Average Loss: {val_loss:.6f}")

# Predict future prices by iteratively removing noise from noisy data.
def predict_future(ddpm, data_loader, device):
    ddpm.model.eval()
    predictions = []
    with torch.no_grad():
        for x_0, _ in data_loader:
            x_0 = x_0.to(device)
            x_t = x_0.clone()
            for t in reversed(range(ddpm.num_steps)):
                if isinstance(t, tuple):
                    t = t[0]  # 解包 tuple，获取整数值
                
                t_tensor = torch.full((x_t.shape[0],), t, device=device, dtype=torch.long)
                x_t, _ = ddpm.reverse_diffusion(x_t, t_tensor)

            predictions.append(x_t[:, -1].cpu().numpy())  #Get the predict value
    return np.concatenate(predictions)



# Generate predictions and display the first 10 results.
predictions = predict_future(ddpm, test_loader, device)

# Print the result
print("part of predit result for the 2023 close price:")
print(predictions[:10])


import matplotlib.pyplot as plt

# Plot true vs. predicted prices for visualization.
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test)), y_test, label="True value", marker='o', linestyle='-', alpha=0.7)
plt.plot(range(len(predictions)), predictions, label="Predit Value", marker='x', linestyle='--', alpha=0.7)
plt.xlabel("Time step")
plt.ylabel("Close price")
plt.title("2023 S&P 500 close price predit")
plt.legend()
plt.grid()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calculate and display MSE and RMSE to evaluate prediction performance.
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = mse ** 0.5 # RMSE is the square root of MSE
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100  # Compute MAPE
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

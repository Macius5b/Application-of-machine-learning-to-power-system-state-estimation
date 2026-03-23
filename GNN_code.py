"""
Graph Neural Network (GNN) for Power System State Estimation
Author: Aleks Piszczek

This script trains and evaluates a GNN model to estimate voltages and phase angles
in the IEEE 39-Bus power system. It demonstrates the potential of machine learning
for monitoring and analyzing power network states.
"""

# ==============================================================================
# Step 1: Import necessary modules
# ==============================================================================
print("--- Step 1: Importing modules ---")
import os
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch_geometric.nn import MessagePassing, BatchNorm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}")
print("All required libraries loaded.")

# ==============================================================================
# Step 2: Load dataset
# ==============================================================================
print("\n--- Step 2: Loading dataset ---")
FILE_PATH = '/Users/alekspiszczek/Downloads/complete_dataset.csv'

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Data file not found: {FILE_PATH}")

print(f'Using data file: "{os.path.basename(FILE_PATH)}"')
df = pd.read_csv(FILE_PATH)
num_simulations = len(df)

# ==============================================================================
# Step 3: Define graph topology (IEEE 39-Bus)
# ==============================================================================
print("\n--- Step 3: Defining network graph topology ---")
edge_list_1_based = [
    [1, 2], [1, 39], [2, 3], [2, 25], [3, 4], [3, 18], [4, 5], [4, 14],
    [5, 6], [5, 8], [6, 7], [6, 11], [7, 8], [8, 9], [9, 39], [10, 11],
    [10, 13], [11, 12], [13, 14], [14, 15], [15, 16], [16, 17], [16, 19],
    [16, 24], [17, 18], [17, 27], [19, 20], [19, 33], [20, 21], [20, 34],
    [21, 22], [22, 23], [22, 35], [23, 24], [23, 36], [25, 26], [25, 37],
    [26, 27], [26, 28], [26, 29], [28, 29], [29, 38], [30, 38], [31, 32],
    [33, 34], [35, 36]
]

num_nodes = 39
num_unique_edges = len(edge_list_1_based)
edge_list_0_based = [[u - 1, v - 1] for u, v in edge_list_1_based]
edge_list_bidirectional = edge_list_0_based + [[v, u] for u, v in edge_list_0_based]
edge_index = torch.tensor(edge_list_bidirectional, dtype=torch.long).t().contiguous()
print(f"Graph structure ready. Nodes: {num_nodes}, Unique edges: {num_unique_edges}")

# ==============================================================================
# Step 4: Prepare and scale dataset
# ==============================================================================
print("\n--- Step 4: Preparing and scaling data ---")
node_cols = num_nodes * 4
node_data_flat = df.iloc[:, 1:(1 + node_cols)].values
edge_data_flat = df.iloc[:, (1 + node_cols):].values

node_data = node_data_flat.reshape(num_simulations, num_nodes, 4)
edge_data = edge_data_flat.reshape(num_simulations, num_unique_edges, 3)

X_nodes = node_data[:, :, :2]  # P and Q measurements at nodes
X_edges = edge_data            # P, Q, loading at edges
Y_nodes = node_data[:, :, 2:]  # True V and Theta

# Train/Validation/Test split
X_nodes_temp, X_nodes_test, X_edges_temp, X_edges_test, Y_nodes_temp, Y_nodes_test = train_test_split(
    X_nodes, X_edges, Y_nodes, test_size=0.2, random_state=42
)
X_nodes_train, X_nodes_val, X_edges_train, X_edges_val, Y_nodes_train, Y_nodes_val = train_test_split(
    X_nodes_temp, X_edges_temp, Y_nodes_temp, test_size=0.25, random_state=42
)

# Scaling
x_node_scaler = StandardScaler().fit(X_nodes_train.reshape(-1, X_nodes_train.shape[-1]))
x_edge_scaler = StandardScaler().fit(X_edges_train.reshape(-1, X_edges_train.shape[-1]))
y_node_scaler = StandardScaler().fit(Y_nodes_train.reshape(-1, Y_nodes_train.shape[-1]))

def scale_data(data, scaler):
    shape = data.shape
    return scaler.transform(data.reshape(-1, shape[-1])).reshape(shape)

X_nodes_train_scaled = scale_data(X_nodes_train, x_node_scaler)
X_nodes_val_scaled = scale_data(X_nodes_val, x_node_scaler)
X_nodes_test_scaled = scale_data(X_nodes_test, x_node_scaler)
X_edges_train_scaled = scale_data(X_edges_train, x_edge_scaler)
X_edges_val_scaled = scale_data(X_edges_val, x_edge_scaler)
X_edges_test_scaled = scale_data(X_edges_test, x_edge_scaler)
Y_nodes_train_scaled = scale_data(Y_nodes_train, y_node_scaler)
Y_nodes_val_scaled = scale_data(Y_nodes_val, y_node_scaler)
Y_nodes_test_scaled = scale_data(Y_nodes_test, y_node_scaler)

print("Data split and scaled.")
print(f"Dataset sizes - Train: {len(X_nodes_train)}, Val: {len(X_nodes_val)}, Test: {len(X_nodes_test)}")

# ==============================================================================
# Step 5: Define Dataset class
# ==============================================================================
class PowerSystemDataset(Dataset):
    """Custom Dataset for PyTorch Geometric graphs."""
    def __init__(self, node_features, edge_features, targets, edge_index):
        self.node_features = torch.tensor(node_features, dtype=torch.float32)
        self.edge_features = torch.tensor(edge_features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.edge_index = edge_index

    def __len__(self):
        return len(self.node_features)

    def __getitem__(self, idx):
        edge_attr_bidirectional = torch.cat([self.edge_features[idx], self.edge_features[idx]], dim=0)
        data = Data(
            x=self.node_features[idx],
            edge_index=self.edge_index,
            edge_attr=edge_attr_bidirectional,
            y=self.targets[idx]
        )
        return data

# Create datasets and loaders
BATCH_SIZE = 32
train_dataset = PowerSystemDataset(X_nodes_train_scaled, X_edges_train_scaled, Y_nodes_train_scaled, edge_index)
val_dataset = PowerSystemDataset(X_nodes_val_scaled, X_edges_val_scaled, Y_nodes_val_scaled, edge_index)
test_dataset = PowerSystemDataset(X_nodes_test_scaled, X_edges_test_scaled, Y_nodes_test_scaled, edge_index)

train_loader = GraphDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = GraphDataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = GraphDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"DataLoaders ready. Batch size: {BATCH_SIZE}")

# ==============================================================================
# Step 6: Define GNN architecture
# ==============================================================================
class GNNLayer(MessagePassing):
    def __init__(self, node_channels, edge_channels, output_channels):
        super().__init__(aggr='mean')
        self.mlp = nn.Sequential(
            nn.Linear(node_channels*2 + edge_channels, output_channels),
            nn.ReLU(),
            nn.Linear(output_channels, output_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.mlp(tmp)

class GNNStateEstimator(nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_output_features, hidden_channels=128):
        super().__init__()
        self.conv1 = GNNLayer(num_node_features, num_edge_features, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GNNLayer(hidden_channels, num_edge_features, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.output_layer = nn.Linear(hidden_channels, num_output_features)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x).relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x).relu()
        out = self.output_layer(x)
        return out

# Initialize model
model = GNNStateEstimator(num_node_features=2, num_edge_features=3, num_output_features=2)
print("GNN architecture defined.")
print(model)

# ==============================================================================
# Step 7: Training setup
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training and evaluation functions
def train_epoch():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def evaluate_model(loader):
    model.eval()
    all_preds_scaled, all_labels_scaled = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr)
            all_preds_scaled.append(out.cpu().numpy())
            all_labels_scaled.append(data.y.cpu().numpy())
    all_preds_scaled = np.vstack(all_preds_scaled)
    all_labels_scaled = np.vstack(all_labels_scaled)
    all_preds_unscaled = y_node_scaler.inverse_transform(all_preds_scaled.reshape(-1, 2)).reshape(all_preds_scaled.shape)
    all_labels_unscaled = y_node_scaler.inverse_transform(all_labels_scaled.reshape(-1, 2)).reshape(all_labels_scaled.shape)
    val_loss = mean_squared_error(all_labels_scaled, all_preds_scaled)
    mae = mean_absolute_error(all_labels_unscaled, all_preds_unscaled)
    rmse = np.sqrt(mean_squared_error(all_labels_unscaled, all_preds_unscaled))
    r2 = r2_score(all_labels_unscaled, all_preds_unscaled)
    return mae, rmse, r2, val_loss, all_labels_unscaled, all_preds_unscaled

# ==============================================================================
# Step 8: Training loop
# ==============================================================================
NUM_EPOCHS = 100
train_loss_history, val_loss_history, val_mae_history = [], [], []
best_val_loss = float('inf')
total_train_time = 0

print(f"Starting training for {NUM_EPOCHS} epochs...")
for epoch in range(1, NUM_EPOCHS+1):
    train_loss = train_epoch()
    val_mae, _, _, val_loss, _, _ = evaluate_model(val_loader)
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    val_mae_history.append(val_mae)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(),'best_model.pth')
    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch {epoch:03d} - Train MSE: {train_loss:.6f}, Val MSE: {val_loss:.6f}')

print("Training finished.")
model.load_state_dict(torch.load('best_model.pth'))

# ==============================================================================
# Step 9: Final evaluation on test set
# ==============================================================================
print("\n--- Final evaluation on test set ---")
start_test_time = time.time()
final_mae, final_rmse, final_r2, _, final_labels, final_preds = evaluate_model(test_loader)
end_test_time = time.time()
total_test_time = end_test_time - start_test_time

print(f"MAE: {final_mae:.6f}, RMSE: {final_rmse:.6f}, R2: {final_r2:.6f}")
print(f"Training time: {total_train_time:.2f}s, Test evaluation time: {total_test_time:.2f}s")
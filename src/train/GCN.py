from os import cpu_count

import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_max_pool
from tqdm import tqdm

from src.train.load.troch_iter_loader import SMILESDataset

torch.manual_seed(0)


class GCNModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels, improved=True)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)  # regression

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = global_max_pool(x, batch)

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)

        return x


def train_epoch(model, loader, optimizer, criterion, device, scaler, use_amp):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        with autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            out = model(batch)

            # Calculate loss
            target = batch.y.view(-1, 1)
            loss = criterion(out, target)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


if __name__ == "__main__":
    hidden_channels = 256
    num_node_features = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    dataset = SMILESDataset("data/chembl_36/graph_train")
    num_workers = min(12, cpu_count() or 1)
    loader_kwargs = {
        "batch_size": 256,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=4,
        )
    train_loader = DataLoader(dataset, **loader_kwargs)
    model = GCNModel(
        num_node_features=num_node_features,
        hidden_channels=hidden_channels,
    ).to(device)
    first_batch = next(iter(train_loader))
    num_node_features = first_batch.x.shape[1]
    print(first_batch.x[0].dtype)
    print(first_batch.y[0].dtype)
    print(first_batch.edge_index[0].dtype)
    # quit()
    print("\n--- Model Architecture ---")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    ## Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    criterion = torch.nn.MSELoss()
    use_amp = device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)

    print("\n--- Training ---")
    num_epochs = 1
    torch._dynamo.config.capture_scalar_outputs = True
    # torch.set_float32_matmul_precision("high")
    model.compile()
    for epoch in range(num_epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, use_amp
        )

        print(f"Epoch {epoch + 1:02d}: Train Loss: {train_loss:.4f}")

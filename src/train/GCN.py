from os import cpu_count

import torch
import torch.nn.functional as F
from tinygrad.helpers import trange
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_max_pool

from src.train.load.troch_iter_loader import SMILESDataset

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCNModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)

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
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = global_max_pool(x, batch)

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)

        return x


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    scaler: GradScaler,
    t,
):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast(device.type, dtype=torch.bfloat16):
            out = model(batch)

            target = batch.y.view(-1, 1)
            loss = loss_fn(out, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs
        t.set_description(f"Train Loss: {total_loss / max(1, total_graphs):.4f}")
        t.update(1)

    return total_loss / max(1, total_graphs)


def valid_epoch(
    model: torch.nn.Module, loader: DataLoader, loss_fn: torch.nn.Module, vt
):
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)

            with autocast(device.type, dtype=torch.bfloat16):
                out = model(batch)

                target = batch.y.view(-1, 1)
                loss = loss_fn(out, target)

            total_loss += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs
            vt.set_description(f"Valid Loss: {total_loss / max(1, total_graphs):.4f}")
            vt.update(1)

    return total_loss / max(1, total_graphs)


## ARGS
BS_SIZE = 256
LR = 2e-4
OPTIM = torch.optim.AdamW
LOSS = torch.nn.MSELoss()
EPOCHS = 10
HIDDEN_CHANNELS = 512
NUM_WORKERS = min(12, cpu_count() or 1)

if __name__ == "__main__":
    train_ds = SMILESDataset("data/chembl_36/graph_train")
    valid_ds = SMILESDataset("data/chembl_36/graph_valid")
    loader_kwargs = {
        "batch_size": BS_SIZE,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
    }
    if NUM_WORKERS > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=4,
        )
    train_loader = DataLoader(train_ds, **loader_kwargs)
    valid_loader = DataLoader(valid_ds, **loader_kwargs)
    first_batch = next(iter(train_loader))
    NUM_NODE_FEATURES = first_batch.x.shape[1]
    model = GCNModel(
        num_node_features=NUM_NODE_FEATURES,
        hidden_channels=HIDDEN_CHANNELS,
    ).to(device)

    ## Optimizer
    optimizer = OPTIM(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()
    scaler = GradScaler(device.type)

    print("\n--- Training ---")
    torch._dynamo.config.capture_scalar_outputs = True

    t = trange(int(EPOCHS * len(train_loader.dataset) / BS_SIZE))
    vt = trange(int(EPOCHS * len(valid_loader.dataset) / BS_SIZE))

    model.compile()
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, t=t)
        valid_loss = valid_epoch(model, valid_loader, criterion, vt=vt)
        print(
            f"\nEpoch {epoch + 1}: Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}"
        )

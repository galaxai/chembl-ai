from __future__ import annotations

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


## ARGS
BS_SIZE = 256
LR = 2e-3
OPTIM = torch.optim.AdamW
LOSS = torch.nn.MSELoss()
EPOCHS = 10
HIDDEN_CHANNELS = 512
NUM_WORKERS = min(12, cpu_count() or 1)
TRAIN_DIR = "data/chembl_36/graph_train"
VALID_DIR = "data/chembl_36/graph_valid"
GRAD_LOG_EPOCHS = 2


def log_metric(logger, name: str, value: float, step: int) -> None:
    if logger:
        logger.log_metric(name, float(value), step)


def log_gradient_stats(model: torch.nn.Module, logger, step: int) -> None:
    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad = param.grad.detach().float()
        log_metric(logger, f"grad_std/{name}", grad.std(unbiased=False).item(), step)
        log_metric(logger, f"grad_var/{name}", grad.var(unbiased=False).item(), step)
        log_metric(logger, f"grad_norm/{name}", grad.norm().item(), step)
        log_metric(
            logger,
            f"grad_zero_fraction/{name}",
            (grad == 0).float().mean().item(),
            step,
        )


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    scaler: GradScaler,
    t,
    epoch: int,
    step_offset: int,
    logger=None,
    grad_log_epochs: int = 0,
):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_graphs = 0
    global_step = step_offset
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast(device.type, dtype=torch.bfloat16):
            out = model(batch)

            target = batch.y.view(-1, 1)
            loss = loss_fn(out, target)

        scaler.scale(loss).backward()
        if logger and epoch < grad_log_epochs:
            scaler.unscale_(optimizer)
            log_gradient_stats(model, logger, global_step)

        scaler.step(optimizer)
        scaler.update()

        loss_value = loss.item()
        total_loss += loss_value * batch.num_graphs
        total_graphs += batch.num_graphs
        log_metric(logger, "train_loss", loss_value, global_step)
        t.set_description(f"Train Loss: {total_loss / max(1, total_graphs):.4f}")
        t.update(1)
        global_step += 1

    return total_loss / max(1, total_graphs), global_step


def valid_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    vt,
    step_offset: int,
    logger=None,
):
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    global_step = step_offset
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)

            with autocast(device.type, dtype=torch.bfloat16):
                out = model(batch)

                target = batch.y.view(-1, 1)
                loss = loss_fn(out, target)

            loss_value = loss.item()
            total_loss += loss_value * batch.num_graphs
            total_graphs += batch.num_graphs
            log_metric(logger, "valid_loss", loss_value, global_step)
            vt.set_description(f"Valid Loss: {total_loss / max(1, total_graphs):.4f}")
            vt.update(1)
            global_step += 1

    return total_loss / max(1, total_graphs), global_step


def train_gcn(
    epochs: int = EPOCHS,
    lr: float = LR,
    batch_size: int = BS_SIZE,
    hidden_channels: int = HIDDEN_CHANNELS,
    train_dir: str = TRAIN_DIR,
    valid_dir: str = VALID_DIR,
    logger=None,
    grad_log_epochs: int = GRAD_LOG_EPOCHS,
):
    train_ds = SMILESDataset(train_dir)
    valid_ds = SMILESDataset(valid_dir)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
        "persistent_workers": True,
        "prefetch_factor": 4,
    }
    train_loader = DataLoader(train_ds, **loader_kwargs)
    valid_loader = DataLoader(valid_ds, **loader_kwargs)
    first_batch = next(iter(train_loader))
    num_node_features = first_batch.x.shape[1]
    model = GCNModel(
        num_node_features=num_node_features,
        hidden_channels=hidden_channels,
    ).to(device)

    optimizer = OPTIM(model.parameters(), lr=lr)
    scaler = GradScaler(device.type)

    print("\n--- Training ---")
    torch._dynamo.config.capture_scalar_outputs = True

    t = trange(epochs * len(train_loader))
    vt = trange(epochs * len(valid_loader))

    model.compile()

    train_step = 0
    valid_step = 0
    for epoch in range(epochs):
        train_loss, train_step = train_epoch(
            model,
            train_loader,
            optimizer,
            LOSS,
            scaler,
            t=t,
            epoch=epoch,
            step_offset=train_step,
            logger=logger,
            grad_log_epochs=grad_log_epochs,
        )
        valid_loss, valid_step = valid_epoch(
            model,
            valid_loader,
            LOSS,
            vt=vt,
            step_offset=valid_step,
            logger=logger,
        )
        log_metric(logger, "epoch_train_loss", train_loss, epoch)
        log_metric(logger, "epoch_valid_loss", valid_loss, epoch)
        print(
            f"\nEpoch {epoch + 1}: Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}"
        )

    return model


if __name__ == "__main__":
    train_gcn()

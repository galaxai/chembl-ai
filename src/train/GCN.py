from __future__ import annotations

from math import isfinite
from os import cpu_count
from pathlib import Path

import torch
import torch.nn.functional as F
from tinygrad.helpers import trange
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import IterableDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

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

        x = F.silu(self.conv1(x, edge_index))
        x = x + F.silu(self.conv2(x, edge_index))
        x = x + F.silu(self.conv3(x, edge_index))
        x = x + F.silu(self.conv4(x, edge_index))
        x = global_mean_pool(x, batch)

        x = self.lin1(x)
        x = F.silu(x)
        x = self.lin2(x)

        return x


## ARGS
BS_SIZE = 256
LR = 2e-4
OPTIM = torch.optim.AdamW
LOSS_NAME = "huber"
HUBER_DELTA = 0.5
LOSS = torch.nn.HuberLoss(delta=HUBER_DELTA)
EPOCHS = 100
HIDDEN_CHANNELS = 512
NUM_WORKERS = min(12, cpu_count() or 1)
TRAIN_DIR = "data/chembl_36/graph_train"
VALID_DIR = "data/chembl_36/graph_valid"
GRAD_LOG_EPOCHS = EPOCHS
GRAD_CLIP_NORM = 1.0
ACTIVATION = "silu"
POOLING = "global_mean_pool"
RESIDUAL_CONNECTIONS = True
EARLY_STOP_PATIENCE = 2
EARLY_STOP_MIN_DELTA = 0.0
MODEL_SAVE_PATH = "artifacts/gcn_best.pt"


def log_metric(logger, name: str, value: float, step: int) -> None:
    if logger and isfinite(value):
        logger.log_metric(name, float(value), step)


def log_gradient_stats(model: torch.nn.Module, logger, step: int) -> None:
    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad = param.grad.detach().float()
        log_metric(logger, f"grad_std/{name}", grad.std(unbiased=False).item(), step)
        log_metric(
            logger,
            f"grad_zero_fraction/{name}",
            (grad == 0).float().mean().item(),
            step,
        )


def log_parameter_stats(model: torch.nn.Module, logger, step: int) -> None:
    for name, param in model.named_parameters():
        value = param.detach().float()
        log_metric(logger, f"param_std/{name}", value.std(unbiased=False).item(), step)
        log_metric(logger, f"param_var/{name}", value.var(unbiased=False).item(), step)
        log_metric(
            logger,
            f"param_zero_fraction/{name}",
            (value == 0).float().mean().item(),
            step,
        )


def batch_regression_sums(
    out: torch.Tensor, target: torch.Tensor
) -> tuple[float, float, float, float, int]:
    pred = out.detach().float().view(-1)
    truth = target.detach().float().view(-1)
    error = pred - truth
    return (
        error.abs().sum().item(),
        error.square().sum().item(),
        truth.sum().item(),
        truth.square().sum().item(),
        truth.numel(),
    )


def r2_from_sums(
    squared_error: float,
    target_sum: float,
    target_squared_sum: float,
    num_targets: int,
) -> float:
    if num_targets <= 1:
        return float("nan")

    total_sum_squares = target_squared_sum - (target_sum * target_sum / num_targets)
    if total_sum_squares <= 0:
        return 1.0 if squared_error <= 1e-12 else 0.0

    return 1.0 - (squared_error / total_sum_squares)


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


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
    grad_clip_norm: float | None = None,
):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_graphs = 0
    total_abs_error = 0.0
    total_squared_error = 0.0
    total_target_sum = 0.0
    total_target_squared_sum = 0.0
    total_targets = 0
    global_step = step_offset
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast(device.type, dtype=torch.bfloat16):
            out = model(batch)

            target = batch.y.view(-1, 1)
            loss = loss_fn(out, target)

        scaler.scale(loss).backward()
        should_log_grads = logger and epoch < grad_log_epochs
        should_clip_grads = grad_clip_norm is not None and grad_clip_norm > 0
        if should_log_grads or should_clip_grads:
            scaler.unscale_(optimizer)

        if should_log_grads:
            log_parameter_stats(model, logger, global_step)
            log_gradient_stats(model, logger, global_step)

        if should_clip_grads:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        loss_value = loss.item()
        (
            batch_abs_error,
            batch_squared_error,
            batch_target_sum,
            batch_target_squared_sum,
            batch_targets,
        ) = batch_regression_sums(out, target)
        batch_mae = batch_abs_error / max(1, batch.num_graphs)
        batch_rmse = (batch_squared_error / max(1, batch.num_graphs)) ** 0.5
        batch_r2 = r2_from_sums(
            batch_squared_error,
            batch_target_sum,
            batch_target_squared_sum,
            batch_targets,
        )
        total_loss += loss_value * batch.num_graphs
        total_graphs += batch.num_graphs
        total_abs_error += batch_abs_error
        total_squared_error += batch_squared_error
        total_target_sum += batch_target_sum
        total_target_squared_sum += batch_target_squared_sum
        total_targets += batch_targets
        log_metric(logger, "train_loss", loss_value, global_step)
        log_metric(logger, "train_mae", batch_mae, global_step)
        log_metric(logger, "train_rmse", batch_rmse, global_step)
        log_metric(logger, "train_r2", batch_r2, global_step)
        t.set_description(f"Train Loss: {total_loss / max(1, total_graphs):.4f}")
        t.update(1)
        global_step += 1

    mean_loss = total_loss / max(1, total_graphs)
    mae = total_abs_error / max(1, total_graphs)
    rmse = (total_squared_error / max(1, total_graphs)) ** 0.5
    r2 = r2_from_sums(
        total_squared_error,
        total_target_sum,
        total_target_squared_sum,
        total_targets,
    )
    return mean_loss, mae, rmse, r2, global_step


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
    total_abs_error = 0.0
    total_squared_error = 0.0
    total_target_sum = 0.0
    total_target_squared_sum = 0.0
    total_targets = 0
    global_step = step_offset
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)

            with autocast(device.type, dtype=torch.bfloat16):
                out = model(batch)

                target = batch.y.view(-1, 1)
                loss = loss_fn(out, target)

            loss_value = loss.item()
            (
                batch_abs_error,
                batch_squared_error,
                batch_target_sum,
                batch_target_squared_sum,
                batch_targets,
            ) = batch_regression_sums(out, target)
            batch_mae = batch_abs_error / max(1, batch.num_graphs)
            batch_rmse = (batch_squared_error / max(1, batch.num_graphs)) ** 0.5
            batch_r2 = r2_from_sums(
                batch_squared_error,
                batch_target_sum,
                batch_target_squared_sum,
                batch_targets,
            )
            total_loss += loss_value * batch.num_graphs
            total_graphs += batch.num_graphs
            total_abs_error += batch_abs_error
            total_squared_error += batch_squared_error
            total_target_sum += batch_target_sum
            total_target_squared_sum += batch_target_squared_sum
            total_targets += batch_targets
            log_metric(logger, "valid_loss", loss_value, global_step)
            log_metric(logger, "valid_mae", batch_mae, global_step)
            log_metric(logger, "valid_rmse", batch_rmse, global_step)
            log_metric(logger, "valid_r2", batch_r2, global_step)
            vt.set_description(f"Valid Loss: {total_loss / max(1, total_graphs):.4f}")
            vt.update(1)
            global_step += 1

    mean_loss = total_loss / max(1, total_graphs)
    mae = total_abs_error / max(1, total_graphs)
    rmse = (total_squared_error / max(1, total_graphs)) ** 0.5
    r2 = r2_from_sums(
        total_squared_error,
        total_target_sum,
        total_target_squared_sum,
        total_targets,
    )
    return mean_loss, mae, rmse, r2, global_step


def train_gcn(
    epochs: int = EPOCHS,
    lr: float = LR,
    batch_size: int = BS_SIZE,
    hidden_channels: int = HIDDEN_CHANNELS,
    train_dir: str = TRAIN_DIR,
    valid_dir: str = VALID_DIR,
    logger=None,
    grad_log_epochs: int = GRAD_LOG_EPOCHS,
    grad_clip_norm: float | None = GRAD_CLIP_NORM,
    early_stop_patience: int | None = EARLY_STOP_PATIENCE,
    early_stop_min_delta: float = EARLY_STOP_MIN_DELTA,
    model_save_path: str = MODEL_SAVE_PATH,
):
    train_ds = SMILESDataset(train_dir)
    valid_ds = SMILESDataset(valid_dir)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
        "persistent_workers": True,
        "prefetch_factor": 4,
    }
    train_loader_kwargs = dict(loader_kwargs)
    valid_loader_kwargs = dict(loader_kwargs)
    if not isinstance(train_ds, IterableDataset):
        train_loader_kwargs["shuffle"] = True
    if not isinstance(valid_ds, IterableDataset):
        valid_loader_kwargs["shuffle"] = False

    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    valid_loader = DataLoader(valid_ds, **valid_loader_kwargs)
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
    best_valid_loss = float("inf")
    best_epoch = -1
    best_model_state = None
    save_path = Path(model_save_path)
    epochs_without_improvement = 0
    for epoch in range(epochs):
        train_loss, train_mae, train_rmse, train_r2, train_step = train_epoch(
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
            grad_clip_norm=grad_clip_norm,
        )
        valid_loss, valid_mae, valid_rmse, valid_r2, valid_step = valid_epoch(
            model,
            valid_loader,
            LOSS,
            vt=vt,
            step_offset=valid_step,
            logger=logger,
        )
        log_metric(logger, "epoch_train_loss", train_loss, epoch)
        log_metric(logger, "epoch_train_mae", train_mae, epoch)
        log_metric(logger, "epoch_train_rmse", train_rmse, epoch)
        log_metric(logger, "epoch_train_r2", train_r2, epoch)
        log_metric(logger, "epoch_valid_loss", valid_loss, epoch)
        log_metric(logger, "epoch_valid_mae", valid_mae, epoch)
        log_metric(logger, "epoch_valid_rmse", valid_rmse, epoch)
        log_metric(logger, "epoch_valid_r2", valid_r2, epoch)

        if valid_loss < best_valid_loss - early_stop_min_delta:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_model_state = cpu_state_dict(model)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "valid_loss": valid_loss,
                    "model_state_dict": best_model_state,
                    "num_node_features": num_node_features,
                    "hidden_channels": hidden_channels,
                },
                save_path,
            )
            epochs_without_improvement = 0
        elif valid_loss > best_valid_loss + early_stop_min_delta:
            epochs_without_improvement += 1

        log_metric(logger, "best_valid_loss", best_valid_loss, epoch)
        log_metric(
            logger, "early_stop_counter", float(epochs_without_improvement), epoch
        )
        print(
            f"\nEpoch {epoch + 1}: "
            f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Train RMSE: {train_rmse:.4f}, Train R2: {train_r2:.4f}, "
            f"Valid Loss: {valid_loss:.4f}, Valid MAE: {valid_mae:.4f}, Valid RMSE: {valid_rmse:.4f}, Valid R2: {valid_r2:.4f}"
        )

        if (
            early_stop_patience is not None
            and epochs_without_improvement >= early_stop_patience
        ):
            print(
                f"Stopping early at epoch {epoch + 1}: valid loss regressed from best "
                f"{best_valid_loss:.4f} at epoch {best_epoch + 1}."
            )
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


if __name__ == "__main__":
    train_gcn()

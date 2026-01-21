from tinygrad import nn
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import trange
from tinygrad.nn import optim, state
from tinygrad.tensor import Tensor

from src.train.load import ParquetDataLoader


class TinyMLP:
    def __init__(self):
        self.l1 = nn.Linear(2048, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, 256)
        self.l4 = nn.Linear(256, 1)

    def __call__(self, x):
        x = self.l1(x).relu()
        x = self.l2(x).relu()
        x = self.l3(x).relu()
        x = self.l4(x)
        return x


EPOCHS = 5
LR = 0.01
BATCH_SIZE = 2028
TRAIN_DIR = "data/chembl_36/fp_train"
VAL_DIR = "data/chembl_36/fp_val"
X_COL = "morgan_fp"
Y_COL = "pIC"


def train_tinymlp(
    epochs: int = EPOCHS,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
    train_dir: str = TRAIN_DIR,
    val_dir: str = VAL_DIR,
    x_col: str = X_COL,
    y_col: str = Y_COL,
    logger=None,
):
    model = TinyMLP()
    dataloader = ParquetDataLoader(
        dir=train_dir, X=x_col, Y=y_col, batch_size=batch_size
    )
    val_dataloader = ParquetDataLoader(
        dir=val_dir, X=x_col, Y=y_col, batch_size=batch_size * 4
    )
    optimizer = optim.Muon(state.get_parameters(model), lr=lr)

    @TinyJit
    @Tensor.train(True)
    def train_step(x: Tensor, y: Tensor) -> Tensor:
        optimizer.zero_grad()
        preds = model(x)
        loss = (preds - y).square().mean()
        loss.backward()
        optimizer.step()
        return loss.realize()

    @TinyJit
    @Tensor.train(False)
    def valid_step(x: Tensor, y: Tensor) -> Tensor:
        preds = model(x)
        loss = (preds - y).square().mean()
        return loss.realize()

    print("Training started")

    total_steps = epochs * len(dataloader)
    t = trange(total_steps)

    def log_metric(name: str, value: float, step: int) -> None:
        if logger:
            logger.log_metric(name, value, step)

    for _ in range(epochs):
        train_losses = []
        valid_losses = []
        loader_iter = iter(dataloader)
        for step in range(len(dataloader)):
            x, y = next(loader_iter)
            loss_value = train_step(x, y).item()
            train_losses.append(loss_value)
            if step % 10 == 9:
                t.set_description(f"loss: {loss_value:6.2f}")
            log_metric("train_loss", loss_value, t.i)
            t.update(1)

        val_iter = iter(val_dataloader)
        for step in range(len(val_dataloader)):
            x, y = next(val_iter)
            item_loss = valid_step(x, y).item()
            valid_losses.append(item_loss)
            log_metric("train_loss", item_loss, t.i)


if __name__ == "__main__":
    train_tinymlp()

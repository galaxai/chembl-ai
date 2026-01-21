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


EPOCHS = 2
LR = 0.01
BATCH_SIZE = 2028
if __name__ == "__main__":
    model = TinyMLP()
    dataloader = ParquetDataLoader(
        dir="data/chembl_36/fp_train", X="morgan_fp", Y="pIC", batch_size=BATCH_SIZE
    )
    optimizer = optim.Muon(state.get_parameters(model), lr=LR)

    @TinyJit
    @Tensor.train()
    def train_step(x: Tensor, y: Tensor) -> Tensor:
        optimizer.zero_grad()
        preds = model(x)
        loss = (preds - y).square().mean()
        loss.backward()
        optimizer.step()
        return loss.realize()

    print("Training started")

    total_steps = EPOCHS * len(dataloader)
    t = trange(total_steps)
    for epoch in range(EPOCHS):
        loader_iter = iter(dataloader)
        for step in range(len(dataloader)):
            x, y = next(loader_iter)
            loss = train_step(x, y)
            if step % 10 == 9:
                t.set_description(f"loss: {loss.item():6.2f}")
            t.update(1)

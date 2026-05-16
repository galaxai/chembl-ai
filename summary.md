# CONFIG

## ARGS
BS_SIZE = 256
LR = 2e-4
OPTIM = torch.optim.AdamW
LOSS_NAME = "huber"
HUBER_DELTA = 0.5
LOSS = torch.nn.HuberLoss(delta=HUBER_DELTA)
EPOCHS = 25
HIDDEN_CHANNELS = 512
NUM_WORKERS = min(12, cpu_count() or 1)
TRAIN_DIR = "data/chembl_36/graph_train"
VALID_DIR = "data/chembl_36/graph_valid"
GRAD_LOG_EPOCHS = EPOCHS
GRAD_CLIP_NORM = 1.0
ACTIVATION = "silu"
POOLING = "global_mean_pool"
RESIDUAL_CONNECTIONS = True

## MODEL

```python
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
```

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool

from src.config import DEFAULT_MODEL_PATH


class GCNModel(torch.nn.Module):
    def __init__(self, num_node_features: int, hidden_channels: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data: Data | Batch) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.silu(self.conv1(x, edge_index))
        x = x + F.silu(self.conv2(x, edge_index))
        x = x + F.silu(self.conv3(x, edge_index))
        x = x + F.silu(self.conv4(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.silu(self.lin1(x))
        return self.lin2(x)


def smiles_to_graph(smiles: str) -> Data:
    """Convert a SMILES string to the graph features used for GCN training."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(
            [
                float(atom.GetAtomicNum()),
                float(atom.GetDegree()),
                float(atom.GetFormalCharge()),
                float(atom.GetHybridization().real),
                float(int(atom.GetIsAromatic())),
                float(atom.GetTotalNumHs()),
            ]
        )

    edge_indices: list[list[int]] = [[], []]
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices[0].extend([i, j])
        edge_indices[1].extend([j, i])

    return Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=torch.tensor(edge_indices, dtype=torch.int64).contiguous(),
    )


def load_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    device: str | torch.device | None = None,
) -> GCNModel:
    """Load a trained GCN checkpoint for inference."""
    resolved_device = torch.device(
        device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(model_path, map_location=resolved_device)
    model = GCNModel(
        num_node_features=checkpoint["num_node_features"],
        hidden_channels=checkpoint["hidden_channels"],
    ).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_pic(
    smiles: str,
    model: GCNModel | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    device: str | torch.device | None = None,
) -> float:
    """Predict pIC from a SMILES string."""
    if model is None:
        model = load_model(model_path=model_path, device=device)

    model_device = next(model.parameters()).device
    graph = smiles_to_graph(smiles)
    batch = Batch.from_data_list([graph]).to(model_device)

    with torch.no_grad():
        prediction = model(batch)

    return float(prediction.view(-1).item())

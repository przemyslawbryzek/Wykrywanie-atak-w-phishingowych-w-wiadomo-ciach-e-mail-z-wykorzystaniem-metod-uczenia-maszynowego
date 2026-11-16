import copy
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import average_precision_score
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, n_inputs, hidden, dropout):
        super().__init__()
        layers, width = [], n_inputs
        for size in hidden:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.net(inputs).squeeze(-1)


def n_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def to_tensor(rows):
    if sp.issparse(rows):
        rows = rows.tocoo()
        indices = torch.from_numpy(np.vstack([rows.row, rows.col]).astype(np.int64))
        values = torch.from_numpy(rows.data.astype(np.float32))
        return torch.sparse_coo_tensor(indices, values, rows.shape, check_invariants=False).to(DEVICE).to_dense()
    return torch.from_numpy(np.ascontiguousarray(rows, dtype=np.float32)).to(DEVICE)


@torch.no_grad()
def predict(model, features, batch_size=2048):
    model.eval()
    scores = [
        torch.sigmoid(model(to_tensor(features[start:start + batch_size]))).cpu().numpy()
        for start in range(0, features.shape[0], batch_size)
    ]
    return np.concatenate(scores)


def train_mlp(
    features, labels, hidden, dropout, learning_rate, weight_decay, seed, max_epochs,
    batch_size=256, holdout=None, patience=None,
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MLP(features.shape[1], hidden, dropout).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.BCEWithLogitsLoss()
    targets = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    history, best_score, best_epoch, best_state = [], -np.inf, max_epochs, None
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(features.shape[0])
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_function(model(to_tensor(features[batch])), targets[batch].to(DEVICE))
            loss.backward()
            optimizer.step()
        if holdout is None:
            continue
        holdout_features, holdout_labels, holdout_weights = holdout
        score = average_precision_score(holdout_labels, predict(model, holdout_features), sample_weight=holdout_weights)
        history.append(score)
        if score > best_score:
            best_score, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch

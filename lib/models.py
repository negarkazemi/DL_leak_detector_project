"""
lib/models.py -- the two MLPs (fully connected only, PyTorch) and one training loop for both.

  detector:  out_dim = 1, sigmoid, binary cross-entropy (class-weighted)
  localiser: out_dim = K, softmax, categorical cross-entropy
"""
import time

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(128, 64), out_dim=1, dropout=0.2, batchnorm=True):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h)]
            if batchnorm:
                layers += [nn.BatchNorm1d(h)]
            layers += [nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]          # logits; sigmoid / softmax applied in the loss / at predict
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def train_mlp(X, y, task="binary", sample_weight=None, hidden=(128, 64), dropout=0.2, batchnorm=True,
              lr=1e-3, weight_decay=1e-4, epochs=20, batch=512, seed=0, verbose=False,
              n_classes=None, class_weight=True):
    """
    task="binary": y in {0,1}; loss = BCE with pos_weight = n_neg / n_pos (class balance)
    task="multi":  y in {0..K-1}; loss = cross-entropy (class weights ~ 1 / frequency if class_weight)
                   n_classes fixes K (needed when some classes have no training sample)
    sample_weight: optional per-row weight (0 = ignore row in the loss).
    Returns (model, info dict with params count and training seconds).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.as_tensor(X, dtype=torch.float32)
    w = torch.ones(len(X)) if sample_weight is None else torch.as_tensor(sample_weight, dtype=torch.float32)

    if task == "binary":
        yt = torch.as_tensor(y, dtype=torch.float32)
        pos = float((yt * w).sum())
        neg = float(((1 - yt) * w).sum())
        model = MLP(X.shape[1], hidden, 1, dropout, batchnorm)
        crit = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(neg / max(pos, 1.0)))
    else:
        yt = torch.as_tensor(y, dtype=torch.long)
        K = n_classes or int(yt.max()) + 1
        cnt = torch.bincount(yt, weights=w, minlength=K).clamp(min=1.0)
        model = MLP(X.shape[1], hidden, K, dropout, batchnorm)
        cw = cnt.sum() / (K * cnt) if class_weight else None
        crit = nn.CrossEntropyLoss(reduction="none", weight=cw)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    t0 = time.time()
    n = len(X)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            if len(b) < 2:
                continue
            out = model(X[b])
            out = out.squeeze(1) if task == "binary" else out
            loss = (crit(out, yt[b]) * w[b]).sum() / w[b].sum().clamp(min=1e-6)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        if verbose:
            print(f"  epoch {ep + 1}/{epochs} loss {tot / n:.4f}")
    model.eval()
    return model, {"params": n_params(model), "train_s": time.time() - t0}


@torch.no_grad()
def predict(model, X, task="binary", batch=8192):
    model.eval()
    X = torch.as_tensor(X, dtype=torch.float32)
    outs = []
    for i in range(0, len(X), batch):
        o = model(X[i:i + batch])
        outs.append(torch.sigmoid(o.squeeze(1)) if task == "binary" else torch.softmax(o, dim=1))
    return torch.cat(outs).numpy()

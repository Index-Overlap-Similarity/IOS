# %%
from pathlib import Path


import argparse
import json
import math
import os
import time
import logging
from contextlib import contextmanager
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset

try:
    import torchvision
    from torchvision import transforms
except Exception as e:
    torchvision = None
    transforms = None

try:
    from sklearn.datasets import fetch_20newsgroups
    from transformers import BertTokenizer

    HAVE_SKLEARN = True
    HAVE_TRANSFORMERS = True
except Exception as e:
    HAVE_SKLEARN = False
    HAVE_TRANSFORMERS = False

# -------------------------
# Utils
# -------------------------


@contextmanager
def timer(
    step_name: str,
    logger: Optional[logging.Logger] = None,
    timings_dict: Optional[Dict[str, float]] = None,
):
    """Context manager for timing operations."""
    start = time.time()
    if logger:
        logger.info(f"Starting: {step_name}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        if timings_dict is not None:
            timings_dict[step_name] = elapsed
        if logger:
            logger.info(f"Completed: {step_name} - Time: {elapsed:.4f}s")
        else:
            print(f"{step_name}: {elapsed:.4f}s")


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("ios_benchmark")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int = 1337):
    import random

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_params_like(vecs: Iterable[torch.Tensor]) -> torch.Tensor:
    flats = []
    for v in vecs:
        if v is None:
            continue
        flats.append(v.reshape(-1))
    if len(flats) == 0:
        return torch.tensor([])
    return torch.cat(flats, dim=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(np.dot(a, b) / denom)


def jaccard_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # a, b are int arrays of indices (unique, sorted not required)
    sa = set(map(int, a.tolist()))
    sb = set(map(int, b.tolist()))
    if not sa and not sb:
        return 1.0
    inter = len(sa.intersection(sb))
    union = len(sa.union(sb))
    return inter / max(union, 1)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)


def oracle_distribution_similarity(p: np.ndarray, q: np.ndarray) -> float:
    # Oracle similarity based on label distributions: 1 - JS divergence
    return 1.0 - js_divergence(p, q)


def rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    # Returns average ranks (1..n) with tie handling, pure numpy (no scipy)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # handle ties
    sorted_x = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        if j - i > 1:
            avg = 0.5 * (i + 1 + j)  # average of ranks i+1..j
            ranks[order[i:j]] = avg
        i = j
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    # Spearman rho = Pearson corr of ranks
    rx = rankdata_average_ties(x)
    ry = rankdata_average_ties(y)
    # subtract mean
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = (np.linalg.norm(rx) * np.linalg.norm(ry)) + 1e-12
    return float(np.dot(rx, ry) / denom)


def topk_recall_at_k(true_rank: List[int], pred_rank: List[int], k: int) -> float:
    true_topk = set(true_rank[:k])
    pred_topk = set(pred_rank[:k])
    if len(true_topk) == 0:
        return 1.0
    return len(true_topk.intersection(pred_topk)) / len(true_topk)


def auc_binary(scores: List[float], labels: List[int]) -> float:
    """
    Compute AUROC from scores and binary labels via the Mann–Whitney U statistic.
    Returns probability that a random positive has higher score than a random negative + 0.5 ties.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # ranks of all scores
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # average ties
    s_sorted = scores[order]
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and s_sorted[j] == s_sorted[i]:
            j += 1
        if j - i > 1:
            avg = 0.5 * (i + 1 + j)
            ranks[order[i:j]] = avg
        i = j
    # sum ranks for positives
    R_pos = ranks[labels == 1].sum()
    n_pos = len(pos)
    n_neg = len(neg)
    U = R_pos - n_pos * (n_pos + 1) / 2.0
    return float(U / (n_pos * n_neg))


# -------------------------
# Models
# -------------------------


class SimpleCNN(nn.Module):
    # For FashionMNIST (1x28x28)
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# CIFAR-style ResNet blocks
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class CIFARResNet(nn.Module):
    # Depth = 6n + 2, use n=5 -> 32 layers
    def __init__(self, depth: int = 32, num_classes: int = 10):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth should be 6n+2"
        n = (depth - 2) // 6
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, n, stride=1)
        self.layer2 = self._make_layer(32, n, stride=2)
        self.layer3 = self._make_layer(64, n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def build_model(model_name: str, dataset_name: str, num_classes: int) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "cnn" and dataset_name == "fmnist":
        return SimpleCNN(num_classes=num_classes)
    elif model_name == "resnet18":
        # Use torchvision resnet18, adjust first conv for CIFAR
        try:
            from torchvision.models import resnet18
        except Exception as e:
            raise RuntimeError("torchvision is required for resnet18.")
        m = resnet18(num_classes=num_classes)
        # For CIFAR inputs (32x32), swap first conv to 3x3, stride1
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        return m
    elif model_name == "resnet50":
        # Use torchvision resnet50, adjust first conv for CIFAR
        try:
            from torchvision.models import resnet50
        except Exception as e:
            raise RuntimeError("torchvision is required for resnet50.")
        m = resnet50(num_classes=num_classes)
        # For CIFAR inputs (32x32), swap first conv to 3x3, stride1
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        return m
    elif model_name == "bert_base":
        # BERT base model - note: this requires transformers library and text data
        try:
            from transformers import BertForSequenceClassification, BertConfig
        except Exception as e:
            raise RuntimeError(
                "transformers library is required for bert_base. Install with: pip install transformers"
            )
        config = BertConfig.from_pretrained("bert-base-uncased", num_labels=num_classes)
        m = BertForSequenceClassification(config)
        return m
    elif model_name == "resnet32":
        return CIFARResNet(depth=32, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


# -------------------------
# Data loading and partitioning
# -------------------------


class TextDataset(Dataset):
    """Dataset for text data with BERT tokenization."""

    def __init__(
        self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def get_dataset(dataset_name: str, root: str = "./data", download: bool = True):
    dataset_name = dataset_name.lower()

    if dataset_name == "20newsgroups":
        if not HAVE_SKLEARN:
            raise RuntimeError(
                "sklearn is required for 20newsgroups. Install with: pip install scikit-learn"
            )
        if not HAVE_TRANSFORMERS:
            raise RuntimeError(
                "transformers is required for 20newsgroups. Install with: pip install transformers"
            )

        # Load 20 Newsgroups dataset
        newsgroups_train = fetch_20newsgroups(
            subset="train", remove=("headers", "footers", "quotes")
        )
        texts = newsgroups_train.data
        labels = newsgroups_train.target

        # Initialize BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Create text dataset
        trainset = TextDataset(texts, labels, tokenizer, max_length=512)
        num_classes = 20
        in_channels = None  # Not applicable for text data
        return trainset, num_classes, in_channels

    if torchvision is None:
        raise RuntimeError(
            "torchvision is required to load image datasets. Please install torchvision."
        )

    if dataset_name == "fmnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        trainset = torchvision.datasets.FashionMNIST(
            root=root, train=True, transform=transform, download=download
        )
        num_classes = 10
        in_channels = 1
    elif dataset_name == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
        trainset = torchvision.datasets.CIFAR10(
            root=root, train=True, transform=transform, download=download
        )
        num_classes = 10
        in_channels = 3
    elif dataset_name == "cifar100":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        trainset = torchvision.datasets.CIFAR100(
            root=root, train=True, transform=transform, download=download
        )
        num_classes = 100
        in_channels = 3
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return trainset, num_classes, in_channels


def labels_of_dataset(dataset: Dataset) -> np.ndarray:
    # Returns a numpy array of labels for indexing
    if isinstance(dataset, TextDataset):
        # TextDataset stores labels in self.labels
        labels = dataset.labels
        # Convert to numpy if it's a torch tensor
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        return np.array(labels, dtype=int)
    elif hasattr(dataset, "targets"):
        lab = dataset.targets
        # Convert to numpy if it's a torch tensor
        if isinstance(lab, torch.Tensor):
            lab = lab.cpu().numpy()
        if isinstance(lab, list):
            return np.array(lab, dtype=int)
        else:
            return np.array(lab, dtype=int)
    elif hasattr(dataset, "train_labels"):  # legacy
        lab = dataset.train_labels
        # Convert to numpy if it's a torch tensor
        if isinstance(lab, torch.Tensor):
            lab = lab.cpu().numpy()
        if isinstance(lab, list):
            return np.array(lab, dtype=int)
        else:
            return np.array(lab, dtype=int)
    else:
        # Fallback: iterate (slow)
        labels = []
        for item in dataset:
            if isinstance(item, dict):
                # TextDataset returns dict
                label_val = item["labels"]
                if isinstance(label_val, torch.Tensor):
                    label_val = label_val.cpu().item()
                labels.append(int(label_val))
            else:
                # Image dataset returns (x, y)
                _, y = item
                if isinstance(y, torch.Tensor):
                    y = y.cpu().item()
                labels.append(int(y))
        return np.array(labels, dtype=int)


def partition_iid(num_clients: int, labels: np.ndarray) -> List[np.ndarray]:
    idxs = np.arange(len(labels))
    np.random.shuffle(idxs)
    parts = np.array_split(idxs, num_clients)
    return [p.astype(int) for p in parts]


def partition_dirichlet(
    num_clients: int, labels: np.ndarray, num_classes: int, alpha: float
) -> List[np.ndarray]:
    # For each class, draw client proportions from Dir(alpha), allocate samples accordingly.
    idxs_by_class = [np.where(labels == c)[0] for c in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idxs = idxs_by_class[c]
        np.random.shuffle(idxs)
        if len(idxs) == 0:
            continue
        # Dirichlet proportions
        proportions = np.random.dirichlet(alpha=np.ones(num_clients) * alpha)
        # Translate proportions to counts
        counts = (proportions * len(idxs)).astype(int)
        # Adjust remainder
        while counts.sum() < len(idxs):
            counts[np.argmax(proportions)] += 1
        # Assign
        start = 0
        for i, cnt in enumerate(counts):
            if cnt > 0:
                client_indices[i].extend(idxs[start : start + cnt].tolist())
                start += cnt
    return [np.array(sorted(ci), dtype=int) for ci in client_indices]


def partition_pathological_classes(
    num_clients: int, labels: np.ndarray, num_classes: int, n_classes_per_client: int
) -> List[np.ndarray]:
    # Each client is assigned a random set of n_classes_per_client classes; it receives a balanced subset from those
    # classes so that all clients have roughly equal total counts.
    idxs_by_class = [np.where(labels == c)[0].tolist() for c in range(num_classes)]
    for c in range(num_classes):
        np.random.shuffle(idxs_by_class[c])

    # Assign class subsets
    client_classes = [
        np.random.choice(num_classes, size=n_classes_per_client, replace=False).tolist()
        for _ in range(num_clients)
    ]
    # Determine per-client target size (roughly equal)
    total = len(labels)
    base = total // num_clients

    client_indices = [[] for _ in range(num_clients)]
    class_ptr = {c: 0 for c in range(num_classes)}
    for i in range(num_clients):
        needed = base
        for c in client_classes[i]:
            avail = len(idxs_by_class[c]) - class_ptr[c]
            take = min(avail, max(1, needed // n_classes_per_client))
            if take > 0:
                start = class_ptr[c]
                end = start + take
                client_indices[i].extend(idxs_by_class[c][start:end])
                class_ptr[c] = end
                needed -= take
        # If still short, greedily fill from assigned classes
        if needed > 0:
            for c in client_classes[i]:
                avail = len(idxs_by_class[c]) - class_ptr[c]
                take = min(avail, needed)
                if take > 0:
                    start = class_ptr[c]
                    end = start + take
                    client_indices[i].extend(idxs_by_class[c][start:end])
                    class_ptr[c] = end
                    needed -= take
                if needed <= 0:
                    break
    # Convert
    return [np.array(sorted(ci), dtype=int) for ci in client_indices]


def build_client_loaders(
    dataset: Dataset,
    client_indices: List[np.ndarray],
    batch_size: int = 64,
    shuffle: bool = True,
):
    loaders = []
    for idxs in client_indices:
        subset = Subset(dataset, idxs.tolist())
        loaders.append(
            DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        )
    return loaders


def empirical_label_distribution(
    labels: np.ndarray, indices: np.ndarray, num_classes: int, eps: float = 1e-8
) -> np.ndarray:
    # Convert to numpy if it's a torch tensor
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().numpy()

    # Ensure both are numpy arrays
    labels = np.asarray(labels, dtype=int)
    indices = np.asarray(indices, dtype=int)

    counts = np.bincount(labels[indices], minlength=num_classes).astype(float)
    counts += eps
    return counts / counts.sum()


# -------------------------
# Importance computation
# -------------------------


def compute_importance_vector(
    model: nn.Module,
    loader: DataLoader,
    importance: str = "magnitude",  # kept for backward-compat; no longer used
    device: str = "cpu",
    max_batches: Optional[int] = None,
    *,
    epochs: int = 1,  # NEW: local training epochs (E)
    lr: float = 0.01,  # NEW: optimizer params
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    return_abs: bool = False,  # NEW: if True, return |weights| (handy for IOS top-k)
) -> np.ndarray:
    """
    Trains the model locally for `epochs` and returns a flattened weight vector.
    - Replaces the old gradient-accumulation ("backward-only") behavior.
    - `max_batches` (if set) limits batches per epoch (useful for quick tests).
    - Set `return_abs=True` if you want absolute weights for top-k/IOS.

    Returns:
        flat (np.ndarray, float32): flattened (optionally absolute) model weights.
    """
    model = model.to(device)
    model.train()

    opt = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    for _ in range(epochs):
        batches_seen = 0
        for batch in loader:
            opt.zero_grad(set_to_none=True)

            # Handle both image data (xb, yb) and text data (dict)
            if isinstance(batch, dict):
                # Text data (BERT): batch is a dict with input_ids, attention_mask, labels
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # BERT models return a tuple, first element is logits
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs[0]
                loss = loss_fn(logits, labels)
            else:
                # Image data: batch is (xb, yb)
                xb, yb = batch
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = loss_fn(logits, yb)

            loss.backward()
            opt.step()

            batches_seen += 1
            if max_batches is not None and batches_seen >= max_batches:
                break

    # After local training, return flattened weights (abs if requested)
    with torch.no_grad():
        params = [p.detach() for p in model.parameters() if p.requires_grad]
        flat = flatten_params_like(params).cpu().numpy().astype(np.float32)
        if return_abs:
            flat = np.abs(flat)
    return flat


def topk_indices_from_vector(vec: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k >= len(vec):
        return np.arange(len(vec), dtype=int)
    idx = np.argpartition(vec, -k)[-k:]
    idx_sorted = idx[np.argsort(-vec[idx])]
    return idx_sorted.astype(int)


# -------------------------
# Pretraining (optional)
# -------------------------


def pretrain_global_model(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 128,
    epochs: int = 0,
    device: str = "cpu",
    lr: float = 0.01,
):
    if epochs <= 0:
        return model
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        for batch in loader:
            opt.zero_grad(set_to_none=True)

            # Handle both image data (xb, yb) and text data (dict)
            if isinstance(batch, dict):
                # Text data (BERT)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs[0]
                loss = loss_fn(logits, labels)
            else:
                # Image data
                xb, yb = batch
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)

            loss.backward()
            opt.step()
    return model


# -------------------------
# Experiment: Build similarities
# -------------------------


def compute_ios_similarity_matrix(
    topk_sets: List[np.ndarray],
) -> np.ndarray:
    """
    Returns IOS similarity matrix (Jaccard similarity on top-k index sets).
    Shape: (n, n)
    """
    n = len(topk_sets)
    S_ios = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        S_ios[i, i] = 1.0
        for j in range(i + 1, n):
            s_ios = jaccard_similarity(topk_sets[i], topk_sets[j])
            S_ios[i, j] = S_ios[j, i] = s_ios
    return S_ios


def compute_cosine_oracle_matrices(
    importance_vectors: List[np.ndarray],
    label_distributions: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (S_cos, S_oracle) similarity matrices.
    - S_cos: Cosine similarity on full importance vectors
    - S_oracle: 1 - JS divergence on label distributions
    Shapes: (n, n)
    """
    n = len(importance_vectors)
    S_cos = np.zeros((n, n), dtype=np.float32)
    S_orc = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        S_cos[i, i] = 1.0
        S_orc[i, i] = 1.0
        for j in range(i + 1, n):
            s_cos = cosine_similarity(importance_vectors[i], importance_vectors[j])
            s_orc = oracle_distribution_similarity(
                label_distributions[i], label_distributions[j]
            )
            S_cos[i, j] = S_cos[j, i] = s_cos
            S_orc[i, j] = S_orc[j, i] = s_orc
    return S_cos, S_orc


def pairwise_matrices(
    importance_vectors: List[np.ndarray],
    topk_sets: List[np.ndarray],
    label_distributions: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (S_ios, S_cos, S_oracle) pairwise similarity matrices.
      - S_ios: Jaccard similarity on top-k index sets
      - S_cos: Cosine similarity on full importance vectors
      - S_oracle: 1 - JS divergence on label distributions
    Shapes: (n, n)
    """
    n = len(importance_vectors)
    S_ios = np.zeros((n, n), dtype=np.float32)
    S_cos = np.zeros((n, n), dtype=np.float32)
    S_orc = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        S_ios[i, i] = 1.0
        S_cos[i, i] = 1.0
        S_orc[i, i] = 1.0
        for j in range(i + 1, n):
            s_ios = jaccard_similarity(topk_sets[i], topk_sets[j])
            s_cos = cosine_similarity(importance_vectors[i], importance_vectors[j])
            s_orc = oracle_distribution_similarity(
                label_distributions[i], label_distributions[j]
            )
            S_ios[i, j] = S_ios[j, i] = s_ios
            S_cos[i, j] = S_cos[j, i] = s_cos
            S_orc[i, j] = S_orc[j, i] = s_orc
    return S_ios, S_cos, S_orc


def upper_triangle_flat(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    iu1 = np.triu_indices(n, k=1)
    return mat[iu1]


def ground_truth_correlation(
    S_ios: np.ndarray, S_cos: np.ndarray, S_oracle: np.ndarray
) -> Dict[str, float]:
    x = upper_triangle_flat(S_oracle)
    y_ios = upper_triangle_flat(S_ios)
    y_cos = upper_triangle_flat(S_cos)
    rho_ios = spearman_corr(x, y_ios)
    rho_cos = spearman_corr(x, y_cos)
    return {"spearman_ios_vs_oracle": rho_ios, "spearman_cos_vs_oracle": rho_cos}


def donor_rankings_from_similarity(S: np.ndarray, i: int) -> List[int]:
    n = S.shape[0]
    order = np.argsort(-S[i])
    order = [int(x) for x in order if x != i]
    return order


def shapley_style_ranking_eval(
    S_ios: np.ndarray, S_cos: np.ndarray, S_oracle: np.ndarray, k: int = 5
) -> Dict[str, float]:
    n = S_ios.shape[0]
    recalls_ios = []
    recalls_cos = []
    try:
        from scipy.stats import kendalltau

        have_kendall = True
    except Exception:
        have_kendall = False
    taus_ios = []
    taus_cos = []

    for i in range(n):
        r_true = donor_rankings_from_similarity(S_oracle, i)
        r_ios = donor_rankings_from_similarity(S_ios, i)
        r_cos = donor_rankings_from_similarity(S_cos, i)
        recalls_ios.append(topk_recall_at_k(r_true, r_ios, k))
        recalls_cos.append(topk_recall_at_k(r_true, r_cos, k))
        if have_kendall:

            def to_rank_map(rank_list):
                return {cid: pos for pos, cid in enumerate(rank_list)}

            m_true = to_rank_map(r_true)
            common = [cid for cid in r_true if cid in m_true]
            a = [m_true[cid] for cid in common]
            b_ios = [to_rank_map(r_ios).get(cid, len(r_ios)) for cid in common]
            b_cos = [to_rank_map(r_cos).get(cid, len(r_cos)) for cid in common]
            taus_ios.append(kendalltau(a, b_ios, variant="b").correlation)
            taus_cos.append(kendalltau(a, b_cos, variant="b").correlation)

    out = {
        "topk_recall_ios@{}".format(k): float(np.mean(recalls_ios)),
        "topk_recall_cos@{}".format(k): float(np.mean(recalls_cos)),
    }
    if len(taus_ios) > 0:
        out.update(
            {
                "kendall_tau_ios": float(np.nanmean(taus_ios)),
                "kendall_tau_cos": float(np.nanmean(taus_cos)),
            }
        )
    return out


# -------------------------
# New Use Case 3: Neighbor selection for personalized aggregation
# -------------------------


def neighbor_selection_eval(
    S_ios: np.ndarray,
    S_cos: np.ndarray,
    S_oracle: np.ndarray,
    label_dists: List[np.ndarray],
    k: int = 5,
    weighting: str = "weighted",
) -> Dict[str, float]:
    """
    For each client i, pick top-k neighbors by IOS / Cosine / Oracle similarities,
    estimate i's label distribution as a neighbor-weighted mixture, and compute JS
    divergence to i's true distribution. Report mean JS (lower is better).
    """
    n = S_ios.shape[0]

    def topk_estimate(S: np.ndarray, i: int) -> np.ndarray:
        order = donor_rankings_from_similarity(S, i)[:k]
        if len(order) == 0:
            return label_dists[i]
        if weighting == "weighted":
            w = np.maximum(S[i, order], 0.0)
            if w.sum() <= 1e-12:
                w = np.ones_like(w)
        else:
            w = np.ones(len(order), dtype=float)
        w = w / w.sum()
        mix = np.zeros_like(label_dists[i])
        for jj, ww in zip(order, w):
            mix += ww * label_dists[jj]
        return mix

    js_ios = []
    js_cos = []
    js_orc = []
    for i in range(n):
        p_true = label_dists[i]
        p_ios = topk_estimate(S_ios, i)
        p_cos = topk_estimate(S_cos, i)
        p_orc = topk_estimate(S_oracle, i)
        js_ios.append(js_divergence(p_true, p_ios))
        js_cos.append(js_divergence(p_true, p_cos))
        js_orc.append(js_divergence(p_true, p_orc))

    return {
        f"neighbor_js_ios@{k}": float(np.mean(js_ios)),
        f"neighbor_js_cos@{k}": float(np.mean(js_cos)),
        f"neighbor_js_oracle@{k}": float(np.mean(js_orc)),
    }


# -------------------------
# New Use Case 4: Drift detection
# -------------------------


def induce_client_drift(
    client_indices: List[np.ndarray],
    labels: np.ndarray,
    num_classes: int,
    drift_clients_mask: np.ndarray,
    swap_frac: float = 0.3,
) -> List[np.ndarray]:
    """
    Creates a new list of indices per client for "time-2" after drift.
    For clients with mask=1, replace a fraction of their samples with samples
    drawn from random *other* classes (sampling with replacement allowed).
    """
    N = len(client_indices)
    all_indices_by_class = [np.where(labels == c)[0] for c in range(num_classes)]
    rng = np.random.default_rng()

    new_indices = []
    for i in range(N):
        idxs = client_indices[i].copy()
        if drift_clients_mask[i] == 0:
            new_indices.append(idxs)
            continue
        m = len(idxs)
        swap = max(1, int(m * swap_frac))
        drop_sel = rng.choice(np.arange(m), size=swap, replace=False)
        keep_mask = np.ones(m, dtype=bool)
        keep_mask[drop_sel] = False
        kept = idxs[keep_mask]

        # choose target classes uniformly at random
        repl = []
        for _ in range(swap):
            c = int(rng.integers(0, num_classes))
            class_pool = all_indices_by_class[c]
            j = int(rng.integers(0, len(class_pool)))
            repl.append(int(class_pool[j]))
        repl = np.array(repl, dtype=int)
        merged = np.concatenate([kept, repl], axis=0)
        new_indices.append(np.array(sorted(merged), dtype=int))
    return new_indices


def drift_detection_eval(
    model_builder,
    model_state_dict,
    dataset: Dataset,
    label_array: np.ndarray,
    num_classes: int,
    client_indices_t1: List[np.ndarray],
    importance: str,
    device: str,
    topk_k: int,
    batch_size: int,
    max_batches_per_client: Optional[int],
    local_epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    drift_frac_clients: float = 0.3,
    drift_swap_frac: float = 0.3,
) -> Dict[str, float]:
    """
    Compute IOS self-overlap between time-1 and time-2 per client and compare with
    actual distribution shift (JS between label distributions at t1 and t2).
    Also report AUROC for detecting which clients drifted.
    """
    N = len(client_indices_t1)
    # Compute t1 label dists
    label_dists_t1 = [
        empirical_label_distribution(label_array, idxs, num_classes)
        for idxs in client_indices_t1
    ]

    # Build loaders and importance/topk at t1
    loaders_t1 = build_client_loaders(
        dataset, client_indices_t1, batch_size=batch_size, shuffle=True
    )

    imp_t1 = []
    topk_t1 = []
    for i in range(N):
        m = model_builder()
        m.load_state_dict(model_state_dict)

        imp = compute_importance_vector(
            m,
            loaders_t1[i],
            importance=importance,
            device=device,
            max_batches=max_batches_per_client,
            epochs=local_epochs,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            return_abs=False,
        )

        # imp = compute_importance_vector(m, loaders_t1[i], importance=importance, device=device, max_batches=max_batches_per_client)
        imp_t1.append(imp)
        topk_t1.append(topk_indices_from_vector(imp, topk_k))

    # Create drift mask
    n_drift = max(1, int(N * drift_frac_clients))
    drift_mask = np.zeros(N, dtype=int)
    drift_mask[np.random.choice(np.arange(N), size=n_drift, replace=False)] = 1

    # Induce drift and recompute
    client_indices_t2 = induce_client_drift(
        client_indices_t1,
        label_array,
        num_classes,
        drift_mask,
        swap_frac=drift_swap_frac,
    )
    label_dists_t2 = [
        empirical_label_distribution(label_array, idxs, num_classes)
        for idxs in client_indices_t2
    ]
    loaders_t2 = build_client_loaders(
        dataset, client_indices_t2, batch_size=batch_size, shuffle=True
    )

    imp_t2 = []
    topk_t2 = []
    for i in range(N):
        m = model_builder()
        m.load_state_dict(model_state_dict)
        imp = compute_importance_vector(
            m,
            loaders_t2[i],
            importance=importance,
            device=device,
            max_batches=max_batches_per_client,
            epochs=local_epochs,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            return_abs=False,
        )

        imp_t2.append(imp)
        topk_t2.append(topk_indices_from_vector(imp, topk_k))

    # Metrics per client
    self_overlap = []
    js_shift = []
    for i in range(N):
        a = set(topk_t1[i].tolist())
        b = set(topk_t2[i].tolist())
        inter = len(a.intersection(b))
        ov = inter / max(len(a.union(b)), 1)
        self_overlap.append(ov)
        js_shift.append(js_divergence(label_dists_t1[i], label_dists_t2[i]))

    self_overlap = np.asarray(self_overlap, dtype=float)
    js_shift = np.asarray(js_shift, dtype=float)

    # Higher score means more drift: 1 - overlap
    drift_scores = 1.0 - self_overlap
    # Correlation with true JS change
    rho = spearman_corr(drift_scores, js_shift)
    # AUROC against drift_mask labels
    auc = auc_binary(drift_scores.tolist(), drift_mask.tolist())

    return {
        "drift_spearman_overlap_vs_JS": float(rho),
        "drift_auroc": float(auc),
        "drift_frac_clients": float(drift_frac_clients),
        "drift_swap_frac": float(drift_swap_frac),
        "drift_mean_self_overlap": float(np.mean(self_overlap)),
        "drift_mean_JS": float(np.mean(js_shift)),
    }


# -------------------------
# Main pipeline
# -------------------------


def main():
    parser = argparse.ArgumentParser(description="IOS ML Benchmark")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["fmnist", "cifar10", "cifar100", "20newsgroups"],
        default="cifar10",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "resnet18", "resnet32", "resnet50", "bert_base"],
        default="resnet18",
    )
    parser.add_argument("--num_clients", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--partition", type=str, choices=["iid", "dir", "patho"], default="dir"
    )
    parser.add_argument("--dir_alpha", type=float, default=0.3)
    parser.add_argument(
        "--patho_n_classes",
        type=int,
        default=2,
        help="Num classes per client for pathological partition",
    )
    parser.add_argument(
        "--importance", type=str, choices=["magnitude", "fisher"], default="fisher"
    )
    parser.add_argument(
        "--topk_frac",
        type=float,
        default=0.12,
        help="Fraction of parameters to keep in IOS set",
    )
    parser.add_argument(
        "--max_batches_per_client",
        type=int,
        default=None,
        help="Limit batches per client when computing importance",
    )
    parser.add_argument(
        "--pretrain_global_epochs",
        type=int,
        default=0,
        help="Optional global pretrain epochs before importance computation",
    )
    parser.add_argument(
        "--usecases",
        type=str,
        choices=["correlation", "shapley", "neighbors", "drift", "all"],
        default="all",
    )
    parser.add_argument("--k_neighbors", type=int, default=5)
    parser.add_argument(
        "--neighbor_weighting",
        type=str,
        choices=["weighted", "uniform"],
        default="weighted",
    )
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Local SGD learning rate"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Local SGD momentum"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="Local SGD weight decay (L2)"
    )
    parser.add_argument("--drift_frac_clients", type=float, default=0.3)
    parser.add_argument("--drift_swap_frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--log_file", type=str, default=None, help="Optional log file path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda:0, cuda:1, cpu, or 'auto' for auto-detection). Default: auto",
    )
    parser.add_argument(
        "--bandwidth_mbps",
        type=float,
        default=200.0,
        help="Network bandwidth in MB/s for communication time calculation (default: 200 MB/s like Amazon EC2)",
    )
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_file)
    logger.info("=" * 80)
    logger.info(
        f"Starting IOS ML Benchmark - Model: {args.model}, Dataset: {args.dataset}"
    )
    logger.info("=" * 80)

    # Track timing for one round - only 2 main phases
    round_timings: Dict[str, float] = {}

    set_seed(args.seed)

    # Device selection with auto-detection
    if args.device is None or args.device.lower() == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
            num_gpus = torch.cuda.device_count()
            logger.info(f"Auto-detected {num_gpus} GPU(s), using: {device}")
        else:
            device = "cpu"
            logger.info("CUDA not available, using CPU")
    else:
        device = args.device
        if device.startswith("cuda:"):
            device_id = int(device.split(":")[1])
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"
            elif device_id >= torch.cuda.device_count():
                logger.warning(
                    f"GPU {device_id} not available (only {torch.cuda.device_count()} GPUs), falling back to cuda:0"
                )
                device = "cuda:0"
        logger.info(f"Using device: {device}")

    # ============================================================
    # PHASE 1: TRAINING (includes loading, setup, and all training)
    # ============================================================
    training_start = time.time()
    logger.info("=" * 80)
    logger.info("PHASE 1: TRAINING (Loading + Setup + Training)")
    logger.info("=" * 80)

    # Load dataset
    trainset, num_classes, in_ch = get_dataset(args.dataset, download=True)
    labels = labels_of_dataset(trainset)
    logger.info(f"Dataset loaded: {len(trainset)} samples, {num_classes} classes")

    # Build partition
    if args.partition == "iid":
        client_idxs = partition_iid(args.num_clients, labels)
    elif args.partition == "dir":
        client_idxs = partition_dirichlet(
            args.num_clients, labels, num_classes, args.dir_alpha
        )
    elif args.partition == "patho":
        client_idxs = partition_pathological_classes(
            args.num_clients, labels, num_classes, args.patho_n_classes
        )
    else:
        raise ValueError(f"Unknown partition: {args.partition}")

    loaders = build_client_loaders(
        trainset, client_idxs, batch_size=args.batch_size, shuffle=True
    )

    # Label distributions (oracle)
    label_dists = [
        empirical_label_distribution(labels, idxs, num_classes) for idxs in client_idxs
    ]

    # Build model
    def model_builder():
        return build_model(args.model, args.dataset, num_classes=num_classes)

    base_model = model_builder()
    total_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    logger.info(f"Model built: {args.model} with {total_params:,} trainable parameters")

    # Optional pretrain on full data
    if args.pretrain_global_epochs > 0:
        base_model = pretrain_global_model(
            base_model,
            trainset,
            batch_size=128,
            epochs=args.pretrain_global_epochs,
            device=device,
            lr=0.01,
        )

    base_state = base_model.state_dict()

    # Calculate model size in bytes (for communication time)
    model_size_bytes = sum(
        p.numel() * p.element_size() for p in base_model.parameters() if p.requires_grad
    )
    logger.info(
        f"Model size: {model_size_bytes:,} bytes ({model_size_bytes / (1024**2):.2f} MB)"
    )

    # Pre-compute parameter count for topk
    with torch.no_grad():
        dummy_vec = (
            flatten_params_like(
                [p.detach() for p in base_model.parameters() if p.requires_grad]
            )
            .cpu()
            .numpy()
        )
        total_params = len(dummy_vec)
    k = max(1, int(args.topk_frac * total_params))

    # Train models per client (training phase - only training, no importance extraction)
    logger.info(f"Training {len(loaders)} clients...")
    trained_models = []
    for cid, loader in enumerate(loaders):
        m_copy = model_builder()
        m_copy.load_state_dict(base_state)

        # Train the model (but don't extract importance yet)
        m_copy = m_copy.to(device)
        m_copy.train()
        opt = torch.optim.SGD(
            m_copy.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss(reduction="mean")

        for _ in range(args.local_epochs):
            batches_seen = 0
            for batch in loader:
                opt.zero_grad(set_to_none=True)

                # Handle both image data (xb, yb) and text data (dict)
                if isinstance(batch, dict):
                    # Text data (BERT)
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = m_copy(input_ids=input_ids, attention_mask=attention_mask)
                    if hasattr(outputs, "logits"):
                        logits = outputs.logits
                    else:
                        logits = outputs[0]
                    loss = loss_fn(logits, labels)
                else:
                    # Image data
                    xb, yb = batch
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = m_copy(xb)
                    loss = loss_fn(logits, yb)

                loss.backward()
                opt.step()

                batches_seen += 1
                if (
                    args.max_batches_per_client is not None
                    and batches_seen >= args.max_batches_per_client
                ):
                    break

        # Store the trained model for later importance extraction
        trained_models.append(m_copy.cpu())

    training_time = time.time() - training_start
    round_timings["Training"] = training_time
    logger.info("=" * 80)
    logger.info(f"PHASE 1 COMPLETED: Training Time = {training_time:.4f}s")
    logger.info("=" * 80)

    # ============================================================
    # PHASE 2: IOS COMPUTATION (importance extraction + top-k + similarity)
    # ============================================================
    ios_start = time.time()
    logger.info("=" * 80)
    logger.info("PHASE 2: IOS COMPUTATION (Importance Extraction + Top-K + Similarity)")
    logger.info("=" * 80)

    # Extract importance vectors and compute top-k sets
    importance_vecs: List[np.ndarray] = []
    topk_sets: List[np.ndarray] = []

    logger.info(
        f"Extracting importance vectors and computing top-k for {len(trained_models)} clients..."
    )
    for cid, trained_model in enumerate(trained_models):
        # Extract importance vector from trained model
        with torch.no_grad():
            params = [p.detach() for p in trained_model.parameters() if p.requires_grad]
            imp = flatten_params_like(params).cpu().numpy().astype(np.float32)

        # Compute top-k indices
        topk = topk_indices_from_vector(imp, k)

        importance_vecs.append(imp)
        topk_sets.append(topk)

    # Compute IOS similarity matrix
    S_ios = compute_ios_similarity_matrix(topk_sets)

    ios_time = time.time() - ios_start
    round_timings["IOS_Computation"] = ios_time
    logger.info("=" * 80)
    logger.info(f"PHASE 2 COMPLETED: IOS Computation Time = {ios_time:.4f}s")
    logger.info("=" * 80)

    # ============================================================
    # PHASE 3: COMMUNICATION TIME (model transfer)
    # ============================================================
    # Bandwidth: default 200 MB/s (like Amazon EC2), can be customized
    bandwidth_mbps = args.bandwidth_mbps
    bandwidth_bps = bandwidth_mbps * 1024 * 1024  # bytes per second

    # Communication time = model size / bandwidth
    communication_time = model_size_bytes / bandwidth_bps
    round_timings["Communication"] = communication_time
    logger.info("=" * 80)
    logger.info("PHASE 3: COMMUNICATION TIME")
    logger.info("=" * 80)
    logger.info(
        f"Model size: {model_size_bytes:,} bytes ({model_size_bytes / (1024**2):.2f} MB)"
    )
    logger.info(f"Bandwidth: {bandwidth_mbps} MB/s")
    logger.info(f"Communication Time: {communication_time:.4f}s")
    logger.info("=" * 80)

    # Compute other similarity matrices (not timed)
    logger.info("Computing Cosine and Oracle similarity matrices...")
    S_cos, S_oracle = compute_cosine_oracle_matrices(importance_vecs, label_dists)

    # Run evaluations (not timed)
    results = {}

    if args.usecases in ("correlation", "all"):
        corr = ground_truth_correlation(S_ios, S_cos, S_oracle)
        results.update(corr)

    if args.usecases in ("shapley", "all"):
        shap = shapley_style_ranking_eval(S_ios, S_cos, S_oracle, k=args.k_neighbors)
        results.update(shap)

    if args.usecases in ("neighbors", "all"):
        neigh = neighbor_selection_eval(
            S_ios,
            S_cos,
            S_oracle,
            label_dists,
            k=args.k_neighbors,
            weighting=args.neighbor_weighting,
        )
        results.update(neigh)

    if args.usecases in ("drift", "all"):
        drift = drift_detection_eval(
            model_builder=model_builder,
            model_state_dict=base_state,
            dataset=trainset,
            label_array=labels,
            num_classes=num_classes,
            client_indices_t1=client_idxs,
            importance=args.importance,
            device=device,
            topk_k=k,
            batch_size=args.batch_size,
            max_batches_per_client=args.max_batches_per_client,
            local_epochs=args.local_epochs,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            drift_frac_clients=args.drift_frac_clients,
            drift_swap_frac=args.drift_swap_frac,
        )
        results.update(drift)

    # Save results
    os.makedirs("ios_outputs", exist_ok=True)
    np.save(os.path.join("ios_outputs", "S_ios.npy"), S_ios)
    np.save(os.path.join("ios_outputs", "S_cos.npy"), S_cos)
    np.save(os.path.join("ios_outputs", "S_oracle.npy"), S_oracle)
    with open(os.path.join("ios_outputs", "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print final summary - 3 timings
    total_time = sum(round_timings.values())

    logger.info("=" * 80)
    logger.info("=== FINAL SUMMARY ===")
    logger.info(f"Training Time:     {round_timings['Training']:.4f}s")
    logger.info(f"IOS Computation:   {round_timings['IOS_Computation']:.4f}s")
    logger.info(f"Communication:     {round_timings['Communication']:.4f}s")
    logger.info(f"Total Time:        {total_time:.4f}s")
    logger.info("=" * 80)

    # Console output
    print("\n" + "=" * 80)
    print("=== TIMING SUMMARY ===")
    print(f"Training Time:     {round_timings['Training']:.4f}s")
    print(f"IOS Computation:   {round_timings['IOS_Computation']:.4f}s")
    print(f"Communication:     {round_timings['Communication']:.4f}s")
    print(f"Total Time:        {total_time:.4f}s")
    print("=" * 80)
    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

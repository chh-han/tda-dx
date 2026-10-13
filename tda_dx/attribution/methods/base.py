from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader


class AttributionMethod(ABC):
    """Base class for data attribution methods.

    To add a new method:
    1. Subclass this and implement setup(), cache_shard(), merge_attribute(), full(), self_sanity()
    2. Register in attributor/methods/__init__.py METHOD_REGISTRY
    """

    @abstractmethod
    def setup(
        self,
        stage_wrapper: torch.nn.Module,
        params: List[torch.nn.Parameter],
        ds_fields: Tuple[str, ...],
        device: str,
        args,  # argparse.Namespace
    ) -> None:
        """Initialize method-specific state. Called once after model is loaded."""

    @abstractmethod
    def cache_shard(
        self,
        shard_loader: DataLoader,
        shard_positions: List[int],
        results_folder: Path,
        shard_id: int,
        num_shards: int,
    ) -> None:
        """Compute and save intermediate representations for one training shard."""

    @abstractmethod
    def merge_attribute(
        self,
        num_train: int,
        results_folder: Path,
        num_shards: int,
        test_loader: DataLoader,
    ) -> torch.Tensor:
        """Load cached shards + compute test representations → scores (num_train, num_test)."""

    @abstractmethod
    def full(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ) -> torch.Tensor:
        """End-to-end attribution without sharding → scores (num_train, num_test)."""

    @abstractmethod
    def self_sanity(
        self,
        train_loader: DataLoader,
        num_train: int,
    ) -> Tuple[Dict, Dict]:
        """Compute self-sanity metrics (score each train sample against itself).

        Returns (metrics, arrays) matching the format of compute_self_sanity_metrics().
        - grad_cos (merge mode): free — reuses cached train_proj, no extra backward.
        - grad_cos (full mode): recomputes train projections.
        - TRAK: uses compute_trak_self_sanity_chunked().
        """

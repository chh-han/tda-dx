"""Block-Projected Influence Function Attributor.

General two-stage gradient compression.
"""

from __future__ import annotations

import inspect
import gc
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Dict, List, Literal, Optional, Tuple, Union

    from torch.utils.data import DataLoader

    from dattri.task import AttributionTask

import logging

import torch
from torch import nn
from tqdm import tqdm

from dattri.algorithm.base import BaseAttributor
from dattri.algorithm.block_projected_if.core.compressor import setup_model_compressors

# Import utilities
from dattri.algorithm.block_projected_if.core.hook import HookManager
from dattri.algorithm.block_projected_if.core.metadata import MetadataManager
from dattri.algorithm.block_projected_if.core.utils import stable_inverse
from dattri.algorithm.block_projected_if.offload import create_offload_manager
from dattri.algorithm.utils import _check_shuffle

logger = logging.getLogger(__name__)


class BlockProjectedIFAttributor(BaseAttributor):
    """Block-Projected Influence Function Attributor.

    A general attributor that computes influence scores using a two-stage
    compression pipeline with projected gradients for efficiency.
    Uses hooks to capture per-sample gradients and applies random projections.

    This is a general implementation that supports arbitrary compressor configurations.
    For specific methods like LoGra or FactGraSS, use the wrapper classes.
    """

    def __init__(
        self,
        task: "AttributionTask",
        layer_names: Optional[
            Union[str, List[str]]
        ] = None,  # Maybe support layer class as input?
        hessian: Literal["Identity", "eFIM"] = "eFIM",
        damping: Optional[float] = None,
        device: str = "cpu",
        sparsifier_kwargs: Optional[Dict[str, Any]] = None,
        projector_kwargs: Optional[Dict[str, Any]] = None,
        offload: Literal["none", "cpu", "disk"] = "cpu",
        cache_dir: Optional[str] = None,
        chunk_size: int = 16,
    ) -> None:
        """Initialize Block-Projected IF attributor.

        Args:
            task: Attribution task containing model, loss function, and checkpoints
            layer_names: Names of layers where gradients will be collected.
                If None, uses all Linear layers.
                You can check the names using model.named_modules().
                HookManager will register hooks to named layers.
            hessian: Type of Hessian approximation ("Identity", "eFIM"). For
                "Identity", the hessian will be taken as the identity matrix.
                For "eFIM", the hessian will be computed as the empirical
                fisher information matrix.
            damping: Damping factor for Hessian inverse (when hessian="eFIM")
            device: Device to run computations on
            sparsifier_kwargs: Arguments for sparsifier stage (first stage compression)
            projector_kwargs: Arguments for projector stage (second stage compression).
                If None, defaults to identity projection (no further compression).
            offload: Memory management strategy ("none", "cpu", "disk"), stating
                the place to offload the gradients.
                "cpu": stores gradients on CPU and moves to device when needed.
                "disk": stores gradients on disk and moves to device when needed.
            cache_dir: Directory for caching (required when offload="disk").
            chunk_size: Chunk size for processing in disk offload.

        Raises:
            ValueError: If cache_dir is None when offload="disk".
        """
        self.task = task
        self.device = device
        self.hessian = hessian
        self.damping = damping
        self.sparsifier_kwargs = sparsifier_kwargs or {
            device: device,
            "proj_dim": 32,
            "proj_max_batch_size": 64,
            "proj_type": "normal",
        }
        # Default projector_kwargs to identity if not specified
        self.projector_kwargs = projector_kwargs or {
            "device": device,
            "proj_dim": -1,
            "proj_max_batch_size": 64,
            "proj_type": "identity",
        }
        self.offload = offload
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size

        # Get model from task and setup
        task._load_checkpoints(0)
        self.model = task.get_model()
        self.model.to(device)

        # Determine layer names
        if layer_names is None:
            # Default to all Linear layers
            self.layer_names = [
                name
                for name, module in self.model.named_modules()
                if isinstance(module, nn.Linear)
            ]
        elif isinstance(layer_names, str):
            self.layer_names = [layer_names]
        else:
            self.layer_names = layer_names

        logger.info(
            "BlockProjectedIFAttributor initialized with %d layers: %s",
            len(self.layer_names),
            self.layer_names,
        )

        # Create offload manager
        self.offload_manager = create_offload_manager(
            offload_type=offload,
            device=device,
            layer_names=self.layer_names,
            cache_dir=cache_dir,
            chunk_size=chunk_size,
        )

        # Initialize metadata manager
        self.metadata = MetadataManager(cache_dir or ".", self.layer_names)

        # Initialize other components
        self.full_train_dataloader: Optional[DataLoader] = None
        self.hook_manager: Optional[HookManager] = None
        self.compressors: Optional[List[Any]] = None
        self.layer_dims: Optional[List[int]] = None
        self.total_proj_dim: Optional[int] = None
        self.release_train_gradients_after_ifvp = False

        # Validation
        if offload == "disk" and cache_dir is None:
            msg = "cache_dir must be provided when offload='disk'"
            raise ValueError(msg)

    def _move_batch_to_device(self, batch):
        """Move a dataloader batch to the configured device."""
        if isinstance(batch, dict):
            return {k: v.to(self.device) for k, v in batch.items()}
        if isinstance(batch, (tuple, list)):
            return tuple(v.to(self.device) for v in batch)
        return batch.to(self.device)

    def _get_batch_size(self, batch) -> int:
        """Infer batch size from a dict/tuple/tensor batch."""
        if isinstance(batch, dict):
            return next(iter(batch.values())).shape[0]
        if isinstance(batch, (tuple, list)):
            return batch[0].shape[0]
        return batch.shape[0]

    def _compute_batch_loss(self, batch) -> torch.Tensor:
        """Compute scalar loss for one batch using the task definition.

        This supports both:
        - dattri-style loss functions: `loss_func(params, data_pair)`
        - legacy project-specific loss functions: `loss_func(model, batch, device)`
        """
        batch = self._move_batch_to_device(batch)
        loss_func = self.task.get_loss_func(flatten=False)
        named_parameters = {
            name: param
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        try:
            return loss_func(named_parameters, batch)
        except TypeError as exc:
            try:
                return loss_func(self.model, batch, self.device)
            except TypeError:
                signature = inspect.signature(loss_func)
                msg = (
                    "Unsupported loss function signature for BlockProjectedIF. "
                    "Expected either (params, batch) or (model, batch, device), "
                    f"but got {signature}."
                )
                raise TypeError(msg) from exc

    def _setup_compressors(self, train_dataloader: DataLoader) -> None:
        """Set up compressors for the model layers.

        Args:
            train_dataloader: DataLoader for training data used to get sample inputs.
        """
        if not self.sparsifier_kwargs and not self.projector_kwargs:
            self.compressors = []
            return

        logger.info("Setting up compressors...")

        # Get sample batch from dataloader
        sample_batch = next(iter(train_dataloader))
        if isinstance(sample_batch, dict):
            sample_inputs = {k: v.to(self.device) for k, v in sample_batch.items()}
        else:
            sample_inputs = tuple(v.to(self.device) for v in sample_batch)

        # Use separate kwargs for sparsifier and projector stages
        self.compressors = setup_model_compressors(
            self.model,
            self.layer_names,
            sparsifier_kwargs=self.sparsifier_kwargs,
            projector_kwargs=self.projector_kwargs,
            sample_inputs=sample_inputs,
            device=self.device,
            forward_func=lambda batch: self._compute_batch_loss(batch),
        )

    def _sync_layer_dims(self) -> None:
        """Synchronize layer dimensions between components."""
        if self.layer_dims is None:
            if self.metadata.layer_dims is not None:
                self.layer_dims = self.metadata.layer_dims
                self.total_proj_dim = self.metadata.total_proj_dim
            elif (
                hasattr(self.offload_manager, "layer_dims")
                and self.offload_manager.layer_dims is not None
            ):
                self.layer_dims = self.offload_manager.layer_dims
                self.total_proj_dim = self.offload_manager.total_proj_dim

        # Sync to all components
        if self.layer_dims is not None:
            if hasattr(self.offload_manager, "layer_dims"):
                self.offload_manager.layer_dims = self.layer_dims
                self.offload_manager.total_proj_dim = self.total_proj_dim
            if self.metadata.layer_dims is None:
                self.metadata.set_layer_dims(self.layer_dims)

    def cache(self, full_train_dataloader: DataLoader) -> None:  # noqa: PLR0912
        """Cache gradients and IFVP for the full training dataset.

        Args:
            full_train_dataloader: DataLoader for full training data
        """
        logger.info(
            "Caching gradients with BlockProjectedIF (offload: %s)",
            self.offload,
        )

        self.full_train_dataloader = full_train_dataloader

        # Setup compressors if not already done
        if self.compressors is None:
            self._setup_compressors(full_train_dataloader)

        # Initialize metadata for complete dataset
        self.metadata.initialize_complete_dataset(full_train_dataloader)

        # Create hook manager
        if self.hook_manager is None:
            self.hook_manager = HookManager(
                self.model,
                self.layer_names,
                device=self.device,
            )
            if self.compressors:
                self.hook_manager.set_compressors(self.compressors)

        # Collect gradients using hooks
        logger.info("Computing gradients using hooks...")

        # Start batch range processing for disk offload (chunk completion tracking)
        if hasattr(self.offload_manager, "start_batch_range_processing"):
            self.offload_manager.start_batch_range_processing(
                0, len(full_train_dataloader),
            )

        for batch_idx, batch in enumerate(
            tqdm(full_train_dataloader, desc="Computing gradients"),
        ):
            self.model.zero_grad()

            # Prepare inputs and compute loss
            loss = self._compute_batch_loss(batch)

            # Backward pass
            loss.backward()

            # Get projected gradients from hooks
            with torch.no_grad():
                compressed_grads = self.hook_manager.get_compressed_grads()

                # Detect layer dimensions on first batch
                batch_size = None
                if self.layer_dims is None:
                    self.layer_dims = []
                    for grad in compressed_grads:
                        if grad is not None and grad.numel() > 0:
                            self.layer_dims.append(
                                grad.shape[1] if grad.dim() > 1 else grad.numel(),
                            )
                            batch_size = grad.shape[0] if grad.dim() > 1 else 1
                        else:
                            self.layer_dims.append(0)

                    self.total_proj_dim = sum(self.layer_dims)
                    logger.info(
                        "Detected layer dimensions: %d layers, total dimension=%d",
                        len(self.layer_dims),
                        self.total_proj_dim,
                    )

                    # Save to metadata manager
                    self.metadata.add_batch_info(batch_idx, batch_size)
                    self.metadata.set_layer_dims(self.layer_dims)

                    # Pass to offload manager if needed
                    if hasattr(self.offload_manager, "layer_dims"):
                        self.offload_manager.layer_dims = self.layer_dims
                        self.offload_manager.total_proj_dim = self.total_proj_dim

                # Store gradients using offload manager
                self.offload_manager.store_gradients(
                    batch_idx,
                    compressed_grads,
                    is_test=False,
                )

            # Periodic empty_cache only on a coarse interval to reduce GPU stalls
            if batch_idx % 200 == 199:
                torch.cuda.empty_cache()

        # Finish batch range processing for disk offload (flushes remaining buffers)
        if hasattr(self.offload_manager, "finish_batch_range_processing"):
            self.offload_manager.finish_batch_range_processing()

        # Save metadata
        self.metadata.save_metadata()

        # Clean up hooks
        self.hook_manager.remove_hooks()
        self.hook_manager = None

        # Compute preconditioners and IFVP if needed
        if self.hessian == "eFIM":
            logger.info("Computing preconditioners...")
            self.compute_preconditioners()
            logger.info("Computing IFVP...")
            self.compute_ifvp()

        logger.info("Caching completed")

    def compute_preconditioners(  # noqa: PLR0912, PLR0914 - Complex preconditioner computation requires multiple branches and variables
        self,
        damping: Optional[float] = None,
    ) -> None:
        """Compute preconditioners (inverse Hessian) from gradients.

        Args:
            damping: Damping factor for numerical stability

        Raises:
            ValueError: If layer dimensions are not found.
        """
        logger.info("Computing preconditioners...")

        if damping is None:
            damping = self.damping

        # Load batch information if needed
        if not self.metadata.batch_info:
            self.metadata._load_metadata_if_exists()

        # Synchronize layer dimensions
        self._sync_layer_dims()

        if self.layer_dims is None:
            msg = (
                "Layer dimensions not found. "
                "Ensure gradients have been computed and stored."
            )
            raise ValueError(
                msg,
            )

        # If hessian type is "Identity", no preconditioners needed
        if self.hessian == "Identity":
            logger.info("Hessian type is 'none', skipping preconditioner computation")
            for layer_idx in range(len(self.layer_names)):
                self.offload_manager.store_preconditioner(layer_idx, None)
            return

        total_samples = self.metadata.get_total_samples()
        logger.info("Computing preconditioners from %s total samples", total_samples)

        # Initialize Hessian accumulators
        hessian_accumulators = []
        sample_counts = [0] * len(self.layer_names)

        for layer_idx in range(len(self.layer_names)):
            layer_dim = self.layer_dims[layer_idx]
            if layer_dim > 0:
                hessian_accumulators.append(
                    torch.zeros(
                        layer_dim,
                        layer_dim,
                        device=self.device,
                        dtype=torch.float32,
                    ),
                )
            else:
                hessian_accumulators.append(None)

        # Use tensor-based dataloader for efficient processing
        dataloader = self.offload_manager.create_gradient_dataloader(
            data_type="gradients",
            batch_size=32,
            pin_memory=True,
        )

        # Precompute layer column offsets once (avoids O(L) sum per chunk).
        layer_offsets = []
        running = 0
        for d in self.layer_dims:
            layer_offsets.append(running)
            running += d

        for chunk_tensor, _batch_mapping in tqdm(
            dataloader,
            desc="Computing preconditioners",
        ):
            # Move chunk to device
            chunk_tensor_device = self.offload_manager.move_to_device(chunk_tensor)

            # Process each layer
            for layer_idx in range(len(self.layer_names)):
                start_col = layer_offsets[layer_idx]
                end_col = start_col + self.layer_dims[layer_idx]
                layer_data = chunk_tensor_device[:, start_col:end_col].detach()

                if layer_data.numel() == 0 or hessian_accumulators[layer_idx] is None:
                    continue

                sample_counts[layer_idx] += layer_data.shape[0]

                # In-place accumulation: H += g^T @ g
                hessian_accumulators[layer_idx].addmm_(layer_data.t(), layer_data)

        # Compute preconditioners from accumulated Hessians
        computed_count = 0
        for layer_idx in tqdm(range(len(self.layer_names)), desc="Computing inverses"):
            hessian_accumulator = hessian_accumulators[layer_idx]
            sample_count = sample_counts[layer_idx]

            if hessian_accumulator is not None and sample_count > 0:
                # Normalize by total number of samples
                hessian = hessian_accumulator / sample_count

                # Compute inverse
                if self.hessian == "eFIM":
                    precond = stable_inverse(hessian, damping=damping)
                    self.offload_manager.store_preconditioner(layer_idx, precond)

                computed_count += 1

        self.offload_manager.wait_for_async_operations()
        logger.info("Computed %s preconditioners", computed_count)

    def compute_ifvp(self) -> None:
        """Compute inverse-Hessian-vector products (IFVP).

        Here we use empirical fisher information matrix.

        Raises:
            ValueError: If layer dimensions are not found.
        """
        logger.info("Computing IFVP...")

        # Load batch information if needed
        if not self.metadata.batch_info:
            self.metadata._load_metadata_if_exists()

        # Synchronize layer dimensions
        self._sync_layer_dims()

        if self.layer_dims is None:
            msg = "Layer dimensions not found."
            raise ValueError(msg)

        # Return raw gradients if Hessian type is "Identity"
        if self.hessian == "Identity":
            logger.debug("Using raw gradients as IFVP since hessian type is 'Identity'")
            self._copy_gradients_as_ifvp()
            return

        # Load all preconditioners and cache them on device once.
        # Hoisting this out of the chunk loop avoids repeatedly transferring
        # ~L * proj_dim^2 floats (potentially > 1 GB per batch) from CPU/disk.
        preconditioners = []
        device_preconds = []
        precompute_offsets = []
        running = 0
        for layer_idx in range(len(self.layer_names)):
            precond = self.offload_manager.retrieve_preconditioner(layer_idx)
            preconditioners.append(precond)
            precompute_offsets.append(running)
            running += self.layer_dims[layer_idx]
            if precond is None:
                device_preconds.append(None)
            else:
                device_preconds.append(self.offload_manager.move_to_device(precond))

        # Get batch mapping
        batch_to_sample_mapping = self.metadata.get_batch_to_sample_mapping()
        processed_batches = 0
        processed_samples = 0

        # Start batch range processing for IFVP storage
        if hasattr(self.offload_manager, "start_batch_range_processing"):
            self.offload_manager.start_batch_range_processing(
                0, len(batch_to_sample_mapping),
            )

        # Use tensor-based dataloader. Larger chunk batch reduces per-chunk
        # overhead (kernel launches, H2D, file open) without changing math.
        dataloader = self.offload_manager.create_gradient_dataloader(
            data_type="gradients",
            batch_size=32,
            pin_memory=True,
        )

        # Cache dtype-cast device preconditioners lazily on first use.
        cast_device_preconds = [None] * len(self.layer_names)

        for chunk_tensor, batch_mapping in tqdm(dataloader, desc="Computing IFVP"):
            # Move chunk to device
            chunk_tensor_device = self.offload_manager.move_to_device(chunk_tensor)

            # Process each batch in the chunk
            for batch_idx, (start_row, end_row) in batch_mapping.items():
                if batch_idx not in batch_to_sample_mapping:
                    continue

                batch_tensor = chunk_tensor_device[start_row:end_row]
                batch_ifvp = []

                # Process each layer
                for layer_idx in range(len(self.layer_names)):
                    if device_preconds[layer_idx] is None:
                        batch_ifvp.append(
                            torch.zeros(
                                batch_tensor.shape[0],
                                self.layer_dims[layer_idx],
                            ),
                        )
                        continue

                    start_col = precompute_offsets[layer_idx]
                    end_col = start_col + self.layer_dims[layer_idx]
                    layer_grad = batch_tensor[:, start_col:end_col]

                    if layer_grad.numel() == 0:
                        batch_ifvp.append(
                            torch.zeros(
                                batch_tensor.shape[0],
                                self.layer_dims[layer_idx],
                            ),
                        )
                        continue

                    # Cast preconditioner to layer dtype once and reuse.
                    cached = cast_device_preconds[layer_idx]
                    if cached is None or cached.dtype != layer_grad.dtype:
                        cached = device_preconds[layer_idx].to(dtype=layer_grad.dtype)
                        cast_device_preconds[layer_idx] = cached

                    # Compute IFVP: H^{-1} @ g  (precond is symmetric SPD)
                    ifvp = layer_grad @ cached
                    batch_ifvp.append(ifvp)

                # Store IFVP for this batch
                self.offload_manager.store_ifvp(batch_idx, batch_ifvp)
                processed_batches += 1
                processed_samples += batch_tensor.shape[0]

            if self.release_train_gradients_after_ifvp:
                self.offload_manager.clear_gradient_batches(list(batch_mapping))

            del chunk_tensor_device

        # Finish batch range processing for IFVP storage
        if hasattr(self.offload_manager, "finish_batch_range_processing"):
            self.offload_manager.finish_batch_range_processing()

        self.offload_manager.wait_for_async_operations()
        logger.info(
            "Computed IFVP for %s batches, %s samples",
            processed_batches,
            processed_samples,
        )

    def _copy_gradients_as_ifvp(self) -> None:
        """Copy gradients as IFVP when hessian type is 'none'."""
        # Ensure layer dimensions are loaded
        if self.layer_dims is None:
            self._sync_layer_dims()

        batch_to_sample_mapping = self.metadata.get_batch_to_sample_mapping()
        processed_batches = 0
        processed_samples = 0

        # Start batch range processing for IFVP storage
        if hasattr(self.offload_manager, "start_batch_range_processing"):
            self.offload_manager.start_batch_range_processing(
                0, len(batch_to_sample_mapping),
            )

        # Process using tensor dataloader
        dataloader = self.offload_manager.create_gradient_dataloader(
            data_type="gradients",
            batch_size=1,
            pin_memory=True,
        )

        for chunk_tensor, batch_mapping in tqdm(
            dataloader,
            desc="Copying gradients as IFVP",
        ):
            for batch_idx, (start_row, end_row) in batch_mapping.items():
                if batch_idx not in batch_to_sample_mapping:
                    continue

                # Extract batch and split into layers
                batch_tensor = chunk_tensor[start_row:end_row]
                gradients = []

                for layer_idx in range(len(self.layer_names)):
                    start_col = sum(self.layer_dims[:layer_idx])
                    end_col = start_col + self.layer_dims[layer_idx]
                    gradients.append(batch_tensor[:, start_col:end_col].contiguous())

                self.offload_manager.store_ifvp(batch_idx, gradients)
                processed_batches += 1
                processed_samples += batch_tensor.shape[0]

        # Finish batch range processing for IFVP storage
        if hasattr(self.offload_manager, "finish_batch_range_processing"):
            self.offload_manager.finish_batch_range_processing()

        logger.info("Copied %s batches as IFVP", processed_batches)

    def compute_self_attribution(  # noqa: PLR0914 - Complex self-attribution computation requires multiple local variables for tracking state
        self,
    ) -> torch.Tensor:
        """Compute self-influence scores.

        Returns:
            Self-influence scores
        """
        logger.info("Computing self-influence scores")

        if not self.metadata.batch_info:
            self.metadata._load_metadata_if_exists()

        # Synchronize layer dimensions
        self._sync_layer_dims()

        # Make sure IFVP is computed
        if not self.offload_manager.has_ifvp():
            logger.info("IFVP not found, computing it now...")
            self.compute_ifvp()

        # Get batch mapping
        batch_to_sample_mapping = self.metadata.get_batch_to_sample_mapping()
        total_samples = self.metadata.get_total_samples()
        self_influence = torch.zeros(total_samples, device="cpu")

        # Use tensor dataloaders for both gradients and IFVP
        grad_dataloader = self.offload_manager.create_gradient_dataloader(
            data_type="gradients",
            batch_size=4,
            pin_memory=True,
        )

        ifvp_dataloader = self.offload_manager.create_gradient_dataloader(
            data_type="ifvp",
            batch_size=4,
            pin_memory=True,
        )

        if grad_dataloader and ifvp_dataloader:
            # Process in parallel
            for (grad_tensor, grad_mapping), (ifvp_tensor, ifvp_mapping) in tqdm(
                zip(grad_dataloader, ifvp_dataloader),
                desc="Computing self-influence",
                total=len(grad_dataloader),
            ):
                # Move to device
                grad_tensor_device = self.offload_manager.move_to_device(grad_tensor)
                ifvp_tensor_device = self.offload_manager.move_to_device(ifvp_tensor)

                # Process each batch
                for batch_idx in grad_mapping:
                    if (
                        batch_idx not in batch_to_sample_mapping
                        or batch_idx not in ifvp_mapping
                    ):
                        continue

                    sample_start, sample_end = batch_to_sample_mapping[batch_idx]

                    # Extract batch slices
                    grad_start, grad_end = grad_mapping[batch_idx]
                    ifvp_start, ifvp_end = ifvp_mapping[batch_idx]

                    batch_grad = grad_tensor_device[grad_start:grad_end]
                    batch_ifvp = ifvp_tensor_device[ifvp_start:ifvp_end]

                    # Compute dot product
                    batch_influence = torch.sum(batch_grad * batch_ifvp, dim=1).cpu()
                    self_influence[sample_start:sample_end] = batch_influence

        return self_influence

    def attribute(  # noqa: PLR0914 - Complex attribution computation requires multiple local variables for dataloader management and tensor processing
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ) -> torch.Tensor:
        """Compute influence scores between training and test samples.

        Args:
            train_dataloader: Training data (can be subset if cache was called)
            test_dataloader: Test data to compute influence for

        Returns:
            Influence score tensor of shape (num_train, num_test)
        """
        logger.info("Computing influence attribution with BlockProjectedIF")

        # Validation
        _check_shuffle(train_dataloader)
        _check_shuffle(test_dataloader)

        # Handle training data
        if self.full_train_dataloader is None:
            warnings.warn(
                "The full training data loader was NOT cached. "
                "Treating the train_dataloader as the full training data loader.",
                stacklevel=2,
            )
            self.cache(train_dataloader)

        # Load batch information if needed
        if not self.metadata.batch_info:
            self.metadata._load_metadata_if_exists()

        # Synchronize layer dimensions
        self._sync_layer_dims()

        # Set up compressors if needed
        if self.compressors is None:
            self._setup_compressors(test_dataloader)

        # Get or compute IFVP
        use_cached_ifvp = self.offload_manager.has_ifvp()
        if use_cached_ifvp:
            logger.info("Using cached IFVP")
        else:
            logger.info("Computing IFVP")
            self.compute_ifvp()

        # Compute test gradients
        logger.info("Computing test gradients")
        test_grads_tensor, _test_batch_mapping = self._compute_test_gradients(
            test_dataloader,
        )

        if test_grads_tensor is None or test_grads_tensor.numel() == 0:
            logger.warning("No test gradients computed")
            num_train = (
                len(train_dataloader.sampler)
                if hasattr(train_dataloader, "sampler")
                else len(train_dataloader.dataset)
            )
            num_test = (
                len(test_dataloader.sampler)
                if hasattr(test_dataloader, "sampler")
                else len(test_dataloader.dataset)
            )
            return torch.zeros(num_train, num_test)

        # Get batch mappings
        batch_to_sample_mapping = self.metadata.get_batch_to_sample_mapping()
        total_train_samples = self.metadata.get_total_samples()
        test_sample_count = test_grads_tensor.shape[0]

        # Initialize result
        if_score = torch.zeros(
            total_train_samples,
            test_sample_count,
            device=self.device,
        )

        # Create dataloader for IFVP. Larger chunk batch reduces I/O / launch
        # overhead; the math is identical.
        train_ifvp_dataloader = self.offload_manager.create_gradient_dataloader(
            data_type="ifvp",
            batch_size=32,
            pin_memory=True,
        )

        logger.info("Starting influence computation")

        # If the entire test grad tensor fits comfortably, keep it on device
        # once instead of re-uploading per chunk.
        try:
            test_grads_device = self.offload_manager.move_to_device(test_grads_tensor)
        except RuntimeError:
            test_grads_device = None

        # Configure test batching for memory efficiency
        test_batch_size = min(256, test_sample_count)

        # Single pass through training IFVP data with nested test batching
        for chunk_tensor, batch_mapping in tqdm(
            train_ifvp_dataloader,
            desc="Computing attribution",
        ):
            # Move train chunk to device
            chunk_tensor_device = self.offload_manager.move_to_device(chunk_tensor).to(
                dtype=test_grads_tensor.dtype,
            )

            # Process test gradients in batches to save memory
            for test_start in range(0, test_sample_count, test_batch_size):
                test_end = min(test_start + test_batch_size, test_sample_count)
                if test_grads_device is not None:
                    test_batch_device = test_grads_device[test_start:test_end]
                else:
                    test_batch = test_grads_tensor[test_start:test_end]
                    test_batch_device = self.offload_manager.move_to_device(test_batch)

                # Efficient batched matrix multiplication
                chunk_scores = torch.matmul(chunk_tensor_device, test_batch_device.t())

                # Map chunk results back to global sample indices
                for batch_idx, (start_row, end_row) in batch_mapping.items():
                    if batch_idx not in batch_to_sample_mapping:
                        continue

                    train_start, train_end = batch_to_sample_mapping[batch_idx]
                    batch_scores = chunk_scores[start_row:end_row]
                    if_score[train_start:train_end, test_start:test_end] = (
                        batch_scores.to(if_score.device)
                    )

            del chunk_tensor_device

        logger.info(
            "Attribution computation completed. Result shape: %s",
            if_score.shape,
        )
        return if_score

    def _compute_test_gradients(  # noqa: PLR0912 - Complex gradient computation requires multiple branches for different input types and error handling
        self,
        test_dataloader: DataLoader,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute gradients for test data using hooks.

        Args:
            test_dataloader: DataLoader for test data to compute gradients for.

        Returns:
            Tuple containing:
                - Tensor of concatenated test gradients
                  Shape:(num_test_samples, total_proj_dim)
                - Dictionary mapping batch indices to (start_row, end_row)
                  positions.

        Raises:
            ValueError: If model can't return loss for dict-style inputs.
        """
        # Create hook manager if needed
        if self.hook_manager is None:
            self.hook_manager = HookManager(
                self.model,
                self.layer_names,
                device=self.device,
            )
            if self.compressors:
                self.hook_manager.set_compressors(self.compressors)

        all_test_grads = []
        test_batch_mapping = {}
        current_row = 0

        for batch_idx, batch in enumerate(
            tqdm(test_dataloader, desc="Computing test gradients"),
        ):
            self.model.zero_grad()
            batch = self._move_batch_to_device(batch)
            batch_size = self._get_batch_size(batch)

            # Prepare inputs and compute loss
            loss = self._compute_batch_loss(batch)

            loss.backward()

            # Get projected gradients
            with torch.no_grad():
                compressed_grads = self.hook_manager.get_compressed_grads()

                # Concatenate layer gradients
                batch_features = []
                for grad_idx, grad in enumerate(compressed_grads):
                    if grad is not None and grad.numel() > 0:
                        batch_features.append(grad.cpu())
                    elif self.layer_dims and grad_idx < len(self.layer_dims):
                        batch_features.append(
                            torch.zeros(batch_size, self.layer_dims[grad_idx]),
                        )

                if batch_features:
                    batch_tensor = torch.cat(batch_features, dim=1)
                    all_test_grads.append(batch_tensor)
                    test_batch_mapping[batch_idx] = (
                        current_row,
                        current_row + batch_size,
                    )
                    current_row += batch_size

        # Clean up hooks
        self.hook_manager.remove_hooks()
        self.hook_manager = None

        # Combine all test gradients
        if all_test_grads:
            return torch.cat(all_test_grads, dim=0), test_batch_mapping
        return torch.empty(0, self.total_proj_dim or 0), {}

"""
attributor/methods/factgrass.py

FactGraSS attribution method wrapper.

This wrapper intentionally follows the upstream GraSS repository's math flow:
- use BlockProjectedIFAttributor directly
- configure FactGraSS through random_mask -> sjlt compression
- keep the shared runner/task pipeline unchanged

Compatibility fix
-----------------
dattri's BlockProjectedIFAttributor has two data-format issues for MusicLM:

1. cache() calls task.original_loss_func(model, batch, device) — a 3-arg
   convention — but our task stores loss_fn(params, data_pair), a 2-arg
   functional loss.  Calling it with 3 positional args raises TypeError.

2. _compute_test_gradients() does `inputs, targets = batch`, which fails
   for MusicLM's 3-field tuples (clap, coarse, fine token ids).

Both are fixed by _MusicLMBlockProjectedIFAttributor below, which patches
task.original_loss_func and overrides _compute_test_gradients to use
stage_wrapper(**data_kwargs) directly — consistent with TRAK/grad_cos.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tda_dx.attribution.methods.base import AttributionMethod
from tda_dx.attribution.methods.block_projected_utils import validate_single_batch_contract
from tda_dx.attribution.model.setup import configure_trainable_params
from tda_dx.attribution.sanity.metrics import compute_self_sanity_metrics
from dattri.algorithm.block_projected_if.core.hook import HookManager


# ─────────────────────────────────────────────────────────────────────────────
# MusicLM-compatible BlockProjectedIFAttributor
# ─────────────────────────────────────────────────────────────────────────────

class _MusicLMBlockProjectedIFAttributor:
    """Thin mixin injected before BlockProjectedIFAttributor in the MRO.

    Fixes two incompatibilities between dattri's BlockProjectedIFAttributor
    and MusicLM's data / loss conventions (see module docstring).
    """

    def __init__(self, *args, stage_wrapper, ds_fields, **kwargs):
        super().__init__(*args, **kwargs)
        self._stage_wrapper = stage_wrapper
        self._ds_fields = ds_fields

        # Patch task.original_loss_func so that cache()'s
        #   loss = self.task.original_loss_func(self.model, batch, self.device)
        # call works without TypeError.  We ignore model/device — stage_wrapper
        # already holds the right model and device closure.
        _sw = stage_wrapper
        _df = ds_fields
        _dev = self.device

        def _musiclm_loss(model_unused, batch, device_unused=None):
            data_kwargs = {k: batch[i].to(_dev) for i, k in enumerate(_df)}
            loss, _, _ = _sw(**data_kwargs, return_loss=True)
            return loss

        self.task.original_loss_func = _musiclm_loss

    # ------------------------------------------------------------------
    # Used by _probe_invalid_layers in FactGraSSMethod
    # ------------------------------------------------------------------
    def _compute_batch_loss(self, batch):
        data_kwargs = {k: batch[i].to(self.device) for i, k in enumerate(self._ds_fields)}
        loss, _, _ = self._stage_wrapper(**data_kwargs, return_loss=True)
        return loss

    # ------------------------------------------------------------------
    # Override: replace `inputs, targets = batch` with stage_wrapper call
    # ------------------------------------------------------------------
    def _cache_gradients_only(self, train_loader: DataLoader, compressor_seed: int) -> None:
        """Cache only compressed training gradients — skip Hessian and IFVP steps.

        Compressor initialisation is isolated from the global RNG by saving/restoring
        state around the setup call, then re-seeding with *compressor_seed*.  Every
        shard job that calls this with the same seed and model will produce bit-for-bit
        identical compressor matrices, which is the single requirement for correct
        shard/merge results.
        """
        # ── Deterministic compressor setup ───────────────────────────────────
        _cpu_state = torch.get_rng_state()
        _cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(compressor_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(compressor_seed)
        if self.compressors is None:
            self._setup_compressors(train_loader)
        torch.set_rng_state(_cpu_state)
        if _cuda_states is not None:
            torch.cuda.set_rng_state_all(_cuda_states)

        self.full_train_dataloader = train_loader
        self.metadata.initialize_complete_dataset(train_loader)

        if self.hook_manager is None:
            self.hook_manager = HookManager(self.model, self.layer_names, device=self.device)
            if self.compressors:
                self.hook_manager.set_compressors(self.compressors)

        if hasattr(self.offload_manager, "start_batch_range_processing"):
            self.offload_manager.start_batch_range_processing(0, len(train_loader))

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Computing gradients")):
            self.model.zero_grad()
            data_kwargs = {k: batch[i].to(self.device) for i, k in enumerate(self._ds_fields)}
            loss, _, _ = self._stage_wrapper(**data_kwargs, return_loss=True)
            loss.backward()

            with torch.no_grad():
                compressed_grads = self.hook_manager.get_compressed_grads()
                if self.layer_dims is None:
                    self.layer_dims, batch_size = [], None
                    for grad in compressed_grads:
                        if grad is not None and grad.numel() > 0:
                            self.layer_dims.append(grad.shape[1] if grad.dim() > 1 else grad.numel())
                            batch_size = grad.shape[0] if grad.dim() > 1 else 1
                        else:
                            self.layer_dims.append(0)
                    self.total_proj_dim = sum(self.layer_dims)
                    self.metadata.add_batch_info(batch_idx, batch_size)
                    self.metadata.set_layer_dims(self.layer_dims)
                    if hasattr(self.offload_manager, "layer_dims"):
                        self.offload_manager.layer_dims = self.layer_dims
                        self.offload_manager.total_proj_dim = self.total_proj_dim
                self.offload_manager.store_gradients(batch_idx, compressed_grads, is_test=False)

            if batch_idx % 200 == 199:
                torch.cuda.empty_cache()

        if hasattr(self.offload_manager, "finish_batch_range_processing"):
            self.offload_manager.finish_batch_range_processing()
        self.metadata.save_metadata()
        self.hook_manager.remove_hooks()
        self.hook_manager = None

    def _compute_test_gradients(self, test_dataloader):
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
            tqdm(test_dataloader, desc="Computing test gradients")
        ):
            self.model.zero_grad()
            batch_size = batch[0].shape[0]

            data_kwargs = {k: batch[i].to(self.device) for i, k in enumerate(self._ds_fields)}
            loss, _, _ = self._stage_wrapper(**data_kwargs, return_loss=True)
            loss.backward()

            with torch.no_grad():
                compressed_grads = self.hook_manager.get_compressed_grads()
                batch_features = []
                for grad_idx, grad in enumerate(compressed_grads):
                    if grad is not None and grad.numel() > 0:
                        batch_features.append(grad.cpu())
                    elif self.layer_dims and grad_idx < len(self.layer_dims):
                        batch_features.append(
                            torch.zeros(batch_size, self.layer_dims[grad_idx])
                        )
                if batch_features:
                    batch_tensor = torch.cat(batch_features, dim=1)
                    all_test_grads.append(batch_tensor)
                    test_batch_mapping[batch_idx] = (current_row, current_row + batch_size)
                    current_row += batch_size

        self.hook_manager.remove_hooks()
        self.hook_manager = None

        if all_test_grads:
            return torch.cat(all_test_grads, dim=0), test_batch_mapping
        return torch.empty(0, self.total_proj_dim or 0), {}


def _resolve_factgrass_layer_names(model: nn.Module) -> List[str]:
    """Match the upstream GraSS default: hook all trainable Linear modules."""
    layer_names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(param.requires_grad for param in module.parameters(recurse=False)):
            layer_names.append(name)
    return layer_names


def _configure_factgrass_params(model: nn.Module) -> List[str]:
    """Restrict grads to the params owned by the selected Linear modules only."""
    layer_names = _resolve_factgrass_layer_names(model)
    if not layer_names:
        raise RuntimeError(
            "FactGraSS could not find any supported trainable nn.Linear layers."
        )

    allowed_prefixes = tuple(f"{name}." for name in layer_names)
    param_names = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and name.startswith(allowed_prefixes)
    ]
    total_params_cnt, trainable_params_cnt = configure_trainable_params(
        model,
        param_names,
    )
    print(
        f"FactGraSS-supported params enabled: "
        f"{trainable_params_cnt:,} / {total_params_cnt:,} "
        f"({100.0 * trainable_params_cnt / total_params_cnt:.4f}%)"
    )
    print(f"FactGraSS hooked linear layers: {len(layer_names)}")
    return layer_names


class FactGraSSMethod(AttributionMethod):
    """Attribution method following the upstream GraSS FactGraSS configuration."""

    def __init__(self):
        self.stage_wrapper = None
        self.params = None
        self.ds_fields = None
        self.device = None
        self.args = None
        self.attributor = None
        self.task = None
        self.inner_transformer = None
        self.layer_names = None

    def _set_trainable_params_for_layers(self, layer_names: List[str]) -> None:
        allowed_prefixes = tuple(f"{name}." for name in layer_names)
        param_names = [
            name
            for name, param in self.inner_transformer.named_parameters()
            if param.requires_grad and name.startswith(allowed_prefixes)
        ]
        total_params_cnt, trainable_params_cnt = configure_trainable_params(
            self.inner_transformer,
            param_names,
        )
        print(
            f"FactGraSS-supported params enabled: "
            f"{trainable_params_cnt:,} / {total_params_cnt:,} "
            f"({100.0 * trainable_params_cnt / total_params_cnt:.4f}%)"
        )
        print(f"FactGraSS hooked linear layers: {len(layer_names)}")

    def _build_attributor(self, layer_names: List[str]) -> None:
        from dattri.algorithm.block_projected_if.block_projected_if import (
            BlockProjectedIFAttributor,
        )

        cache_dir = self.args.factgrass_cache_dir
        if cache_dir is None and self.args.factgrass_offload == "disk":
            cache_dir = str(Path(self.args.results_folder) / "factgrass_cache")

        intermediate_dim = self.args.proj_dim * self.args.factgrass_blowup_factor
        sqrt_intermediate_dim = int(math.sqrt(intermediate_dim))
        if sqrt_intermediate_dim * sqrt_intermediate_dim != intermediate_dim:
            raise ValueError(
                "FactGraSS requires proj_dim * blowup_factor to be a perfect square. "
                f"Got proj_dim={self.args.proj_dim}, "
                f"blowup_factor={self.args.factgrass_blowup_factor}, "
                f"intermediate_dim={intermediate_dim}."
            )

        sparsifier_kwargs = {
            "device": self.device,
            "proj_dim": sqrt_intermediate_dim,
            "proj_max_batch_size": 64,
            "proj_type": "random_mask",
        }
        projector_kwargs = {
            "device": self.device,
            "proj_dim": self.args.proj_dim,
            "proj_max_batch_size": 64,
            "proj_type": "sjlt" if self.device == "cuda" else "normal",
        }

        # Build a one-off class that combines the MusicLM compatibility mixin
        # with BlockProjectedIFAttributor.  Using type() keeps _build_attributor
        # re-callable (e.g. after invalid-layer filtering) without a module-level
        # name clash.
        MusicLMFactGraSSAttributor = type(
            "_MusicLMFactGraSSAttributor",
            (_MusicLMBlockProjectedIFAttributor, BlockProjectedIFAttributor),
            {},
        )

        self.attributor = MusicLMFactGraSSAttributor(
            task=self.task,
            layer_names=layer_names,
            hessian=self.args.factgrass_hessian,
            damping=self.args.factgrass_damping,
            device=self.device,
            sparsifier_kwargs=sparsifier_kwargs,
            projector_kwargs=projector_kwargs,
            offload=self.args.factgrass_offload,
            cache_dir=cache_dir,
            chunk_size=self.args.factgrass_chunk_size,
            stage_wrapper=self.stage_wrapper,
            ds_fields=self.ds_fields,
        )
        self.layer_names = layer_names

    def _probe_invalid_layers(self, train_loader: DataLoader) -> List[Tuple[str, object]]:
        if self.attributor.compressors is None:
            self.attributor._setup_compressors(train_loader)

        batch = next(iter(train_loader))
        expected_batch_size = batch[0].shape[0]

        # Layers with None compressors can't be projected — mark invalid immediately
        # without a backward pass (avoids AttributeError: NoneType.forward).
        probeable_names = []
        probeable_compressors = []
        invalid: List[Tuple[str, object]] = []
        for name, comp in zip(self.layer_names, self.attributor.compressors):
            if comp is None:
                invalid.append((name, None))
            else:
                probeable_names.append(name)
                probeable_compressors.append(comp)

        if not probeable_names:
            return invalid

        hook_manager = HookManager(
            self.inner_transformer,
            probeable_names,
            device=self.device,
        )
        hook_manager.set_compressors(probeable_compressors)

        try:
            self.inner_transformer.zero_grad(set_to_none=True)
            loss = self.attributor._compute_batch_loss(batch)
            loss.backward()
            compressed = hook_manager.get_compressed_grads()
        finally:
            hook_manager.remove_hooks()
            self.inner_transformer.zero_grad(set_to_none=True)

        for idx, grad in enumerate(compressed):
            name = probeable_names[idx]
            if grad is None:
                invalid.append((name, None))
            elif grad.dim() != 2 or grad.shape[0] != expected_batch_size:
                invalid.append((name, tuple(grad.shape)))
        return invalid

    def setup(
        self,
        stage_wrapper: torch.nn.Module,
        params: List[torch.nn.Parameter],
        ds_fields: Tuple[str, ...],
        device: str,
        args,
    ) -> None:
        self.stage_wrapper = stage_wrapper
        self.params = params
        self.ds_fields = ds_fields
        self.device = device
        self.args = args

    def set_task_and_transformer(self, task, inner_transformer, loss_fn):
        """Set the dattri AttributionTask and construct the FactGraSS attributor."""
        del loss_fn

        self.task = task
        self.inner_transformer = inner_transformer

        layer_names = _configure_factgrass_params(inner_transformer)
        self.layer_names = layer_names
        self._build_attributor(layer_names)

    def _validate_single_batch_contract(self, train_loader: DataLoader) -> None:
        invalid = self._probe_invalid_layers(train_loader)
        if invalid:
            invalid_names = {name for name, _shape in invalid}
            preview = ", ".join(f"{name}:{shape}" for name, shape in invalid[:10])
            valid_layer_names = [
                name for name in self.layer_names
                if name not in invalid_names
            ]
            if not valid_layer_names:
                raise RuntimeError(
                    "FactGraSS could not find any valid per-sample linear layers after "
                    f"excluding invalid layers: {preview}"
                )
            print(
                "FactGraSS filtering out non-samplewise linear layers: "
                f"{preview}"
            )
            self._set_trainable_params_for_layers(valid_layer_names)
            self._build_attributor(valid_layer_names)

        validate_single_batch_contract(
            method_label="FactGraSS",
            attributor=self.attributor,
            inner_transformer=self.inner_transformer,
            layer_names=self.layer_names,
            train_loader=train_loader,
            device=self.device,
        )

    # ─── Streaming (compute-and-discard) helpers ─────────────────────────────

    def _compressor_seed(self) -> int:
        return self.args.seed ^ 0xFA47CA5E

    def _setup_compressors_seeded(self, sample_loader: DataLoader) -> None:
        """Initialise compressors under an isolated, deterministic RNG seed."""
        seed = self._compressor_seed()
        _cpu = torch.get_rng_state()
        _cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if self.attributor.compressors is None:
            self.attributor._setup_compressors(sample_loader)
        torch.set_rng_state(_cpu)
        if _cuda is not None:
            torch.cuda.set_rng_state_all(_cuda)

    def _save_streaming_artefacts(self, shared_root: Path) -> None:
        """Shard-0 only: persist valid layer names for subsequent phases.

        Compressors are NOT saved to disk because the SJLT/random_mask closures
        cannot be pickled.  Instead every phase re-derives them from the same
        deterministic seed (_compressor_seed), which guarantees bit-identical
        matrices as long as the same model architecture and seed are used.
        """
        with open(str(shared_root / "valid_layer_names.json"), "w") as f:
            json.dump(self.layer_names, f)
        print(f"[streaming] valid layer names saved → {shared_root}")

    def _load_streaming_artefacts(self, shared_root: Path) -> None:
        """Load valid layer names from shard-0; compressors are re-derived from seed."""
        ln_path = shared_root / "valid_layer_names.json"
        if ln_path.exists():
            with open(str(ln_path)) as f:
                valid = json.load(f)
            if valid != self.layer_names:
                print(f"[streaming] layer names updated: {len(self.layer_names)} → {len(valid)}")
                self.layer_names = valid

    # ─── Phase 1: per-shard Hessian accumulation ─────────────────────────────

    def hessian_shard(
        self,
        shard_loader: DataLoader,
        shard_id: int,
        num_shards: int,
    ) -> None:
        """Stream training samples for this shard; accumulate partial H; no gradient storage.

        Mathematical guarantee: every shard uses the same compressors (identical
        seeded RNG) so partial H matrices live in the same projected space and
        can be summed exactly in the merge step.
        """
        shared_root = Path(self.args.factgrass_cache_dir)
        shared_root.mkdir(parents=True, exist_ok=True)

        # Validate (may filter invalid layers); reset attributor with valid layers
        self._validate_single_batch_contract(shard_loader)
        self._build_attributor(self.layer_names)
        self._setup_compressors_seeded(shard_loader)

        if shard_id == 0:
            self._save_streaming_artefacts(shared_root)

        compressors = self.attributor.compressors
        hook_manager = HookManager(self.inner_transformer, self.layer_names, device=self.device)
        hook_manager.set_compressors(compressors)

        layer_dims: Optional[List[int]] = None
        hessian_partial: Optional[List[Optional[torch.Tensor]]] = None
        sample_counts: Optional[List[int]] = None

        for batch_idx, batch in enumerate(tqdm(shard_loader, desc=f"[H-shard {shard_id}/{num_shards}]")):
            self.inner_transformer.zero_grad()
            data_kwargs = {k: batch[i].to(self.device) for i, k in enumerate(self.ds_fields)}
            loss, _, _ = self.stage_wrapper(**data_kwargs, return_loss=True)
            loss.backward()

            with torch.no_grad():
                grads = hook_manager.get_compressed_grads()

                if layer_dims is None:
                    layer_dims, sample_counts, hessian_partial = [], [], []
                    for g in grads:
                        ld = (g.shape[1] if g is not None and g.numel() > 0 and g.dim() > 1
                              else (g.numel() if g is not None and g.numel() > 0 else 0))
                        layer_dims.append(ld)
                        sample_counts.append(0)
                        hessian_partial.append(
                            torch.zeros(ld, ld, device=self.device, dtype=torch.float32)
                            if ld > 0 else None
                        )

                for li, g in enumerate(grads):
                    if hessian_partial[li] is None or g is None or g.numel() == 0:
                        continue
                    gf = g.to(torch.float32).detach()
                    sample_counts[li] += gf.shape[0]
                    hessian_partial[li].addmm_(gf.t(), gf)

            # Coarse empty_cache: every 200 batches instead of every 10
            # (per-batch empty_cache stalls the GPU; not needed in steady state).
            if batch_idx % 200 == 199:
                torch.cuda.empty_cache()

        hook_manager.remove_hooks()

        # Save partial H in float16 to halve disk usage (4 GB instead of 8 GB per shard)
        out_path = shared_root / f"partial_H_{shard_id:03d}_of_{num_shards:03d}.pt"
        torch.save({
            "hessian_sums": [h.half().cpu() if h is not None else None
                             for h in hessian_partial],
            "sample_counts": sample_counts,
            "layer_dims": layer_dims,
        }, str(out_path))
        print(f"[H-shard {shard_id}] partial H saved → {out_path}")

    # ─── Phase 2: H^{-1} inversion + test gradients ──────────────────────────

    def precompute(self, test_loader: DataLoader, num_shards: int) -> None:
        """Merge all partial H files, compute H^{-1}, compute and save test gradients.

        Runs once after all hessian_shard jobs complete.
        """
        from dattri.algorithm.block_projected_if.core.utils import stable_inverse

        shared_root = Path(self.args.factgrass_cache_dir)

        # Load valid layer names from shard 0
        self._load_streaming_artefacts(shared_root)

        # ── Merge partial H files sequentially (peak disk: all shards on disk;
        #    peak memory: 2 × 8 GB — running accumulator + one loaded shard)
        print("[precompute] Merging partial Hessian files...")
        hessian_total: Optional[List[Optional[torch.Tensor]]] = None
        total_counts: Optional[List[int]] = None
        layer_dims: Optional[List[int]] = None

        for sid in tqdm(range(num_shards), desc="  H-merge"):
            path = shared_root / f"partial_H_{sid:03d}_of_{num_shards:03d}.pt"
            data = torch.load(str(path), map_location="cpu")
            if hessian_total is None:
                layer_dims = data["layer_dims"]
                hessian_total = [h.float() if h is not None else None
                                 for h in data["hessian_sums"]]
                total_counts = list(data["sample_counts"])
            else:
                for li, h in enumerate(data["hessian_sums"]):
                    if h is not None and hessian_total[li] is not None:
                        hessian_total[li] += h.float()
                        total_counts[li] += data["sample_counts"][li]

        # ── Compute H^{-1} per layer (normalise by GRAND TOTAL N, not per-shard N)
        print("[precompute] Computing H^{-1} (normalization by total N)...")
        preconditioners: List[Optional[torch.Tensor]] = []
        damping = self.args.factgrass_damping
        for li in tqdm(range(len(layer_dims)), desc="  inverting"):
            H, n = hessian_total[li], total_counts[li]
            if H is not None and n > 0:
                P = stable_inverse(H.to(self.device) / n, damping=damping)
                preconditioners.append(P.cpu())
            else:
                preconditioners.append(None)
            hessian_total[li] = None  # free memory immediately
            torch.cuda.empty_cache()

        precond_path = shared_root / "preconditioners.pt"
        torch.save({"preconditioners": preconditioners, "layer_dims": layer_dims},
                   str(precond_path))
        print(f"[precompute] H^{{-1}} saved → {precond_path}")

        # ── Compute test gradients (the ~93-minute fixed cost, computed once) ──
        print("[precompute] Computing test gradients...")
        # Re-derive compressors from seed — identical to all hessian_shard jobs.
        # test_loader provides a sample batch for shape inference; layer dims match train.
        self._build_attributor(self.layer_names)
        self._setup_compressors_seeded(test_loader)
        self.attributor.layer_dims = layer_dims
        self.attributor.total_proj_dim = sum(layer_dims)

        test_grads, _ = self.attributor._compute_test_gradients(test_loader)

        test_grads_path = shared_root / "test_gradients.pt"
        torch.save(test_grads.cpu(), str(test_grads_path))
        print(f"[precompute] Test gradients saved → {test_grads_path}  shape={test_grads.shape}")

        torch.save({"layer_dims": layer_dims, "num_shards": num_shards},
                   str(shared_root / "stream_metadata.pt"))

    # ─── Phase 3: per-shard score computation ────────────────────────────────

    def score_shard(
        self,
        shard_loader: DataLoader,
        shard_id: int,
        num_shards: int,
    ) -> None:
        """Recompute training gradients on-the-fly; compute score rows; discard immediately.

        Storage: only the (N_shard × N_test) score rows are written to disk.
        """
        shared_root = Path(self.args.factgrass_cache_dir)

        # Load valid layer names from shard 0
        self._load_streaming_artefacts(shared_root)

        precond_data = torch.load(str(shared_root / "preconditioners.pt"), map_location="cpu")
        preconditioners_cpu: List[Optional[torch.Tensor]] = precond_data["preconditioners"]
        layer_dims: List[int] = precond_data["layer_dims"]

        test_grads_raw: torch.Tensor = torch.load(
            str(shared_root / "test_gradients.pt"), map_location="cpu"
        )
        num_test = test_grads_raw.shape[0]
        print(f"[score_shard {shard_id}] test_grads={test_grads_raw.shape}, "
              f"{len(layer_dims)} layers, {num_test} test samples")

        # Pre-load preconditioners (~8 GB) and test_grads (~14 GB) to GPU once.
        # Eliminates per-batch CPU→GPU transfers (precond: 120 × 64 MB per batch).
        # test_grads always go to GPU; preconditioners only when VRAM is large enough
        # (A40 48 GB: both preloaded; A100 40 GB: precond on CPU, moved per-layer).
        vram_total_gb = torch.cuda.get_device_properties(self.device).total_memory / 1e9
        preload_precond = vram_total_gb >= 45.0
        print(f"[score_shard {shard_id}] GPU VRAM: {vram_total_gb:.0f} GB — "
              f"{'pre-loading preconditioners' if preload_precond else 'keeping preconditioners on CPU (per-layer transfer)'}", flush=True)

        preconditioners: List[Optional[torch.Tensor]] = [
            p.to(device=self.device, dtype=torch.float32) if (p is not None and preload_precond) else p
            for p in preconditioners_cpu
        ]
        del preconditioners_cpu

        test_grads_gpu: list[torch.Tensor | None] = []
        col = 0
        for li, ld in enumerate(layer_dims):
            if ld == 0 or preconditioners[li] is None:
                test_grads_gpu.append(None)
            else:
                test_grads_gpu.append(
                    test_grads_raw[:, col:col + ld].to(device=self.device, dtype=torch.float32)
                )
            col += ld
        del test_grads_raw
        torch.cuda.empty_cache()
        print(f"[score_shard {shard_id}] pre-loaded. "
              f"GPU mem: {torch.cuda.memory_allocated(self.device)/1e9:.1f} GB", flush=True)

        # Re-derive compressors from seed — guaranteed identical to hessian_shard and precompute.
        self._build_attributor(self.layer_names)
        self._setup_compressors_seeded(shard_loader)
        compressors = self.attributor.compressors

        # Set up hook manager
        hook_manager = HookManager(self.inner_transformer, self.layer_names, device=self.device)
        hook_manager.set_compressors(compressors)

        num_train_shard = len(shard_loader.dataset)
        scores = torch.zeros(num_train_shard, num_test, dtype=torch.float32)
        sample_idx = 0

        for batch in tqdm(shard_loader, desc=f"[score-shard {shard_id}/{num_shards}]"):
            self.inner_transformer.zero_grad()
            data_kwargs = {k: batch[i].to(self.device) for i, k in enumerate(self.ds_fields)}
            loss, _, _ = self.stage_wrapper(**data_kwargs, return_loss=True)
            loss.backward()
            del loss

            with torch.no_grad():
                grads = hook_manager.get_compressed_grads()
                bs = batch[0].shape[0]
                batch_score = torch.zeros(bs, num_test, device=self.device)

                for li, (g, ld) in enumerate(zip(grads, layer_dims)):
                    if ld == 0 or g is None or g.numel() == 0 or preconditioners[li] is None:
                        continue
                    test_l = test_grads_gpu[li]
                    if test_l is None:
                        continue

                    gf = g.to(torch.float32)
                    P = preconditioners[li] if preconditioners[li].device == gf.device \
                        else preconditioners[li].to(device=self.device, dtype=torch.float32)
                    ifvp = torch.matmul(P, gf.t()).t()              # (bs, ld)
                    # NOTE: equivalent to `ifvp @ test_l.t()` but transposed-out.
                    # On torch 2.10+cu128 / sm_80 (A100) + sm_90 (H100), the small-m large-n
                    # variant `cublasSgemm(m=bs, n=num_test, k=ld)` returns
                    # CUBLAS_STATUS_NOT_INITIALIZED (cuBLAS heuristic bug, OK on sm_86 / A40).
                    # Computing the transposed shape (m=num_test, n=bs) sidesteps the broken
                    # kernel selection; result is .t()'d back.
                    batch_score += torch.matmul(test_l, ifvp.t()).t()   # (bs, num_test)
                    del ifvp, gf

                scores[sample_idx:sample_idx + bs] = batch_score.cpu()
                sample_idx += bs

        hook_manager.remove_hooks()

        out_path = shared_root / f"scores_{shard_id:03d}_of_{num_shards:03d}.pt"
        torch.save(scores, str(out_path))
        print(f"[score_shard {shard_id}] scores saved → {out_path}  shape={scores.shape}")

    # ─── Phase 4: score merge (called via merge_attribute with streaming flag) ─

    def _streaming_merge_scores(self, num_shards: int) -> torch.Tensor:
        """Concatenate per-shard score rows into the final (N_train, N_test) matrix."""
        shared_root = Path(self.args.factgrass_cache_dir)
        rows = []
        for sid in range(num_shards):
            path = shared_root / f"scores_{sid:03d}_of_{num_shards:03d}.pt"
            rows.append(torch.load(str(path), map_location="cpu"))
        scores = torch.cat(rows, dim=0)
        print(f"[streaming_merge] final score matrix: {scores.shape}")
        return scores

    # ─── Existing disk-offload shard/merge (unchanged) ───────────────────────

    def cache_shard(
        self,
        shard_loader: DataLoader,
        shard_positions: List[int],
        results_folder: Path,
        shard_id: int,
        num_shards: int,
    ) -> None:
        """Compute and store compressed gradients for one training shard.

        Shard 0 saves valid_layer_names.json to the shared cache root.
        Compressors are re-derived from seed in every phase (not pickled).
        """
        shared_root = Path(self.args.factgrass_cache_dir)
        shard_cache_dir = shared_root / f"shard_{shard_id:03d}_of_{num_shards:03d}"
        shard_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[shard {shard_id}/{num_shards}] cache dir: {shard_cache_dir}")

        # Validate contract once per shard (fast: one backward pass).
        # May update self.layer_names to exclude invalid layers.
        print(f"[shard {shard_id}/{num_shards}] Validating FactGraSS contract...")
        self._validate_single_batch_contract(shard_loader)

        # Rebuild attributor pointing to the shard-specific cache dir,
        # using the (possibly filtered) self.layer_names from validation.
        saved_cache = self.args.factgrass_cache_dir
        self.args.factgrass_cache_dir = str(shard_cache_dir)
        self._build_attributor(self.layer_names)
        self.args.factgrass_cache_dir = saved_cache

        # Deterministic compressor seed — same constant XOR for all shards.
        # Using XOR keeps it independent of the data-sampling seed while
        # remaining fully reproducible given args.seed.
        compressor_seed = self.args.seed ^ 0xFA47CA5E

        self.attributor._cache_gradients_only(shard_loader, compressor_seed)

        # Shard 0 writes valid layer names used by the merge job.
        # Compressors are NOT pickled (closures aren't serialisable);
        # the merge job re-derives them from the same seed.
        if shard_id == 0:
            layer_names_path = shared_root / "valid_layer_names.json"
            with open(str(layer_names_path), "w") as f:
                json.dump(self.layer_names, f)
            print(f"[shard 0] Valid layer names saved → {layer_names_path}")

    def merge_attribute(
        self,
        num_train: int,
        results_folder: Path,
        num_shards: int,
        test_loader: DataLoader,
        shard_source_folder: Optional[Path] = None,
    ) -> torch.Tensor:
        """Dispatch to streaming or disk-offload merge depending on flag."""
        if getattr(self.args, 'factgrass_streaming', False):
            return self._streaming_merge_scores(num_shards)

        """Merge shard gradients: accumulate Hessian, compute IFVPs, return scores.

        Mathematical correctness guarantee
        -----------------------------------
        The Hessian H = (1/N) Σ_i g_i g_i^T is computed by accumulating raw outer-
        product sums from every shard and dividing by the grand total N at the end —
        not by averaging per-shard Hessians.  Combined with the identical compressors
        (loaded from shard 0's saved pkl), this produces bit-for-bit the same H as a
        single full-dataset run (up to float32 accumulation order, which is negligible).
        """
        from dattri.algorithm.block_projected_if.core.metadata import MetadataManager
        from dattri.algorithm.block_projected_if.core.utils import stable_inverse
        from dattri.algorithm.block_projected_if.offload import create_offload_manager

        shared_root = Path(self.args.factgrass_cache_dir)
        merged_cache_dir = shared_root / "merge_cache"
        merged_cache_dir.mkdir(parents=True, exist_ok=True)
        chunk_size = self.args.factgrass_chunk_size

        # ── Load reference artefacts written by shard 0 ──────────────────────
        layer_names_path = shared_root / "valid_layer_names.json"
        if layer_names_path.exists():
            with open(str(layer_names_path)) as f:
                valid_layer_names = json.load(f)
            if valid_layer_names != self.layer_names:
                print(f"[merge] Layer names updated: {len(self.layer_names)} → {len(valid_layer_names)} valid")
                self.layer_names = valid_layer_names

        # ── Layer dims from shard 0 metadata ─────────────────────────────────
        shard_0_dir = shared_root / f"shard_000_of_{num_shards:03d}"
        meta_0 = MetadataManager(str(shard_0_dir), self.layer_names)
        layer_dims = meta_0.layer_dims
        total_proj_dim = meta_0.total_proj_dim
        if layer_dims is None:
            raise RuntimeError(f"layer_dims not found in shard 0 metadata: {shard_0_dir}")
        print(f"[merge] {len(layer_dims)} layers, total_proj_dim={total_proj_dim}")

        # ── Inventory all shards ──────────────────────────────────────────────
        shard_metas = []
        total_samples_all = 0
        total_batches_all = 0
        for sid in range(num_shards):
            shard_dir = shared_root / f"shard_{sid:03d}_of_{num_shards:03d}"
            meta = MetadataManager(str(shard_dir), self.layer_names)
            shard_metas.append(meta)
            total_samples_all += meta.get_total_samples()
            total_batches_all += meta.get_total_batches()
        print(f"[merge] {num_shards} shards · {total_samples_all} samples · {total_batches_all} batches")

        # ── Pass 1: Accumulate raw outer-product sums for Hessian ─────────────
        # H_l = (1/N_l) Σ g_l_i g_l_i^T  where N_l = total samples seen per layer
        print("[merge] Pass 1 — accumulating Hessian outer products...")
        hessian_sums: List[Optional[torch.Tensor]] = []
        sample_counts = [0] * len(layer_dims)
        for ld in layer_dims:
            hessian_sums.append(
                torch.zeros(ld, ld, device=self.device, dtype=torch.float32) if ld > 0 else None
            )

        # Precompute layer column offsets once (avoids O(L) sum per chunk).
        layer_offsets = []
        _running = 0
        for _d in layer_dims:
            layer_offsets.append(_running)
            _running += _d

        for sid in range(num_shards):
            shard_dir = shared_root / f"shard_{sid:03d}_of_{num_shards:03d}"
            offload = create_offload_manager("disk", device=self.device,
                                             layer_names=self.layer_names,
                                             cache_dir=str(shard_dir), chunk_size=chunk_size)
            offload.layer_dims = layer_dims
            offload.total_proj_dim = total_proj_dim
            # Larger chunk batch reduces I/O / launch overhead; math identical.
            grad_dl = offload.create_gradient_dataloader("gradients", batch_size=32, pin_memory=True)
            for chunk_tensor, _bmap in tqdm(grad_dl, desc=f"  shard {sid} H-accum"):
                chunk = chunk_tensor.to(self.device)
                for li, ld in enumerate(layer_dims):
                    if hessian_sums[li] is None:
                        continue
                    sc = layer_offsets[li]
                    layer_data = chunk[:, sc:sc + ld].to(torch.float32).detach()
                    sample_counts[li] += layer_data.shape[0]
                    hessian_sums[li].addmm_(layer_data.t(), layer_data)

        # ── Compute H^{-1} per layer ──────────────────────────────────────────
        # Normalise by the TOTAL sample count across all shards (not per-shard).
        print("[merge] Computing preconditioners (H^{-1})...")
        preconditioners: List[Optional[torch.Tensor]] = []
        damping = self.args.factgrass_damping
        for li in tqdm(range(len(layer_dims)), desc="  inverting"):
            H = hessian_sums[li]
            n = sample_counts[li]
            if H is not None and n > 0:
                P = stable_inverse(H / n, damping=damping)
                preconditioners.append(P)
            else:
                preconditioners.append(None)
            hessian_sums[li] = None  # release GPU memory immediately
        del hessian_sums
        torch.cuda.empty_cache()

        # ── Pass 2: Apply H^{-1} to each shard's gradients → IFVP ───────────
        print("[merge] Pass 2 — computing IFVPs across all shards...")
        merged_offload = create_offload_manager("disk", device=self.device,
                                                layer_names=self.layer_names,
                                                cache_dir=str(merged_cache_dir), chunk_size=chunk_size)
        merged_offload.layer_dims = layer_dims
        merged_offload.total_proj_dim = total_proj_dim
        merged_offload.start_batch_range_processing(0, total_batches_all)

        merged_meta = MetadataManager(str(merged_cache_dir), self.layer_names)
        merged_meta.set_layer_dims(layer_dims)

        global_batch_idx = 0
        global_sample_idx = 0

        # `preconditioners[li]` lives on self.device after stable_inverse above.
        # We cache the dtype-cast version per layer to avoid redundant cast every batch.

        for sid in range(num_shards):
            shard_dir = shared_root / f"shard_{sid:03d}_of_{num_shards:03d}"
            shard_meta = MetadataManager(str(shard_dir), self.layer_names)
            shard_offload = create_offload_manager("disk", device=self.device,
                                                   layer_names=self.layer_names,
                                                   cache_dir=str(shard_dir), chunk_size=chunk_size)
            shard_offload.layer_dims = layer_dims
            shard_offload.total_proj_dim = total_proj_dim
            batch_to_sample = shard_meta.get_batch_to_sample_mapping()
            grad_dl = shard_offload.create_gradient_dataloader("gradients", batch_size=32, pin_memory=True)

            # Cache dtype-cast device preconditioners lazily on first batch.
            cast_preconds = [None] * len(layer_dims)

            for chunk_tensor, batch_mapping in tqdm(grad_dl, desc=f"  shard {sid} IFVP"):
                chunk = chunk_tensor.to(self.device)
                for local_batch_idx in sorted(batch_mapping.keys()):
                    if local_batch_idx not in batch_to_sample:
                        continue
                    sr, er = batch_mapping[local_batch_idx]
                    batch_grad = chunk[sr:er]

                    batch_ifvp = []
                    for li, ld in enumerate(layer_dims):
                        sc = layer_offsets[li]
                        if ld == 0 or preconditioners[li] is None:
                            batch_ifvp.append(torch.zeros(batch_grad.shape[0], ld, device="cpu"))
                            continue
                        layer_grad = batch_grad[:, sc:sc + ld]
                        if cast_preconds[li] is None or cast_preconds[li].dtype != layer_grad.dtype:
                            cast_preconds[li] = preconditioners[li].to(dtype=layer_grad.dtype)
                        # P symmetric SPD ⇒ (P @ g.T).T == g @ P
                        ifvp = (layer_grad @ cast_preconds[li]).cpu()
                        batch_ifvp.append(ifvp)

                    merged_offload.store_ifvp(global_batch_idx, batch_ifvp)
                    n_samples = er - sr
                    merged_meta.batch_info[global_batch_idx] = {
                        "sample_count": n_samples,
                        "start_idx": global_sample_idx,
                    }
                    global_sample_idx += n_samples
                    global_batch_idx += 1

        merged_offload.finish_batch_range_processing()
        merged_offload.wait_for_async_operations()
        merged_meta.total_samples = global_sample_idx
        merged_meta.save_metadata()
        print(f"[merge] IFVPs written: {global_batch_idx} batches, {global_sample_idx} samples")

        # ── Set up attributor for test-gradient + score computation ──────────
        print("[merge] Setting up attributor for score computation...")
        saved_cache = self.args.factgrass_cache_dir
        self.args.factgrass_cache_dir = str(merged_cache_dir)
        self._build_attributor(self.layer_names)
        self.args.factgrass_cache_dir = saved_cache

        # Re-derive compressors from seed — identical to cache_shard jobs (same seed, same model).
        # Closures cannot be pickled, so we never serialise them; seed guarantees consistency.
        self._setup_compressors_seeded(test_loader)
        print("[merge] Compressors re-derived from seed")

        # MetadataManager in the new attributor already loaded merged batch_metadata.json.
        # Sync layer_dims from it so the attributor knows projection dimensions.
        self.attributor._sync_layer_dims()

        # Prevent attribute() from re-triggering cache() — IFVP is already on disk.
        self.attributor.full_train_dataloader = True  # truthy sentinel; attribute() checks `is None`

        print("[merge] Computing attribution scores (test gradients + score matrix)...")
        return self.attributor.attribute(
            train_dataloader=test_loader,  # unused — IFVP is cached; satisfies API signature
            test_dataloader=test_loader,
        )

    def full(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ) -> torch.Tensor:
        print("Validating FactGraSS projected-gradient contract on first batch...")
        self._validate_single_batch_contract(train_loader)
        print("Caching FactGraSS training representations...")
        self.attributor.cache(train_loader)
        print("Computing FactGraSS attribution scores...")
        return self.attributor.attribute(train_loader, test_loader)

    def self_sanity(
        self,
        train_loader: DataLoader,
        num_train: int,
    ) -> Tuple[Dict, Dict]:
        print("Validating FactGraSS projected-gradient contract on first batch...")
        self._validate_single_batch_contract(train_loader)
        print("Caching FactGraSS training representations...")
        self.attributor.cache(train_loader)
        print("Computing FactGraSS self-sanity scores...")
        scores = self.attributor.attribute(train_loader, train_loader)
        scores_np = scores.detach().cpu().numpy()
        if scores_np.shape != (num_train, num_train):
            raise RuntimeError(
                "FactGraSS self-sanity requires raw score shape (num_train, num_train); "
                f"got {scores_np.shape}, expected {(num_train, num_train)}"
            )
        return compute_self_sanity_metrics(scores_np)


__all__ = ["FactGraSSMethod"]

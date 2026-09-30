# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import replace

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import (
    mxfp4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    TRITON_BACKENDS,
    Mxfp4MoeBackend,
    convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
    convert_weight_to_mxfp4_moe_kernel_format,
    make_mxfp4_moe_kernel,
    make_mxfp4_moe_quant_config,
    mxfp4_round_up_hidden_size_and_intermediate_size,
    select_deepseek_v4_mxfp4_moe_backend,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.utils import (
    is_weights_pre_processed,
    replace_parameter,
    set_weight_attrs,
)
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)


class Mxfp4Config(QuantizationConfig):
    """Canonical base config for MXFP4 quantization.

    Subclasses override get_name() and override_quantization_method() to
    register themselves as the handler for a specific checkpoint format.
    """

    def __init__(self, ignored_layers: list[str] | None = None):
        super().__init__()
        self.ignored_layers = ignored_layers

    @classmethod
    def from_config(cls, config):
        return cls()

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def _make_moe_method(self, moe: FusedMoEConfig) -> FusedMoEMethodBase:
        """MoE method for RoutedExperts. Subclasses override to pick a
        checkpoint-specific kernel family."""
        return Mxfp4MoEMethod(moe)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            if self.ignored_layers and is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            logger.debug_once(
                "MXFP4 linear layer is not implemented - falling back to "
                "UnquantizedLinearMethod.",
            )
            return UnquantizedLinearMethod()
        elif isinstance(layer, RoutedExperts):
            return self._make_moe_method(layer.moe_config)
        elif isinstance(layer, Attention):
            logger.debug_once(
                "MXFP4 attention layer is not implemented. "
                "Skipping quantization for this layer.",
            )
        return None


class GptOssMxfp4Config(Mxfp4Config):
    """MXFP4 config for GPT-OSS checkpoints.

    Checkpoints carry ``"quant_method": "mxfp4"`` in their JSON config.
    override_quantization_method() maps that to the canonical internal name
    so that the rest of the loading path uses "gpt_oss_mxfp4" consistently.
    """

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "gpt_oss_mxfp4"

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        # Match both "mxfp4" (original checkpoint value) and "gpt_oss_mxfp4"
        # (already normalized by verify_and_update_model_config) so that
        # explicit --quantization mxfp4 from the user doesn't cause a mismatch.
        if not (
            isinstance(hf_quant_cfg, dict)
            and hf_quant_cfg.get("quant_method") in ("mxfp4", "gpt_oss_mxfp4")
        ):
            return None
        # Require explicit confirmation that this is a GPT-OSS model.
        # Do NOT fall back to returning the override when hf_config is None,
        # as that would silently claim all mxfp4 checkpoints.
        model_type = getattr(hf_config, "model_type", None)
        if model_type != "gpt_oss":
            return None
        return "gpt_oss_mxfp4"

    def _make_moe_method(self, moe: FusedMoEConfig) -> FusedMoEMethodBase:
        return GptOssMxfp4MoEMethod(moe)


class GptOssMxfp4MoEMethod(FusedMoEMethodBase):
    """MXFP4 MoE quantization method."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.weight_dtype = "gpt_oss_mxfp4"
        self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(moe)

        self.max_capture_size = moe.max_capture_size

        self._cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
        self.moe_kernel: mk.FusedMoEKernel | None = None

        # Used for triton kernel precision configs
        self.w13_precision_config = None
        self.w2_precision_config = None

    @property
    def skip_forward_padding(self) -> bool:
        # SM100_FI_MXFP4_MXFP8_TRTLLM supports padding with mxfp8 quant
        # so can skip the padding in the forward before applying the moe method
        return self.mxfp4_backend == Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8

    # TODO(bnell): move to MK/expert_class?
    @property
    def has_unpadded_output(self) -> bool:
        return self.mxfp4_backend in [
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
        ]

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        hidden_size, intermediate_size_per_partition = super().maybe_roundup_sizes(
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            act_dtype=act_dtype,
            moe_parallel_config=moe_parallel_config,
        )
        return mxfp4_round_up_hidden_size_and_intermediate_size(
            self.mxfp4_backend, hidden_size, intermediate_size_per_partition
        )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.num_experts = num_experts
        weight_dtype = torch.uint8
        scale_dtype = torch.uint8
        mxfp4_block = 32

        layer.params_dtype = params_dtype
        layer.num_experts = num_experts
        self.intermediate_size = intermediate_size_per_partition
        self.hidden_size = hidden_size

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        w13_weight_scale.quant_method = "block"

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        w2_weight_scale.quant_method = "block"

        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    self.moe.w13_num_shards * intermediate_size_per_partition,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    hidden_size,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def _setup_kernel(
        self,
        layer: RoutedExperts,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
    ) -> None:
        num_experts = self.num_experts
        intermediate_size = self.intermediate_size
        hidden_size = self.hidden_size
        sf_block_size = 32

        # Shape assertions
        assert (
            w13.dim() == 3
            and w13.shape[0] == num_experts
            and w13.shape[1] == intermediate_size * self.moe.w13_num_shards
            and w13.shape[2] == hidden_size // 2
        )
        assert (
            w13_scale.dim() == 3
            and w13_scale.shape[0] == num_experts
            and w13_scale.shape[1] == intermediate_size * self.moe.w13_num_shards
            and w13_scale.shape[2] == hidden_size // sf_block_size
        )
        assert (
            w2.dim() == 3
            and w2.shape[0] == num_experts
            and w2.shape[1] == hidden_size
            and w2.shape[2] == intermediate_size // 2
        )
        assert (
            w2_scale.dim() == 3
            and w2_scale.shape[1] == hidden_size
            and w2_scale.shape[2] == intermediate_size // sf_block_size
        )
        if w13_bias is not None:
            assert (
                w13_bias.dim() == 2
                and w13_bias.shape[0] == num_experts
                and w13_bias.shape[1] == intermediate_size * self.moe.w13_num_shards
            )
        if w2_bias is not None:
            assert (
                w2_bias.dim() == 2
                and w2_bias.shape[0] == num_experts
                and w2_bias.shape[1] == hidden_size
            )

        # Convert weights to kernel format
        w13, w2, w13_scale, w2_scale, w13_bias, w2_bias = (
            convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
                mxfp4_backend=self.mxfp4_backend,
                layer=layer,
                w13_weight=w13,
                w2_weight=w2,
                w13_weight_scale=w13_scale,
                w2_weight_scale=w2_scale,
                w13_bias=w13_bias,
                w2_bias=w2_bias,
                _cache_permute_indices=self._cache_permute_indices,
            )
        )

        # For TRITON backends, weights are wrapped tensors from triton_kernels
        # that don't support .detach(). Manually assign parameters.
        if self.mxfp4_backend not in TRITON_BACKENDS:
            replace_parameter(layer, "w13_weight", w13)
            replace_parameter(layer, "w2_weight", w2)
            replace_parameter(layer, "w13_weight_scale", w13_scale)
            replace_parameter(layer, "w2_weight_scale", w2_scale)
        else:
            layer.w13_weight = w13
            layer.w2_weight = w2
            self.w13_precision_config = w13_scale
            self.w2_precision_config = w2_scale

        if w13_bias is not None and w2_bias is not None:
            replace_parameter(layer, "w13_bias", w13_bias)
            replace_parameter(layer, "w2_bias", w2_bias)

        # Build quant config
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

        # Build kernel (modular or monolithic)
        if self.moe_quant_config is not None and self.experts_cls is not None:
            self.moe_kernel = make_mxfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                mxfp4_backend=self.mxfp4_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
            )
            self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        w13_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)

        if self.mxfp4_backend == Mxfp4MoeBackend.NONE:
            return

        self._setup_kernel(layer, w13, w2, w13_scale, w2_scale, w13_bias, w2_bias)

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        w1_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)

        if self.mxfp4_backend in TRITON_BACKENDS:
            # TRITON backends free w13/w2_weight_scale after swizzling; the
            # swizzled scales live inside the precision configs instead.
            assert self.w13_precision_config is not None
            assert self.w2_precision_config is not None
            w1_scale = self.w13_precision_config
            w2_scale = self.w2_precision_config
        else:
            w1_scale = layer.w13_weight_scale
            w2_scale = layer.w2_weight_scale

        return make_mxfp4_moe_quant_config(
            mxfp4_backend=self.mxfp4_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            gemm1_alpha=1.702,
            gemm1_beta=1.0,
            swiglu_limit=7.0,
            layer=layer,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            router_logits=router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
        )


class Mxfp4MoEMethod(FusedMoEMethodBase):
    """MXFP4 MoE quantization method."""

    supports_pre_processed_weights = True

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)

        self.weight_dtype = "mxfp4"
        self.mxfp4_backend, self.experts_cls = select_deepseek_v4_mxfp4_moe_backend(moe)

        self.max_capture_size = moe.max_capture_size

        self._cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
        self.moe_kernel: mk.FusedMoEKernel | None = None

        # Used for triton kernel precision configs
        self.w13_precision_config = None
        self.w2_precision_config = None

    @property
    def supports_eplb(self) -> bool:
        return True

    @property
    def skip_forward_padding(self) -> bool:
        # SM100_FI_MXFP4_MXFP8_TRTLLM supports padding with mxfp8 quant
        # so can skip the padding in the forward before applying the moe method
        return self.mxfp4_backend == Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8

    # TODO(bnell): move to MK/expert_class?
    @property
    def has_unpadded_output(self) -> bool:
        return self.mxfp4_backend in [
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
        ]

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        hidden_size, intermediate_size_per_partition = super().maybe_roundup_sizes(
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            act_dtype=act_dtype,
            moe_parallel_config=moe_parallel_config,
        )
        return mxfp4_round_up_hidden_size_and_intermediate_size(
            self.mxfp4_backend,
            hidden_size,
            intermediate_size_per_partition,
            activation=self.moe.activation,
        )

    @staticmethod
    def _encode_mxfp4_weight_scale(loaded_weight: torch.Tensor) -> torch.Tensor:
        if loaded_weight.dtype == torch.uint8:
            return loaded_weight
        if loaded_weight.dtype == torch.float8_e8m0fnu:
            return loaded_weight.view(torch.uint8)
        if loaded_weight.is_floating_point():
            return loaded_weight.to(torch.float8_e8m0fnu).view(torch.uint8)
        return loaded_weight

    @staticmethod
    def get_scale_weight_loader(weight_loader):
        def mxfp4_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
            return_success: bool = False,
        ) -> bool | None:
            loaded_weight = Mxfp4MoEMethod._encode_mxfp4_weight_scale(loaded_weight)
            return weight_loader(
                param,
                loaded_weight,
                weight_name,
                shard_id,
                expert_id,
                return_success=return_success,
            )

        return mxfp4_weight_loader

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.logical_num_experts = num_experts
        self._nvme_paging_enabled = False
        self._nvme_store = None
        self._nvme_parts: dict[int, set[str]] = {}
        self._nvme_completed_experts: set[int] = set()
        self._nvme_store_dir = os.environ.get(
            "VLLM_DSV41_NVME_EXPERT_STORE_DIR", ""
        ).strip()

        allocated_experts = num_experts
        if self._nvme_store_dir:
            cfg = get_current_vllm_config_or_none()
            architecture = (
                cfg.model_config.architecture
                if cfg is not None and cfg.model_config is not None
                else None
            )
            if architecture != "DeepseekV41ForCausalLM":
                raise ValueError(
                    "VLLM_DSV41_NVME_EXPERT_STORE_DIR is restricted to "
                    f"DeepseekV41ForCausalLM, got {architecture!r}"
                )
            if self.moe.tp_size != 1 or self.moe.ep_size != 1:
                raise ValueError(
                    "DSV4.1 NVMe expert paging baseline requires TP=1 and EP=1"
                )
            if self.moe.has_bias:
                raise ValueError(
                    "DSV4.1 NVMe expert paging does not support expert biases"
                )
            if (
                self.mxfp4_backend
                != Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8
            ):
                raise ValueError(
                    "DGX Spark NVMe expert paging requires the FlashInfer "
                    "CUTLASS MXFP8xMXFP4 backend; start vLLM with "
                    "--moe-backend flashinfer_cutlass"
                )
            if self.experts_cls is None or self.experts_cls.__name__ != "FlashInferExperts":
                raise ValueError(
                    "DGX Spark NVMe paging requires FlashInferExperts, got "
                    f"{self.experts_cls}"
                )
            slots_text = os.environ.get(
                "VLLM_DSV41_NVME_EXPERT_CACHE_SLOTS"
            )
            if slots_text is None:
                raise ValueError(
                    "VLLM_DSV41_NVME_EXPERT_CACHE_SLOTS must be set explicitly "
                    "when NVMe paging is enabled. Start with 64 on a 128 GB "
                    "DGX Spark correctness run."
                )
            try:
                slots = int(slots_text)
            except ValueError as exc:
                raise ValueError(
                    "VLLM_DSV41_NVME_EXPERT_CACHE_SLOTS must be an integer"
                ) from exc
            if not self.moe.experts_per_token <= slots <= num_experts:
                raise ValueError(
                    "VLLM_DSV41_NVME_EXPERT_CACHE_SLOTS must be between "
                    f"top-k={self.moe.experts_per_token} and {num_experts}"
                )
            allocated_experts = slots
            self._nvme_paging_enabled = True

        self.num_experts = allocated_experts
        self._nvme_uva_slots = (
            self._nvme_paging_enabled
            and os.environ.get("VLLM_DSV41_NVME_UVA_SLOTS", "1") != "0"
        )
        if self._nvme_uva_slots and not is_uva_available():
            raise RuntimeError(
                "VLLM_DSV41_NVME_UVA_SLOTS=1 requires CUDA UVA host mapping"
            )
        slot_backing: dict[str, torch.Tensor] = {}
        slot_backing_owners: list[torch.Tensor] = []

        def make_slot_parameter(
            name: str,
            shape: tuple[int, ...],
            dtype: torch.dtype,
        ) -> torch.nn.Parameter:
            if not self._nvme_uva_slots:
                return torch.nn.Parameter(
                    torch.zeros(*shape, dtype=dtype),
                    requires_grad=False,
                )

            # O_DIRECT can target the UVA backing itself when every slot starts
            # on a filesystem block boundary. Over-allocate one pinned byte
            # buffer and carve a 4 KiB-aligned contiguous tensor view from it.
            numel = 1
            for dim in shape:
                numel *= dim
            element_size = torch.empty((), dtype=dtype).element_size()
            nbytes = numel * element_size
            owner = torch.zeros(
                nbytes + 4096,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            shift = (-owner.data_ptr()) % 4096
            raw = owner[shift : shift + nbytes]
            backing = raw.view(dtype).reshape(shape)
            assert backing.is_contiguous()
            assert backing.data_ptr() % 4096 == 0
            slot_backing_owners.append(owner)

            view = get_accelerator_view_from_cpu_tensor(backing)
            slot_backing[name] = backing
            return torch.nn.Parameter(view, requires_grad=False)

        weight_dtype = torch.uint8
        scale_dtype = torch.uint8
        mxfp4_block = 32

        layer.params_dtype = params_dtype
        layer.num_experts = num_experts
        self.intermediate_size = intermediate_size_per_partition
        self.hidden_size = hidden_size
        weight_loader = extra_weight_attrs.pop("weight_loader")
        scale_weight_loader = Mxfp4MoEMethod.get_scale_weight_loader(weight_loader)

        w13_weight = make_slot_parameter(
            "w13",
            (
                allocated_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size // 2,
            ),
            weight_dtype,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        set_weight_attrs(w13_weight, {"weight_loader": weight_loader})

        w13_weight_scale = make_slot_parameter(
            "w13_scale",
            (
                allocated_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size // mxfp4_block,
            ),
            scale_dtype,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w13_weight_scale, {"weight_loader": scale_weight_loader})
        w13_weight_scale.quant_method = "block"

        w2_weight = make_slot_parameter(
            "w2",
            (
                allocated_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
            ),
            weight_dtype,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        set_weight_attrs(w2_weight, {"weight_loader": weight_loader})

        w2_weight_scale = make_slot_parameter(
            "w2_scale",
            (
                allocated_experts,
                hidden_size,
                intermediate_size_per_partition // mxfp4_block,
            ),
            scale_dtype,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, {"weight_loader": scale_weight_loader})
        w2_weight_scale.quant_method = "block"

        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    allocated_experts,
                    self.moe.w13_num_shards * intermediate_size_per_partition,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
            set_weight_attrs(w13_bias, {"weight_loader": weight_loader})

            w2_bias = torch.nn.Parameter(
                torch.zeros(
                    allocated_experts,
                    hidden_size,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)
            set_weight_attrs(w2_bias, {"weight_loader": weight_loader})

        if self._nvme_paging_enabled:
            layer._dsv41_nvme_paging = True
            if self._nvme_uva_slots:
                layer._dsv41_nvme_cpu_backing = slot_backing
                # Keep the over-allocation that owns every aligned view alive.
                layer._dsv41_nvme_cpu_backing_owners = slot_backing_owners
            self._init_nvme_expert_store(layer)

    def _init_nvme_expert_store(self, layer: RoutedExperts) -> None:
        from vllm.model_executor.layers.fused_moe.expert_disk_store import (
            DiskExpertStore,
        )

        cfg = get_current_vllm_config_or_none()
        assert cfg is not None and cfg.model_config is not None
        safe_layer = layer.layer_name.replace("/", "_").replace(".", "_")
        path = os.path.join(self._nvme_store_dir, safe_layer + ".experts")
        direct_io = os.environ.get("VLLM_DSV41_NVME_DIRECT_IO", "1") != "0"
        specs = [
            (
                "w13",
                (
                    self.moe.w13_num_shards * self.intermediate_size,
                    self.hidden_size // 2,
                ),
                torch.uint8,
            ),
            (
                "w2",
                (self.hidden_size, self.intermediate_size // 2),
                torch.uint8,
            ),
            (
                "w13_scale",
                (
                    self.moe.w13_num_shards * self.intermediate_size,
                    self.hidden_size // 32,
                ),
                torch.uint8,
            ),
            (
                "w2_scale",
                (self.hidden_size, self.intermediate_size // 32),
                torch.uint8,
            ),
        ]
        self._nvme_store = DiskExpertStore.create_for_streaming(
            path,
            self.logical_num_experts,
            specs,
            identity={
                "layout": "deepseek-v41-flashinfer-cutlass-mxfp4-mxfp8-v1",
                "model": str(cfg.model_config.model),
                "revision": str(cfg.model_config.revision),
                "layer": layer.layer_name,
            },
            direct_io=direct_io,
        )

    def stream_expert_weight(
        self,
        layer: RoutedExperts,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> bool:
        if not self._nvme_paging_enabled:
            return False
        if "bias" in weight_name or "input_scale" in weight_name:
            return False
        if shard_id not in ("w1", "w2", "w3"):
            return False

        store = self._nvme_store
        assert store is not None
        if store.is_complete:
            return True
        if expert_id in self._nvme_completed_experts:
            raise RuntimeError(
                f"duplicate tensor arrived after expert {expert_id} was sealed"
            )

        is_scale = "scale" in weight_name
        field_name = (
            "w2_scale" if is_scale and shard_id == "w2" else
            "w13_scale" if is_scale else
            "w2" if shard_id == "w2" else
            "w13"
        )
        field = store.fields[field_name]
        src = loaded_weight.detach()
        if src.device.type != "cpu":
            src = src.cpu()
        src = src.contiguous().reshape(-1).view(torch.uint8)

        expected_nbytes = field.nbytes
        byte_offset = 0
        if shard_id in ("w1", "w3"):
            expected_nbytes //= self.moe.w13_num_shards
            if shard_id == "w3":
                byte_offset = expected_nbytes
        if src.numel() != expected_nbytes:
            raise ValueError(
                f"NVMe expert tensor size mismatch for {layer.layer_name} "
                f"expert={expert_id} {shard_id} scale={is_scale}: "
                f"checkpoint={src.numel()} store={expected_nbytes}"
            )

        store.write_field(
            expert_id,
            field_name,
            src,
            byte_offset=byte_offset,
        )
        part = f"{shard_id}:{'scale' if is_scale else 'weight'}"
        parts = self._nvme_parts.setdefault(expert_id, set())
        if part in parts:
            raise RuntimeError(
                f"duplicate expert component {part} for expert {expert_id}"
            )
        parts.add(part)
        expected = {
            "w1:weight", "w2:weight", "w3:weight",
            "w1:scale", "w2:scale", "w3:scale",
        }
        if parts == expected:
            self._convert_completed_nvme_expert(layer, expert_id)
            self._nvme_completed_experts.add(expert_id)
            del self._nvme_parts[expert_id]
        return True

    def _convert_completed_nvme_expert(
        self, layer: RoutedExperts, expert_id: int
    ) -> None:
        """Convert one raw checkpoint expert once, then persist runtime layout."""
        store = self._nvme_store
        assert store is not None
        row = torch.empty(store.record_stride, dtype=torch.uint8, pin_memory=True)
        store.read_working_record(expert_id, row)

        device = layer.w13_weight.device
        raw_w13 = store.field_view(row, "w13").unsqueeze(0).to(device)
        raw_w2 = store.field_view(row, "w2").unsqueeze(0).to(device)
        raw_w13_scale = store.field_view(row, "w13_scale").unsqueeze(0).to(device)
        raw_w2_scale = store.field_view(row, "w2_scale").unsqueeze(0).to(device)

        (
            runtime_w13,
            runtime_w2,
            runtime_w13_scale,
            runtime_w2_scale,
            _,
            _,
        ) = convert_weight_to_mxfp4_moe_kernel_format(
            mxfp4_backend=self.mxfp4_backend,
            layer=layer,
            w13_weight=raw_w13,
            w2_weight=raw_w2,
            w13_weight_scale=raw_w13_scale,
            w2_weight_scale=raw_w2_scale,
            _cache_permute_indices=self._cache_permute_indices,
            activation=self.moe.activation,
        )

        store.field_view(row, "w13").copy_(
            runtime_w13[0].detach().to("cpu").view(torch.uint8)
        )
        store.field_view(row, "w2").copy_(
            runtime_w2[0].detach().to("cpu").view(torch.uint8)
        )
        store.field_view(row, "w13_scale").copy_(
            runtime_w13_scale[0].detach().to("cpu").view(torch.uint8)
        )
        store.field_view(row, "w2_scale").copy_(
            runtime_w2_scale[0].detach().to("cpu").view(torch.uint8)
        )
        store.write_record(expert_id, row)

    def _load_seed_slots(self, layer: RoutedExperts) -> None:
        store = self._nvme_store
        assert store is not None and store.is_complete
        row = torch.empty(store.record_stride, dtype=torch.uint8, pin_memory=True)
        for slot in range(self.num_experts):
            store.read_record(slot, row)
            layer.w13_weight.data[slot].view(torch.uint8).copy_(
                store.field_view(row, "w13"), non_blocking=False
            )
            layer.w2_weight.data[slot].view(torch.uint8).copy_(
                store.field_view(row, "w2"), non_blocking=False
            )
            layer.w13_weight_scale.data[slot].view(torch.uint8).copy_(
                store.field_view(row, "w13_scale"), non_blocking=False
            )
            layer.w2_weight_scale.data[slot].view(torch.uint8).copy_(
                store.field_view(row, "w2_scale"), non_blocking=False
            )

    def _setup_kernel(
        self,
        layer: RoutedExperts,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
    ) -> None:
        num_experts = self.num_experts
        intermediate_size = self.intermediate_size
        hidden_size = self.hidden_size
        sf_block_size = 32

        # Shape assertions — skipped for SITU since its kernel handles native
        # (non-256-aligned) intermediate sizes without prior round-up.
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation

        if self.moe.activation != MoEActivation.SITU:
            assert (
                w13.dim() == 3
                and w13.shape[0] == num_experts
                and w13.shape[1] == intermediate_size * self.moe.w13_num_shards
                and w13.shape[2] == hidden_size // 2
            )
            assert (
                w13_scale.dim() == 3
                and w13_scale.shape[0] == num_experts
                and w13_scale.shape[1] == intermediate_size * self.moe.w13_num_shards
                and w13_scale.shape[2] == hidden_size // sf_block_size
            )
            assert (
                w2.dim() == 3
                and w2.shape[0] == num_experts
                and w2.shape[1] == hidden_size
                and w2.shape[2] == intermediate_size // 2
            )
            assert (
                w2_scale.dim() == 3
                and w2_scale.shape[1] == hidden_size
                and w2_scale.shape[2] == intermediate_size // sf_block_size
            )
            if w13_bias is not None:
                assert (
                    w13_bias.dim() == 2
                    and w13_bias.shape[0] == num_experts
                    and w13_bias.shape[1] == intermediate_size * self.moe.w13_num_shards
                )
            if w2_bias is not None:
                assert (
                    w2_bias.dim() == 2
                    and w2_bias.shape[0] == num_experts
                    and w2_bias.shape[1] == hidden_size
                )

        # Convert weights to kernel format
        w13, w2, w13_scale, w2_scale, w13_bias, w2_bias = (
            convert_weight_to_mxfp4_moe_kernel_format(
                mxfp4_backend=self.mxfp4_backend,
                layer=layer,
                w13_weight=w13,
                w2_weight=w2,
                w13_weight_scale=w13_scale,
                w2_weight_scale=w2_scale,
                w13_bias=w13_bias,
                w2_bias=w2_bias,
                _cache_permute_indices=self._cache_permute_indices,
                activation=self.moe.activation,
            )
        )

        # For TRITON backends, weights are wrapped tensors from triton_kernels
        # that don't support .detach(). Manually assign parameters.
        is_gfx1250 = False
        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_gfx1250

            is_gfx1250 = on_gfx1250()

        uses_triton_weight_format = self.mxfp4_backend in TRITON_BACKENDS or (
            self.mxfp4_backend == Mxfp4MoeBackend.AITER_MXFP4_BF16 and is_gfx1250
        )
        if not uses_triton_weight_format:
            replace_parameter(layer, "w13_weight", w13)
            replace_parameter(layer, "w2_weight", w2)
            replace_parameter(layer, "w13_weight_scale", w13_scale)
            replace_parameter(layer, "w2_weight_scale", w2_scale)
        else:
            layer.w13_weight = w13
            layer.w2_weight = w2
            self.w13_precision_config = w13_scale
            self.w2_precision_config = w2_scale

        if w13_bias is not None and w2_bias is not None:
            replace_parameter(layer, "w13_bias", w13_bias)
            replace_parameter(layer, "w2_bias", w2_bias)

        # Build quant config
        self._build_moe_kernel(layer)

    def _build_moe_kernel(self, layer: RoutedExperts) -> None:
        """Build the modular MoE kernel from the (already in-format) weights."""
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config is not None and self.experts_cls is not None:
            kernel_moe = self.moe
            routing_tables = layer._expert_routing_tables()
            if self._nvme_paging_enabled:
                # Router stays in the original logical expert space while the
                # FlashInfer expert kernel sees only the physical resident slots.
                kernel_moe = replace(
                    self.moe,
                    num_experts=self.num_experts,
                    num_local_experts=self.num_experts,
                    num_logical_experts=self.num_experts,
                )
                routing_tables = None
            self.moe_kernel = make_mxfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=kernel_moe,
                mxfp4_backend=self.mxfp4_backend,
                experts_cls=self.experts_cls,
                routing_tables=routing_tables,
            )
            self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def process_weights_after_loading(self, layer):
        if self.mxfp4_backend == Mxfp4MoeBackend.NONE:
            return

        if self._nvme_paging_enabled:
            store = self._nvme_store
            assert store is not None
            if self._nvme_parts:
                raise RuntimeError(
                    f"incomplete NVMe expert records remain: "
                    f"{len(self._nvme_parts)} experts"
                )
            store.finalize()

            # Disk records are already in FlashInfer CUTLASS
            # runtime layout, so do not call _setup_kernel (it would convert
            # them a second time).
            self._build_moe_kernel(layer)

            from vllm.model_executor.layers.fused_moe.mxfp4_direct_disk_provider import (
                FlashInferMxfp4DiskExpertProvider,
            )

            read_batch = int(
                os.environ.get("VLLM_DSV41_NVME_EXPERT_READ_BATCH", "6")
            )
            layer.expert_weight_provider = FlashInferMxfp4DiskExpertProvider(
                layer_id=int(layer.layer_name.split(".layers.", 1)[1].split(".", 1)[0]),
                global_num_experts=self.logical_num_experts,
                store=store,
                w13=layer.w13_weight,
                w2=layer.w2_weight,
                w13_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                cpu_backing=getattr(
                    layer, "_dsv41_nvme_cpu_backing", None
                ),
                read_batch=read_batch,
            )
            return

        if is_weights_pre_processed():
            if self.mxfp4_backend != Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8:
                raise RuntimeError(
                    "pre-processed weights require FLASHINFER_TRTLLM_MXFP4_MXFP8 "
                    f"moe backend, got {self.mxfp4_backend}"
                )
            self._build_moe_kernel(layer)
            return

        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        w13_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)

        self._setup_kernel(layer, w13, w2, w13_scale, w2_scale, w13_bias, w2_bias)

    def get_fused_moe_quant_config(
        self,
        layer: RoutedExperts,
    ) -> FusedMoEQuantConfig | None:
        w1_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)
        swiglu_limit = getattr(layer, "swiglu_limit", None)

        is_gfx1250 = False
        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_gfx1250

            is_gfx1250 = on_gfx1250()

        if self.mxfp4_backend in TRITON_BACKENDS or (
            self.mxfp4_backend == Mxfp4MoeBackend.AITER_MXFP4_BF16 and is_gfx1250
        ):
            # TRITON backends free w13/w2_weight_scale after swizzling; the
            # swizzled scales live inside the precision configs instead.
            assert self.w13_precision_config is not None
            assert self.w2_precision_config is not None
            w1_scale = self.w13_precision_config
            w2_scale = self.w2_precision_config
        else:
            w1_scale = layer.w13_weight_scale
            w2_scale = layer.w2_weight_scale

        if self.mxfp4_backend == Mxfp4MoeBackend.EMULATION:
            # Canonical ``mxfp4`` checkpoints are weight-only W4A16. The
            # generic EMULATION config is W4A4, so preserve BF16 activations
            # while the fallback dequantizes only the weights.
            return mxfp4_w4a16_moe_quant_config(
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_bias=w1_bias,
                w2_bias=w2_bias,
                gemm1_clamp_limit=swiglu_limit,
            )

        return make_mxfp4_moe_quant_config(
            mxfp4_backend=self.mxfp4_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            swiglu_limit=swiglu_limit,
            layer=layer,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        assert not self.is_monolithic
        assert self.moe_kernel is not None

        provider = layer.expert_weight_provider
        if provider is None:
            return self.moe_kernel.apply(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=layer.activation,
                global_num_experts=layer.global_num_experts,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                expert_map=layer.expert_map,
                shared_experts=shared_experts,
                shared_experts_input=shared_experts_input,
            )

        groups = provider.partition(topk_ids)
        if len(groups) == 1:
            required = groups[0]
            resident, missing = provider.split_resident(required)

            # Normal all-hit/all-miss cases stay on the single-kernel path.
            # Mixed residency can overlap missing NVMe reads with useful routed
            # compute, but only when shared experts are externally scheduled.
            split_miss_min_tokens = int(
                os.environ.get(
                    "VLLM_DSV41_NVME_SPLIT_MISS_MIN_TOKENS", "64"
                )
            )
            split_miss = (
                x.shape[0] >= split_miss_min_tokens
                and bool(resident)
                and bool(missing)
                and not (
                    shared_experts is not None
                    and self.moe_kernel.can_overlap_shared_experts
                )
            )
            if not split_miss:
                prepared = provider.prepare_keys(required)
                output = self.moe_kernel.apply(
                    hidden_states=x,
                    w1=prepared.w1,
                    w2=prepared.w2,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    activation=layer.activation,
                    global_num_experts=layer.global_num_experts,
                    apply_router_weight_on_input=layer.apply_router_weight_on_input,
                    expert_map=prepared.expert_map,
                    shared_experts=shared_experts,
                    shared_experts_input=shared_experts_input,
                )
                provider.mark_compute_submitted(required)
                return output

            # Start all missing reads before computing the already-resident
            # routed contribution.
            prefetched = provider.prefetch_keys(required)
            current = provider.current_weights()
            result = torch.zeros_like(x)

            resident_membership = torch.zeros(
                layer.global_num_experts,
                dtype=torch.bool,
                device=topk_ids.device,
            )
            resident_ids = torch.tensor(
                [key.expert_id for key in resident],
                dtype=torch.long,
                device=topk_ids.device,
            )
            resident_membership[resident_ids] = True
            topk_long = topk_ids.to(dtype=torch.long)
            valid = topk_long >= 0
            safe_ids = topk_long.clamp_min(0)
            resident_active = valid & resident_membership[safe_ids]
            resident_rows = torch.nonzero(
                resident_active.any(dim=-1), as_tuple=False
            ).squeeze(-1)

            if resident_rows.numel():
                selected_active = resident_active.index_select(0, resident_rows)
                selected_ids = topk_ids.index_select(0, resident_rows)
                selected_weights = topk_weights.index_select(0, resident_rows)
                fallback = resident[0].expert_id
                hit_ids = torch.where(
                    selected_active,
                    selected_ids,
                    torch.full_like(selected_ids, fallback),
                )
                hit_weights = torch.where(
                    selected_active,
                    selected_weights,
                    torch.zeros_like(selected_weights),
                )
                hit_partial = self.moe_kernel.apply(
                    hidden_states=x.index_select(0, resident_rows),
                    w1=current.w1,
                    w2=current.w2,
                    topk_weights=hit_weights,
                    topk_ids=hit_ids,
                    activation=layer.activation,
                    global_num_experts=layer.global_num_experts,
                    apply_router_weight_on_input=layer.apply_router_weight_on_input,
                    expert_map=current.expert_map,
                    shared_experts=None,
                    shared_experts_input=None,
                )
                if isinstance(hit_partial, UnfinalizedMoEOutput):
                    raise RuntimeError(
                        "NVMe split-miss path does not support deferred MoE finalize"
                    )
                result.index_add_(0, resident_rows, hit_partial)
                provider.mark_compute_submitted(resident)

            # At this point NVMe reads have run in parallel with the hit-side
            # kernel. Publishing waits only if the GPU is still consuming an
            # overwritten UVA slot.
            prepared = provider.activate_prefetch(prefetched)

            missing_membership = torch.zeros_like(resident_membership)
            missing_ids = torch.tensor(
                [key.expert_id for key in missing],
                dtype=torch.long,
                device=topk_ids.device,
            )
            missing_membership[missing_ids] = True
            missing_active = valid & missing_membership[safe_ids]
            missing_rows = torch.nonzero(
                missing_active.any(dim=-1), as_tuple=False
            ).squeeze(-1)
            if missing_rows.numel():
                selected_active = missing_active.index_select(0, missing_rows)
                selected_ids = topk_ids.index_select(0, missing_rows)
                selected_weights = topk_weights.index_select(0, missing_rows)
                fallback = missing[0].expert_id
                miss_ids = torch.where(
                    selected_active,
                    selected_ids,
                    torch.full_like(selected_ids, fallback),
                )
                miss_weights = torch.where(
                    selected_active,
                    selected_weights,
                    torch.zeros_like(selected_weights),
                )
                miss_partial = self.moe_kernel.apply(
                    hidden_states=x.index_select(0, missing_rows),
                    w1=prepared.w1,
                    w2=prepared.w2,
                    topk_weights=miss_weights,
                    topk_ids=miss_ids,
                    activation=layer.activation,
                    global_num_experts=layer.global_num_experts,
                    apply_router_weight_on_input=layer.apply_router_weight_on_input,
                    expert_map=prepared.expert_map,
                    shared_experts=None,
                    shared_experts_input=None,
                )
                if isinstance(miss_partial, UnfinalizedMoEOutput):
                    raise RuntimeError(
                        "NVMe split-miss path does not support deferred MoE finalize"
                    )
                result.index_add_(0, missing_rows, miss_partial)
                provider.mark_compute_submitted(missing)

            return result

        # Wide prefill can touch more unique experts than the resident cache.
        # On the TP1 paging path shared experts are orchestrated by MoERunner
        # outside this routed kernel (NO_OVERLAP or aux-stream overlap), so
        # every routed partition can execute only tokens that actually select
        # one of its resident experts.
        if (
            shared_experts is not None
            and self.moe_kernel.can_overlap_shared_experts
        ):
            raise RuntimeError(
                "partitioned NVMe prefill requires externally scheduled shared "
                "experts; internal shared-expert overlap is not supported"
            )

        result = torch.zeros_like(x)
        topk_ids_long = topk_ids.to(dtype=torch.long)
        valid_route = topk_ids_long >= 0
        safe_topk_ids = topk_ids_long.clamp_min(0)

        # Build the 384-entry expert->partition map once. Each route then needs
        # one indexed lookup instead of reconstructing a boolean membership
        # tensor for every cache-sized partition.
        group_of_expert_cpu = [-1] * layer.global_num_experts
        for group_index, group in enumerate(groups):
            for key in group:
                group_of_expert_cpu[key.expert_id] = group_index
        group_of_expert = torch.tensor(
            group_of_expert_cpu,
            dtype=torch.int16,
            device=topk_ids.device,
        )
        route_group = group_of_expert[safe_topk_ids]
        prefetched = None
        for group_index, group in enumerate(groups):
            prepared = (
                provider.prepare_keys(group)
                if prefetched is None
                else provider.activate_prefetch(prefetched)
            )
            next_prefetch = (
                provider.prefetch_keys(groups[group_index + 1])
                if group_index + 1 < len(groups)
                else None
            )

            active = valid_route & (route_group == group_index)
            token_rows = torch.nonzero(
                active.any(dim=-1), as_tuple=False
            ).squeeze(-1)

            if token_rows.numel() == 0:
                prefetched = next_prefetch
                continue

            fallback = group[0].expert_id
            run_x = x.index_select(0, token_rows)
            selected_active = active.index_select(0, token_rows)
            selected_ids = topk_ids.index_select(0, token_rows)
            selected_weights = topk_weights.index_select(0, token_rows)
            run_ids = torch.where(
                selected_active,
                selected_ids,
                torch.full_like(selected_ids, fallback),
            )
            run_weights = torch.where(
                selected_active,
                selected_weights,
                torch.zeros_like(selected_weights),
            )

            partial = self.moe_kernel.apply(
                hidden_states=run_x,
                w1=prepared.w1,
                w2=prepared.w2,
                topk_weights=run_weights,
                topk_ids=run_ids,
                activation=layer.activation,
                global_num_experts=layer.global_num_experts,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                expert_map=prepared.expert_map,
                # Shared experts were already scheduled by MoERunner. Passing
                # them here would couple shared compute to one routed partition.
                shared_experts=None,
                shared_experts_input=None,
            )
            provider.mark_compute_submitted(group)
            prefetched = next_prefetch
            if isinstance(partial, UnfinalizedMoEOutput):
                raise RuntimeError(
                    "partitioned NVMe prefill does not support deferred MoE finalize"
                )
            result.index_add_(0, token_rows, partial)

        return result

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            router_logits=router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
        )

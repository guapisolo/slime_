import re
import torch
from .qwen2 import convert_qwen2_to_hf


def convert_mimo_to_hf(args, name, param):
    """
    Convert MiMo model parameters from Megatron to HuggingFace format.

    MiMo extends Qwen2 with MTP (Multi-Token Prediction) layers.
    """

    if "mtp" in name:
        return convert_mimo_mtp_param(args, name, param)

    return convert_qwen2_to_hf(args, name, param)


def convert_mimo_mtp_param(args, name, param):
    """
    Convert MTP layer parameters from Megatron to HuggingFace format.

    MTP layers in MiMo contain:
    - LayerNorms (input, post_attention, token, hidden, final)
    - Input projection
    - Self attention (reuses Qwen2 attention structure)
    - MLP (reuses Qwen2 MLP structure)
    """

    mtp_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
    match = re.match(mtp_pattern, name)

    if not match:
        raise ValueError(f"Invalid MTP parameter name: {name}")

    layer_idx, component = match.groups()

    # Direct mappings for MTP-specific components
    direct_mappings = {
        "input_layernorm.weight": f"model.mtp_layers.{layer_idx}.input_layernorm.weight",
        "post_attention_layernorm.weight": f"model.mtp_layers.{layer_idx}.post_attention_layernorm.weight",
        "token_layernorm.weight": f"model.mtp_layers.{layer_idx}.token_layernorm.weight",
        "hidden_layernorm.weight": f"model.mtp_layers.{layer_idx}.hidden_layernorm.weight",
        "final_layernorm.weight": f"model.mtp_layers.{layer_idx}.final_layernorm.weight",
        "input_proj.weight": f"model.mtp_layers.{layer_idx}.input_proj.weight",
    }

    # Check direct mappings first
    if component in direct_mappings:
        return [(direct_mappings[component], param)]

    # Handle self-attention components
    if component.startswith("self_attention."):
        return convert_mimo_mtp_attention(args, layer_idx, component, param)

    # Handle MLP components
    if component.startswith("mlp."):
        return convert_mimo_mtp_mlp(args, layer_idx, component, param)

    raise ValueError(f"Unknown MTP component: {component} in {name}")


def convert_mimo_mtp_attention(args, layer_idx, component, param):
    """
    Convert MTP self-attention components, reusing Qwen2 attention logic.
    """
    # Remove "self_attention." prefix
    attn_component = component[15:]  # len("self_attention.") = 15

    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    # Map attention components
    if attn_component == "linear_proj.weight":
        return [(f"model.mtp_layers.{layer_idx}.self_attn.o_proj.weight", param)]

    elif attn_component == "linear_qkv.weight":
        # Split QKV weights following Qwen2 logic
        param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
        q_param, k_param, v_param = torch.split(param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1)
        q_param = q_param.reshape(-1, args.hidden_size)
        k_param = k_param.reshape(-1, args.hidden_size)
        v_param = v_param.reshape(-1, args.hidden_size)
        return [
            (f"model.mtp_layers.{layer_idx}.self_attn.q_proj.weight", q_param),
            (f"model.mtp_layers.{layer_idx}.self_attn.k_proj.weight", k_param),
            (f"model.mtp_layers.{layer_idx}.self_attn.v_proj.weight", v_param),
        ]

    elif attn_component == "linear_qkv.bias":
        # Split QKV biases if present
        param = param.view(args.num_query_groups, -1)
        q_bias, k_bias, v_bias = torch.split(
            param,
            split_size_or_sections=[value_num_per_group * head_dim, head_dim, head_dim],
            dim=1,
        )
        q_bias = q_bias.contiguous().flatten()
        k_bias = k_bias.contiguous().flatten()
        v_bias = v_bias.contiguous().flatten()
        return [
            (f"model.mtp_layers.{layer_idx}.self_attn.q_proj.bias", q_bias),
            (f"model.mtp_layers.{layer_idx}.self_attn.k_proj.bias", k_bias),
            (f"model.mtp_layers.{layer_idx}.self_attn.v_proj.bias", v_bias),
        ]

    # Q/K LayerNorm (if present)
    elif attn_component == "q_layernorm.weight":
        return [(f"model.mtp_layers.{layer_idx}.self_attn.q_norm.weight", param)]
    elif attn_component == "k_layernorm.weight":
        return [(f"model.mtp_layers.{layer_idx}.self_attn.k_norm.weight", param)]

    else:
        raise ValueError(f"Unknown MTP attention component: {attn_component}")


def convert_mimo_mtp_mlp(args, layer_idx, component, param):
    """
    Convert MTP MLP components, reusing Qwen2 MLP logic.
    """
    # Remove "mlp." prefix
    mlp_component = component[4:]  # len("mlp.") = 4

    if mlp_component == "linear_fc1.weight":
        # Split gate and up projections
        gate_weight, up_weight = param.chunk(2, dim=0)
        return [
            (f"model.mtp_layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
            (f"model.mtp_layers.{layer_idx}.mlp.up_proj.weight", up_weight),
        ]

    elif mlp_component == "linear_fc2.weight":
        return [(f"model.mtp_layers.{layer_idx}.mlp.down_proj.weight", param)]

    else:
        raise ValueError(f"Unknown MTP MLP component: {mlp_component}")

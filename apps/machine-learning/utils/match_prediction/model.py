import torch
import torch.nn as nn
import pickle
from sklearn.preprocessing import LabelEncoder
from utils.match_prediction import PATCH_MAPPING_PATH, CHAMPION_ID_ENCODER_PATH
from utils.match_prediction.column_definitions import (
    KNOWN_CATEGORICAL_COLUMNS_NAMES,
    COLUMNS,
    POSITIONS,
)
from utils.match_prediction.task_definitions import TASKS, TaskType
from utils.match_prediction.config import TrainingConfig
from typing import Optional, Tuple


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return self.dropout(down_proj)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors."""
    # q, k shape: [batch, heads, seq_len, head_dim]
    # cos, sin shape: [batch, seq_len, rope_dim]

    # Get the dimensions
    head_dim = q.shape[-1]

    # Ensure cos and sin have the right dimension (should match head_dim)
    if cos.shape[-1] != head_dim:
        # If dimensions don't match, we need to either truncate or pad
        if cos.shape[-1] > head_dim:
            # Truncate if rotary dim is larger
            cos = cos[..., :head_dim]
            sin = sin[..., :head_dim]
        else:
            # Pad with zeros if rotary dim is smaller - less ideal but prevents errors
            pad_dim = head_dim - cos.shape[-1]
            zeros_pad = torch.zeros_like(cos[..., :1]).expand(*cos.shape[:-1], pad_dim)
            cos = torch.cat([cos, zeros_pad], dim=-1)
            sin = torch.cat([sin, zeros_pad], dim=-1)

    # Reshape for broadcasting
    cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)  # [batch, 1, seq_len, head_dim]

    # Apply rotary embeddings
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=13, base=10000.0):
        super().__init__()
        self.dim = dim

        # Ensure dim is even for proper rotation
        if dim % 2 != 0:
            dim = dim - 1
            print(f"Warning: RoPE dimension should be even. Adjusted to {dim}")

        # Create RoPE embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, :, :])

    def forward(self, x, position_ids):
        """
        Args:
            x: (batch, seq_len, hidden_size)
            position_ids: (batch, seq_len)
        Returns:
            cos, sin for rotary embeddings
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Get cos and sin values for the positions we need
        cos = self.cos_cached[:, position_ids.reshape(-1)]
        sin = self.sin_cached[:, position_ids.reshape(-1)]

        # Reshape to match the expected shape
        cos = cos.reshape(batch_size, seq_len, self.dim)
        sin = sin.reshape(batch_size, seq_len, self.dim)

        return cos, sin


class Qwen3Attention(nn.Module):
    """Multi-headed attention adapted from Qwen3"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = dropout

        # Initialize projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # Add normalization for q and k like in Qwen3
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=rms_norm_eps)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        cos, sin = position_embeddings

        # Project to query, key, value
        query_states = self.q_proj(hidden_states).view(
            batch_size, seq_length, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states).view(
            batch_size, seq_length, self.num_kv_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_length, self.num_kv_heads, self.head_dim
        )

        # Apply normalization as in Qwen3
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Rearrange for attention computation
        query_states = query_states.transpose(
            1, 2
        )  # (batch, num_heads, seq_len, head_dim)
        key_states = key_states.transpose(
            1, 2
        )  # (batch, num_kv_heads, seq_len, head_dim)
        value_states = value_states.transpose(
            1, 2
        )  # (batch, num_kv_heads, seq_len, head_dim)

        # Apply RoPE embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # For GQA (Grouped Query Attention), duplicate the keys and values if necessary
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        )

        # Softmax and dropout
        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(hidden_states.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        )

        # Output projection and dropout
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output


class Qwen3Block(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Pre-normalization layers (Qwen3 uses pre-norm)
        self.attn_norm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp_norm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)

        # Attention block
        self.attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            rms_norm_eps=rms_norm_eps,
        )

        # MLP block
        self.mlp = Qwen3MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )

    def forward(self, hidden_states, position_embeddings):
        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3DownProjection(nn.Module):
    """Qwen3-style projection layer with gated MLP for dimension reduction"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        # Pre-normalization
        self.norm = Qwen3RMSNorm(input_dim, eps=rms_norm_eps)

        # Calculate intermediate size for MLP
        intermediate_size = min(input_dim * 2, 2048)

        # Use Qwen3MLP for the projection
        self.mlp = Qwen3MLP(
            hidden_size=input_dim,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )

        # Final projection to target dimension
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply normalization
        x = self.norm(x)

        # Apply MLP
        x = self.mlp(x)

        # Project to target dimension
        x = self.proj(x)

        # Apply dropout
        x = self.dropout(x)

        return x


class Model(nn.Module):
    def __init__(
        self,
        config: TrainingConfig,
        hidden_dims=None,
        dropout=None,
    ):
        super(Model, self).__init__()

        # Use config values for hidden_dims and dropout if not provided
        if hidden_dims is None:
            hidden_dims = config.hidden_dims
        if dropout is None:
            dropout = config.dropout

        # Load patch mapping and stats
        with open(PATCH_MAPPING_PATH, "rb") as f:
            self.patch_mapping = pickle.load(f)["mapping"]

        with open(CHAMPION_ID_ENCODER_PATH, "rb") as f:
            self.champion_id_encoder: LabelEncoder = pickle.load(f)["mapping"]

        self.num_champions = len(self.champion_id_encoder.classes_)
        # Number of unique patches
        self.num_patches = len(self.patch_mapping)

        # Define model dimensions - fixed to 128 for sequence processing
        self.hidden_dim = 128  # Fixed embedding dimension for all tokens

        # Champion embeddings
        self.champion_embed_dim = config.champion_embed_dim
        self.champion_patch_embed_dim = config.champion_patch_embed_dim

        # Create embeddings with their natural sizes (will be padded later)
        self.champion_embedding = nn.Embedding(
            self.num_champions, self.champion_embed_dim
        )
        self.patch_embedding = nn.Embedding(self.num_patches, config.patch_embed_dim)
        self.champion_patch_embedding = nn.Embedding(
            self.num_champions * self.num_patches, self.champion_patch_embed_dim
        )

        # Categorical feature embeddings
        self.queue_type_embedding = nn.Embedding(
            len(COLUMNS["queue_type"].possible_values), config.queue_type_embed_dim
        )
        self.elo_embedding = nn.Embedding(
            len(COLUMNS["elo"].possible_values), config.elo_embed_dim
        )

        # Padding projections to ensure all tokens have same hidden_dim (128)
        self.queue_type_padding = nn.Linear(
            config.queue_type_embed_dim, self.hidden_dim
        )
        self.patch_padding = nn.Linear(config.patch_embed_dim, self.hidden_dim)
        self.elo_padding = nn.Linear(config.elo_embed_dim, self.hidden_dim)

        # Champion token projections (to replace manual slicing)
        self.champion_projection = nn.Linear(
            config.champion_embed_dim + config.champion_patch_embed_dim, self.hidden_dim
        )

        # RoPE positional embeddings
        num_heads = config.num_attention_heads
        head_dim = self.hidden_dim // num_heads

        # Fixed sequence length of 13 (10 champions + queue_type + patch + elo)
        self.seq_length = 13

        self.rotary_emb = Qwen3RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max(
                self.seq_length, config.max_position_embeddings
            ),
        )

        # Create multiple transformer blocks (4)
        num_kv_heads = (
            config.num_kv_heads
            if config.num_kv_heads is not None
            else max(1, num_heads // 4)
        )
        intermediate_size = int(self.hidden_dim * config.intermediate_size_factor)

        self.num_layers = 4  # Number of transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                Qwen3Block(
                    hidden_size=self.hidden_dim,
                    intermediate_size=intermediate_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    dropout=dropout,
                    rms_norm_eps=config.rms_norm_eps,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Create projection layers with Qwen3MLP for dimension reduction
        self.projections = nn.ModuleList()
        prev_dim = self.hidden_dim

        for i, dim in enumerate(hidden_dims[1:], 1):
            self.projections.append(
                Qwen3DownProjection(
                    input_dim=prev_dim,
                    output_dim=dim,
                    dropout=dropout * 0.5 if i == len(hidden_dims) - 1 else dropout,
                    rms_norm_eps=config.rms_norm_eps,
                )
            )
            prev_dim = dim

        # Final normalization
        self.final_norm = Qwen3RMSNorm(hidden_dims[-1], eps=config.rms_norm_eps)

        # Output layers
        self.output_layers = nn.ModuleDict()
        for task_name, task_def in TASKS.items():
            if task_def.task_type in [
                TaskType.BINARY_CLASSIFICATION,
                TaskType.REGRESSION,
            ]:
                self.output_layers[task_name] = nn.Linear(hidden_dims[-1], 1)
            # No need for specific handling for bucketed tasks here,
            # as they are already covered by BINARY_CLASSIFICATION
            elif task_name.startswith("win_prediction_"):
                # Already handled by the BINARY_CLASSIFICATION check above
                pass
            else:
                raise ValueError(f"Unknown task type: {task_def.task_type}")

    def forward(self, features):
        batch_size = features["champion_ids"].size(0)
        device = features["champion_ids"].device

        # Process champions (first 10 tokens)
        champion_ids = features["champion_ids"]  # (batch_size, 10)
        patch_indices = features["patch"]  # (batch_size,)

        # Expand patch indices for each champion
        patch_indices_expanded = patch_indices.unsqueeze(1).expand(
            -1, 10
        )  # (batch_size, 10)

        # Compute champion+patch combined indices
        combined_indices = (
            champion_ids * self.num_patches + patch_indices_expanded
        )  # (batch_size, 10)

        # Get champion embeddings and champion-patch embeddings
        champion_embeds = self.champion_embedding(
            champion_ids
        )  # (batch_size, 10, champion_embed_dim)
        champion_patch_embeds = self.champion_patch_embedding(
            combined_indices
        )  # (batch_size, 10, champion_patch_embed_dim)

        # Initialize list to store all sequence tokens without normalization
        sequence_tokens = []

        # Process each champion token - FIXED: use proper projection instead of manual slicing
        for i in range(10):
            # Concatenate champion and champion-patch embeddings
            combined_embed = torch.cat(
                [champion_embeds[:, i], champion_patch_embeds[:, i]], dim=-1
            )
            # Project to hidden dimension using a linear layer
            champion_token = self.champion_projection(combined_embed)
            sequence_tokens.append(champion_token.unsqueeze(1))

        # Process queue_type (token 10)
        queue_type_embed = self.queue_type_embedding(
            features["queue_type"]
        )  # (batch_size, queue_type_embed_dim)
        queue_type_padded = self.queue_type_padding(
            queue_type_embed
        )  # (batch_size, hidden_dim)
        sequence_tokens.append(queue_type_padded.unsqueeze(1))

        # Process patch (token 11)
        patch_embed = self.patch_embedding(
            patch_indices
        )  # (batch_size, patch_embed_dim)
        patch_padded = self.patch_padding(patch_embed)  # (batch_size, hidden_dim)
        sequence_tokens.append(patch_padded.unsqueeze(1))

        # Process elo (token 12)
        elo_embed = self.elo_embedding(features["elo"])  # (batch_size, elo_embed_dim)
        elo_padded = self.elo_padding(elo_embed)  # (batch_size, hidden_dim)
        sequence_tokens.append(elo_padded.unsqueeze(1))

        # Concatenate all tokens to form the sequence
        sequence = torch.cat(sequence_tokens, dim=1)

        # Create position IDs for the sequence
        position_ids = (
            torch.arange(self.seq_length, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        # Generate RoPE position embeddings (cos, sin)
        position_embeddings = self.rotary_emb(sequence, position_ids)

        # Apply multiple transformer blocks sequentially
        for transformer_block in self.transformer_blocks:
            sequence = transformer_block(sequence, position_embeddings)

        # Mean pooling over sequence dimension
        x = sequence.mean(dim=1)  # (batch_size, hidden_dim)

        # Apply projections
        for projection in self.projections:
            x = projection(x)

        # Final normalization
        x = self.final_norm(x)

        # Generate outputs
        outputs = {}
        for task_name, output_layer in self.output_layers.items():
            outputs[task_name] = output_layer(x).squeeze(-1)

        return outputs

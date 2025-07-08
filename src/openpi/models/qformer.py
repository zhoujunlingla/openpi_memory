import flax.linen as nn
import jax.numpy as jnp
import jax
from typing import Callable


class _CrossAttentionBlock(nn.Module):
    """Single Q-Former block: LN → cross-attn (Q=query, K/V=image) → residual →
    LN → MLP → residual."""

    embed_dim: int
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        queries: jnp.ndarray,
        image_tokens: jnp.ndarray,
        *,
        deterministic: bool = True,
    ) -> jnp.ndarray:  # noqa: D401
        # queries.shape  = (B, Nq, D)
        # image_tokens = (B, Nt, D)
        x = nn.LayerNorm()(queries)
        k = nn.LayerNorm()(image_tokens)  # normalize keys for stability
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=queries.dtype,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
            name="cross_attn",
        )(x, k, k)
        x = queries + attn_out  # residual

        # Feed-forward
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.embed_dim * self.mlp_ratio, name="ffn_1")(y)
        y = nn.gelu(y, approximate=False)
        y = nn.Dense(self.embed_dim, name="ffn_2")(y)
        if self.dropout_rate > 0:
            y = nn.Dropout(self.dropout_rate)(y, deterministic=deterministic)
        x = x + y
        return x


class QFormer(nn.Module):
    """Minimal BLIP-style Q-Former (query transformer)."""

    embed_dim: int
    num_query_tokens: int = 32
    depth: int = 2
    num_heads: int = 8
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self, image_tokens: jnp.ndarray, *, deterministic: bool = True
    ) -> jnp.ndarray:  # noqa: D401
        # image_tokens: (B, Nt, D)
        batch_size, _, _ = image_tokens.shape

        # learnable query embeddings
        query_tokens = self.param(
            "query_tokens",
            nn.initializers.normal(stddev=0.02),
            (self.num_query_tokens, self.embed_dim),
        )
        # broadcast to batch dimension
        queries = jnp.broadcast_to(
            query_tokens[None, :, :],
            (batch_size, self.num_query_tokens, self.embed_dim),
        )

        x = queries
        for i in range(self.depth):
            x = _CrossAttentionBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                name=f"block_{i}",
            )(x, image_tokens, deterministic=deterministic)
        return x  # (B, num_query_tokens, D)

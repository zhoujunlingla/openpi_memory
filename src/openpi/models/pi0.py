import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override
from collections.abc import Sequence, Mapping
from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.models.openpi_memory import OpenPIMemory
# Q-Former replaced by an MLP summarizer; import no longer required.

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"],
    embedding_dim: int,
    min_period: float,
    max_period: float,
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    # Use LoRA variants by default so that fine-tuning only updates lightweight
    # adapter parameters instead of the full backbone (lower GPU memory).
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48

    # 只保留长期记忆参数
    long_memory_length: int = 256
    merge_length: int = 18  # 新增merge_length参数用于长期记忆合并

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(
        self, *, batch_size: int = 1
    ) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct(
            [batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32
        )
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct(
                    [batch_size, self.max_token_len], jnp.int32
                ),
                tokenized_prompt_mask=jax.ShapeDtypeStruct(
                    [batch_size, self.max_token_len], bool
                ),
            )
        action_spec = jax.ShapeDtypeStruct(
            [batch_size, self.action_horizon, self.action_dim], jnp.float32
        )

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0(_model.BaseModel):
    def __init__(self, config: Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        self.config = config
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(
            next(iter(config.fake_obs().images.values())), train=False, rngs=rngs
        )
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # ------------------------------------------------------------------
        # MLP summarizer to map variable-length memory tokens into a fixed
        # number (32) of visual tokens. We implement a *token-dimension*
        # MLP: (B, N, D) → (B, 32, D) by reshaping to (B·D, N) and applying
        # two Linear layers that operate on the token axis (N).
        # ------------------------------------------------------------------
        self.num_memory_tokens = 32
        # 最大保留帧数（合并后）= long_memory_length (默认 256)
        self.max_frames = config.long_memory_length

        # Token-wise feature projection (D 维)
        self.mem_mlp_in = nnx.Linear(paligemma_config.width,
                                     paligemma_config.width,
                                     rngs=rngs)
        self.mem_mlp_out = nnx.Linear(paligemma_config.width,
                                      paligemma_config.width,
                                      rngs=rngs)

        # Token-dimension MLP for compressing long-term memory to 32 visual tokens
        # First compression: current frame -> 32 visual tokens before storing in long-term memory
        # These layers operate on the token axis (N) to compress (B, N, D) -> (B, 32, D)
        self.first_compression_in = nnx.Linear(self.max_frames, 64, rngs=rngs)
        self.first_compression_out = nnx.Linear(64, self.num_memory_tokens, rngs=rngs)
        
        # Second compression: long-term memory -> 256 visual tokens before input to VLM
        # These layers operate on the token axis (N) to compress (B, N, D) -> (B, 256, D)
        self.second_compression_in = nnx.Linear(self.max_frames, 128, rngs=rngs)
        self.second_compression_out = nnx.Linear(128, 256, rngs=rngs)
       
        self.state_proj = nnx.Linear(
            config.action_dim, action_expert_config.width, rngs=rngs
        )
        self.action_in_proj = nnx.Linear(
            config.action_dim, action_expert_config.width, rngs=rngs
        )
        self.action_time_mlp_in = nnx.Linear(
            2 * action_expert_config.width, action_expert_config.width, rngs=rngs
        )
        self.action_time_mlp_out = nnx.Linear(
            action_expert_config.width, action_expert_config.width, rngs=rngs
        )
        self.action_out_proj = nnx.Linear(
            action_expert_config.width, config.action_dim, rngs=rngs
        )
        # ---------------------------------------------------------------
        # 只保留长期记忆
        self.memory = OpenPIMemory(
            long_len=config.long_memory_length,
            merge_len=config.merge_length,
            state_len=config.long_memory_length,
        )
        self.use_memory = True

    def compress_current_frame(self, image_tokens: jnp.ndarray) -> jnp.ndarray:
        """第一次压缩：将当前帧的SigLIP token压缩成32个视觉token，用于存储到长期记忆中。
        
        Args:
            image_tokens: (B, N, D) 当前帧的SigLIP编码token
            
        Returns:
            compressed_tokens: (B, 32, D) 压缩后的32个视觉token
        """
        bsz, n_tok, d_feat = image_tokens.shape
        
        # 1) Feature-wise MLP (先投影 D 维)
        projected = self.mem_mlp_out(
            nnx.swish(self.mem_mlp_in(image_tokens))
        )  # (B, N, D)
        
        # 2) Token-dimension MLP: (B, N, D) → (B, 32, D)
        mem_flat = projected.transpose(0, 2, 1).reshape(-1, n_tok)  # (B*D, N)
        mem_flat = nnx.swish(self.first_compression_in(mem_flat))  # (B*D, 64)
        mem_flat = self.first_compression_out(mem_flat)  # (B*D, 32)
        mem_seq = mem_flat.reshape(bsz, d_feat, self.num_memory_tokens)  # (B, D, 32)
        compressed_tokens = mem_seq.transpose(0, 2, 1)  # (B, 32, D)
        
        return compressed_tokens

    def compress_long_term_memory(self, mem_stack: jnp.ndarray) -> jnp.ndarray:
        """第二次压缩：将长期记忆压缩成256个视觉token，用于输入到VLM。
        
        Args:
            mem_stack: (B, N, D) 长期记忆中的token
            
        Returns:
            compressed_tokens: (B, 256, D) 压缩后的256个视觉token
        """
        bsz, n_tok, d_feat = mem_stack.shape
        
        # 1) Feature-wise MLP (先投影 D 维)
        projected = self.mem_mlp_out(
            nnx.swish(self.mem_mlp_in(mem_stack))
        )  # (B, N, D)
        
        # 2) Token-dimension MLP: (B, N, D) → (B, 256, D)
        mem_flat = projected.transpose(0, 2, 1).reshape(-1, n_tok)  # (B*D, N)
        mem_flat = nnx.swish(self.second_compression_in(mem_flat))  # (B*D, 128)
        mem_flat = self.second_compression_out(mem_flat)  # (B*D, 256)
        mem_seq = mem_flat.reshape(bsz, d_feat, 256)  # (B, D, 256)
        compressed_tokens = mem_seq.transpose(0, 2, 1)  # (B, 256, D)
        
        return compressed_tokens

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[
        at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        image_tokens_list = []
        image_mask_list = []
        ar_mask_list = []
        for name in obs.images:
            img_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            image_tokens_list.append(img_tokens)
            image_mask_list.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=img_tokens.shape[1],
                )
            )
            ar_mask_list += [False] * img_tokens.shape[1]
        if image_tokens_list:
            image_tokens = jnp.concatenate(image_tokens_list, axis=1)  # [b, s_img, emb]
            image_mask = jnp.concatenate(image_mask_list, axis=1)      # [b, s_img]
            ar_mask_img = ar_mask_list                                 # [s_img]
            # 只拼接长期记忆的视觉特征
            if self.use_memory and self.memory is not None:
                # ------------------------------------------------------
                # 1) 取出批量长期记忆 (B, N, D)
                # ------------------------------------------------------
                mem_stack = self.memory.get_batched_memory(
                    image_tokens.shape[0], image_tokens.shape[2]
                )  # (B, N, D)

                if mem_stack.shape[1] > 0:
                    # 2) pad / truncate 到固定帧数 self.max_frames
                    if mem_stack.shape[1] < self.max_frames:
                        pad_len = self.max_frames - mem_stack.shape[1]
                        pad = jnp.zeros(
                            (mem_stack.shape[0], pad_len, mem_stack.shape[2]),
                            dtype=mem_stack.dtype,
                        )
                        mem_stack = jnp.concatenate([mem_stack, pad], axis=1)
                    elif mem_stack.shape[1] > self.max_frames:
                        mem_stack = mem_stack[:, : self.max_frames, :]

                    # 第二次压缩：长期记忆 -> 256个视觉token
                    q_tokens = self.compress_long_term_memory(mem_stack)
                else:
                    q_tokens = jnp.zeros(
                        (
                            image_tokens.shape[0],
                            256,  # 第二次压缩后的token数量
                            image_tokens.shape[2],
                        ),
                        dtype=image_tokens.dtype,
                    )
                # Replace original token concatenation with [his]/[cur] markers
                # --------------------------------------------------------------
                # Insert special history ([his]) & current ([cur]) tokens
                # --------------------------------------------------------------
                bsz, _, d_feat = image_tokens.shape
                his_tok = jnp.full((bsz, 1, d_feat), -1.0, dtype=image_tokens.dtype)
                cur_tok = jnp.full((bsz, 1, d_feat), 1.0, dtype=image_tokens.dtype)

                tokens.append(his_tok)
                tokens.append(q_tokens)
                tokens.append(cur_tok)
                tokens.append(image_tokens)

                # masks
                his_mask = jnp.ones((bsz, 1), dtype=jnp.bool_)
                q_mask = jnp.ones((q_tokens.shape[0], q_tokens.shape[1]), dtype=jnp.bool_)
                cur_mask = jnp.ones((bsz, 1), dtype=jnp.bool_)
                input_mask.append(his_mask)
                input_mask.append(q_mask)
                input_mask.append(cur_mask)
                input_mask.append(image_mask)

                # ar_mask: non-autoregressive for [his], memory, [cur]
                ar_mask += [False]                 # [his]
                ar_mask += [False] * q_tokens.shape[1]
                ar_mask += [False]                 # [cur]
                ar_mask += ar_mask_img
                # end memory branch modifications
            else:
                # Insert [cur] token before current frame tokens when no long-term memory is used
                bsz, _, d_feat = image_tokens.shape
                cur_tok = jnp.full((bsz, 1, d_feat), 1.0, dtype=image_tokens.dtype)

                tokens.append(cur_tok)
                tokens.append(image_tokens)

                cur_mask = jnp.ones((bsz, 1), dtype=jnp.bool_)
                input_mask.append(cur_mask)
                input_mask.append(image_mask)

                ar_mask += [False]                 # [cur]
                ar_mask += ar_mask_img
                # end no-memory branch modifications
        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]
    ]:
        input_mask = []
        ar_mask = []
        tokens = []

        # Only use *current* robot state as a single token (no memory).
        state_token = self.state_proj(obs.state)[:, None, :]  # (B,1,emb)
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        # image/language inputs do not attend to state or actions
        ar_mask += [True]

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
        )
        # mix timestep + action information using an MLP
        action_tokens = self.action_in_proj(noisy_actions)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        collect_attn: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train
        )

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
            observation, x_t, time
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        # ------------------------------------------------------------------
        # Forward pass through the language model. When *collect_attn* is
        # true we also request the mutable "stats" collection so that we can
        # compute attention-based metrics between current visual tokens and
        # the short-/long-term memory tokens.
        # ------------------------------------------------------------------
        # Call LLM and gracefully handle whether stats are returned.
        if collect_attn:
            ret = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                mutable=["stats"],  # may be ignored by implementation
            )
        else:
            ret = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
            )

        # Unpack variable-length return (common patterns: ((out1,out2), kv_cache) or ((out1,out2), kv_cache, mut))
        if not (isinstance(ret, tuple) and len(ret) >= 1):
            raise ValueError("Unexpected return signature from PaliGemma.llm")

        (prefix_out, suffix_out) = ret[0]

        # Heuristically identify kv_cache and mut.  The last element often
        # contains mutable collections if it is a Mapping with "stats" key.
        kv_cache = None
        mut = None
        for elem in ret[1:]:
            if mut is None and isinstance(elem, dict | Mapping):
                # Potential mutable collection
                mut = elem
            elif kv_cache is None:
                kv_cache = elem

        # `kv_cache` can still be None during training (not used here).

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # ------------------------------------------------------------------
        # Compute loss: mean-squared error between predicted and target noise.
        # Attention-based diagnostic metrics have been removed; this function
        # now focuses purely on diffusion loss.
        # ------------------------------------------------------------------

        loss_vals = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        # ------------------------------------------------------------------
        # Update visual memory during training so that short/long memories are
        # non-empty and attention metrics become meaningful.
        # ------------------------------------------------------------------
        if self.use_memory and train:
            image_tokens_list = []
            for name in observation.images:
                img_tokens, _ = self.PaliGemma.img(observation.images[name], train=False)
                image_tokens_list.append(img_tokens)
            if image_tokens_list:
                all_tokens = jnp.concatenate(image_tokens_list, axis=1)  # (B, s_img_total, D)
                # 第一次压缩：当前帧 -> 32个视觉token
                compressed_tokens = self.compress_current_frame(all_tokens)
                for b in range(int(compressed_tokens.shape[0])):
                    self.memory.update_image_memory(b, compressed_tokens[b])

        return loss_vals

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(
            rng, (batch_size, self.action_horizon, self.action_dim)
        )

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate(
                [prefix_attn_mask, suffix_attn_mask], axis=-1
            )
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1)
                - 1
            )

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))

        # 在推理完成后，更新长期视觉记忆，以便下一次推理使用。
        if self.use_memory:
            image_tokens_list = []
            for name in observation.images:
                img_tokens, _ = self.PaliGemma.img(
                    observation.images[name], train=False
                )
                image_tokens_list.append(img_tokens)
            if image_tokens_list:
                all_image_tokens = jnp.concatenate(image_tokens_list, axis=1)  # (B, s_img_total, D)
                # 第一次压缩：当前帧 -> 32个视觉token
                compressed_tokens = self.compress_current_frame(all_image_tokens)
                for b in range(int(compressed_tokens.shape[0])):
                    self.memory.update_image_memory(b, compressed_tokens[b])

        return x_0

    # ------------------------------------------------------------------
    # Forward (predict noise)
    # ------------------------------------------------------------------
    @at.typecheck
    def __call__(
        self,
        observation: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah ad"]:
        """Forward pass used for diffusion loss.

        Args:
            observation: Pre-processed observation batch.
            noisy_actions: The noised action sequence (same shape as target actions).
            timestep: Diffusion timestep in \[0,1\] for each batch element.
            train: Whether in training mode (passed to dropout etc.).

        Returns:
            Predicted noise with same shape as *actions* (B, action_horizon, action_dim).
        """

        # 1. Prefix pass (images, prompt, etc.) ---------------------------------
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1  # (b, p_len)

        # Feed only to first expert (PaliGemma); second expert gets None
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
            deterministic=not train,
        )

        # 2. Suffix pass (state, actions, timestep) -----------------------------
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
            observation, noisy_actions, timestep
        )

        # how suffix tokens attend among themselves
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        # how suffix tokens attend to prefix tokens (all allowed)
        prefix_attn_mask_to_suffix = einops.repeat(
            prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
        )
        full_attn_mask = jnp.concatenate(
            [prefix_attn_mask_to_suffix, suffix_attn_mask], axis=-1
        )  # shape (b, s_len, p_len + s_len)

        # positions for suffix tokens start after prefix length
        suffix_positions = (
            jnp.sum(prefix_mask, axis=-1)[:, None]
            + jnp.cumsum(suffix_mask, axis=-1)
            - 1
        )

        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=suffix_positions,
            kv_cache=kv_cache,
            deterministic=not train,
        )

        # We only care about the last action_horizon tokens (they correspond to actions)
        predicted_noise = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return predicted_noise

    def init(self):
        """Convenience method for initializing all parameters, necessary due to the quirks of linen."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],           # embedded
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),           # positions
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),  # mask
            kv_cache=None,
            deterministic=True,
            # collect_attn=False,  # attention logging disabled
        )

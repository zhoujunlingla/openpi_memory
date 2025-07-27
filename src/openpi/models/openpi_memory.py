import jax.numpy as jnp
from typing import List


class OpenPIMemory:
    """Batch-aware长期视觉记忆。

    每个 batch 样本拥有独立的历史帧列表，结构为

        long_image_memory: List[List[jnp.ndarray]]

    外层索引 = batch index，内层 list 存不同时间帧的视觉 token（shape 通常为
    (s_img_total, D)）。合并逻辑沿用原来按相邻帧最大相似度合并的启发式。"""

    def __init__(self, long_len: int, merge_len: int, state_len: int):
        # long_image_memory[b] -> List[jnp.ndarray] (frames) for sample *b*
        self.long_image_memory: List[List[jnp.ndarray]] = []
        self.long_len = long_len
        self.merge_len = merge_len
        # state 相关保持兼容但不再使用
        self.state_len = state_len
        self._dummy_state_memory: List[jnp.ndarray] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_batch(self, batch_idx: int):  # noqa: D401, ANN001
        """确保 long_image_memory 至少能索引到 batch_idx。"""
        while len(self.long_image_memory) <= batch_idx:
            self.long_image_memory.append([])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_image_memory(self, batch_idx: int, image_features: jnp.ndarray):
        """向第 ``batch_idx`` 个样本的长期记忆追加一帧视觉特征。"""
        self._ensure_batch(batch_idx)
        self.long_image_memory[batch_idx].append(image_features)
        self._merge_long_memory(batch_idx)

    def _merge_long_memory(self, batch_idx: int):
        """对指定 batch 的长期记忆做相邻帧合并。
        
        当记忆长度超过 long_len 时触发合并，重复合并直到记忆长度 ≤ merge_len。
        """
        mem_list = self.long_image_memory[batch_idx]
        # 当记忆长度超过 long_len 时触发合并
        if len(mem_list) > self.long_len:
            while len(mem_list) > self.merge_len:
                max_sim = None
                max_idx = None
                for i in range(len(mem_list) - 1):
                    a = mem_list[i]
                    b = mem_list[i + 1]
                    sim = jnp.mean(jnp.dot(a.flatten(), b.flatten()))
                    if (max_sim is None) or (sim > max_sim):
                        max_sim = sim
                        max_idx = i
                if max_idx is not None:
                    merged = (mem_list[max_idx] + mem_list[max_idx + 1]) / 2
                    mem_list[max_idx] = merged
                    del mem_list[max_idx + 1]
                else:
                    break
            # 写回（必要时）
            self.long_image_memory[batch_idx] = mem_list

    def get_batched_memory(self, batch_size: int, feature_dim: int) -> jnp.ndarray:
        """返回 shape = (batch_size, N, feature_dim) 的批量记忆。

        N 为该批次中 *最大* 记忆长度；不足者用 0 补齐。"""
        batched = []
        max_len = 0
        for b in range(batch_size):
            if b < len(self.long_image_memory):
                mem_list = self.long_image_memory[b]
                if mem_list:
                    arr = jnp.stack(mem_list, axis=0)  # (n_b, D)
                else:
                    arr = jnp.zeros((0, feature_dim))
            else:
                arr = jnp.zeros((0, feature_dim))
            batched.append(arr)
            max_len = max(max_len, arr.shape[0])

        # pad 到统一长度
        padded = []
        for arr in batched:
            if arr.shape[0] < max_len:
                pad = jnp.zeros((max_len - arr.shape[0], feature_dim), dtype=arr.dtype)
                arr = jnp.concatenate([arr, pad], axis=0)
            padded.append(arr)

        return jnp.stack(padded, axis=0)  # (B, N, D)

    def update_state_memory(self, state: jnp.ndarray):
        """Deprecated: kept for API compatibility.

        State information is now treated as *per-frame* and is **not stored**
        in long-term memory.  This method therefore performs no operation.
        """
        pass

    def get_concat_image_features(
        self, batch_idx: int, current_image_features: jnp.ndarray
    ) -> jnp.ndarray:
        """获取指定批次的历史记忆与当前帧特征的拼接。
        
        Args:
            batch_idx: 批次索引
            current_image_features: 当前帧的图像特征
            
        Returns:
            拼接后的特征数组
        """
        # 确保批次存在
        self._ensure_batch(batch_idx)
        
        # 获取当前批次的历史记忆
        batch_memory = self.long_image_memory[batch_idx]
        
        # 将历史记忆与当前帧特征拼接
        all_features = batch_memory + [current_image_features]
        
        if all_features:
            return jnp.concatenate(all_features, axis=0)
        else:
            return current_image_features

    def get_concat_states(self, current_state: jnp.ndarray) -> jnp.ndarray:
        """Return the current state only (history no longer stored)."""
        return current_state

    def reset(self):  # noqa: D401
        self.long_image_memory.clear()
        self._dummy_state_memory.clear()

    # ---------------------------------------------------------------------
    # Equality & hashing helpers
    # ---------------------------------------------------------------------
    # During training initialization we compare PyTrees for structural
    # equality. Different `
    def __eq__(self, other):  # noqa: D401, ANN001
        return isinstance(other, OpenPIMemory)

    def __hash__(self):  # noqa: D401
        # All instances are considered equal, so use a constant hash value.
        return 0
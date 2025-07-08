import jax.numpy as jnp
from typing import List


class OpenPIMemory:
    def __init__(self, short_len: int, long_len: int, merge_len: int, state_len: int):
        self.short_image_memory: List[jnp.ndarray] = []
        self.long_image_memory: List[jnp.ndarray] = []
        self.short_len = short_len
        self.long_len = long_len
        self.merge_len = merge_len
        self.state_len = state_len  # record state memory capacity
        self.state_memory: List[jnp.ndarray] = []  # 保留所有历史 state

    def update_image_memory(self, image_features: jnp.ndarray):
        self.short_image_memory.append(image_features)
        # 只保留 short_len 个，超出的先弹出（如 MovieChat）
        while len(self.short_image_memory) > self.short_len:
            self.short_image_memory.pop(0)
        # Memory consolidation: merge similar short-term memories and
        # then move them into long-term memory.
        self._merge_short_memory()
        # After merging, move all remaining short-term memories to
        # long-term storage.
        self.long_image_memory.extend(self.short_image_memory)
        self.short_image_memory.clear()
        # 合并 long memory，直到长度不超过 long_len
        self._merge_long_memory()

    def _merge_short_memory(self):
        """Merge adjacent short-term memories until the list length
        falls below or equals ``merge_len``.

        The heuristic merges the pair of *adjacent* image feature
        vectors with the highest cosine-like similarity (dot product
        over flattened features). The merged vector is their simple
        arithmetic mean. The process repeats until the desired length
        is reached or no further pairs can be merged.
        """
        while len(self.short_image_memory) > self.merge_len:
            max_sim = None
            max_idx = None
            # Search for the most similar adjacent pair.
            for i in range(len(self.short_image_memory) - 1):
                a = self.short_image_memory[i]
                b = self.short_image_memory[i + 1]
                sim = jnp.mean(jnp.dot(a.flatten(), b.flatten()))
                if (max_sim is None) or (sim > max_sim):
                    max_sim = sim
                    max_idx = i
            # If a pair is found, merge them; otherwise break.
            if max_idx is not None:
                merged = (
                    self.short_image_memory[max_idx]
                    + self.short_image_memory[max_idx + 1]
                ) / 2
                # Replace first item with merged result and delete second.
                self.short_image_memory[max_idx] = merged
                del self.short_image_memory[max_idx + 1]
            else:
                break

    def _merge_long_memory(self):
        """Merge adjacent long-term memories until the list length
        falls below or equals ``long_len``.

        The heuristic merges the pair of *adjacent* image feature
        vectors with the highest cosine-like similarity (dot product
        over flattened features). The merged vector is their simple
        arithmetic mean. The process repeats until the desired length
        is reached or no further pairs can be merged.
        """
        while len(self.long_image_memory) > self.long_len:
            max_sim = None
            max_idx = None
            # Search for the most similar adjacent pair.
            for i in range(len(self.long_image_memory) - 1):
                a = self.long_image_memory[i]
                b = self.long_image_memory[i + 1]
                sim = jnp.mean(jnp.dot(a.flatten(), b.flatten()))
                if (max_sim is None) or (sim > max_sim):
                    max_sim = sim
                    max_idx = i
            # If a pair is found, merge them; otherwise break.
            if max_idx is not None:
                merged = (
                    self.long_image_memory[max_idx]
                    + self.long_image_memory[max_idx + 1]
                ) / 2
                # Replace first item with merged result and delete second.
                self.long_image_memory[max_idx] = merged
                del self.long_image_memory[max_idx + 1]
            else:
                break

    def update_state_memory(self, state: jnp.ndarray):
        self.state_memory.append(state)
        # Keep only the most recent `state_len` states if a limit is set (>0).
        if self.state_len > 0:
            while len(self.state_memory) > self.state_len:
                self.state_memory.pop(0)

    def get_concat_image_features(
        self, current_image_features: jnp.ndarray
    ) -> jnp.ndarray:
        all_features = (
            self.long_image_memory + self.short_image_memory + [current_image_features]
        )
        if all_features:
            return jnp.concatenate(all_features, axis=0)
        else:
            return current_image_features

    def get_concat_states(self, current_state: jnp.ndarray) -> jnp.ndarray:
        all_states = self.state_memory + [current_state]
        if all_states:
            return jnp.concatenate(all_states, axis=0)
        else:
            return current_state

    def reset(self):
        self.short_image_memory.clear()
        self.long_image_memory.clear()
        self.state_memory.clear()

    # ---------------------------------------------------------------------
    # Equality & hashing helpers
    # ---------------------------------------------------------------------
    # During training initialization we compare PyTrees for structural
    # equality. Different `OpenPIMemory` instances are semantically
    # interchangeable at this stage (they contain no trainable weights).
    # Implementing `__eq__` to always return `True` (for other
    # `OpenPIMemory` objects) prevents spurious mismatches due to differing
    # object IDs. We also implement `__hash__` to satisfy the requirement
    # that objects overriding `__eq__` should be hashable if they are to be
    # used as dictionary keys inside PyTrees.

    def __eq__(self, other):  # noqa: D401, ANN001
        return isinstance(other, OpenPIMemory)

    def __hash__(self):  # noqa: D401
        # All instances are considered equal, so use a constant hash value.
        return 0

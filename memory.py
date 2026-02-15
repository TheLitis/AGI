"""
Replay buffer utilities and episodic memory sampling for world/self training.

Behavior unchanged; documentation and light style only.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import random

# =========================
#  Replay Buffer
# =========================


@dataclass
class Transition:
    """Single transition for replay buffer (observations, events, ids)."""
    obs_patch: np.ndarray
    energy: float
    action: int
    reward: float  # env-reward (trait-based), без любопытства
    done: bool
    next_obs_patch: np.ndarray
    next_energy: float
    death_flag: float  # 1.0, если эпизод кончился смертью, 0.0 иначе

    # сырые события среды
    got_food: float
    took_damage: float
    moved: float
    alive: float

    # id сценария (для перцепции и world model)
    scenario_id: int
    # id среды (для multi-env)
    env_id: int
    # optional tag for lifelong regimes / chapters
    regime_name: str = ""


class ReplayBuffer:
    """Cyclic replay buffer supporting sequential sampling with/without events."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.pos = 0

    def push(self, transition: Transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample_sequences(
        self,
        batch_size: int,
        seq_len: int,
    ) -> Tuple[np.ndarray, ...]:
        """
        Возвращает батч последовательностей длиной seq_len.
        Для простоты — подряд идущие куски.
        """
        assert len(self.buffer) >= seq_len
        obs_seq = []
        H_seq = []
        a_seq = []
        r_seq = []
        d_seq = []
        death_seq = []
        scenario_seq = []
        env_seq = []

        for _ in range(batch_size):
            start = random.randint(0, len(self.buffer) - seq_len)
            seq = self.buffer[start : start + seq_len]

            obs_seq.append(np.stack([tr.obs_patch for tr in seq], axis=0))  # (T,p,p)
            H_seq.append(
                np.array([tr.energy for tr in seq], dtype=np.float32).reshape(-1, 1)
            )
            a_seq.append(np.array([tr.action for tr in seq], dtype=np.int64))
            r_seq.append(np.array([tr.reward for tr in seq], dtype=np.float32))
            d_seq.append(np.array([float(tr.done) for tr in seq], dtype=np.float32))
            death_seq.append(
                np.array([tr.death_flag for tr in seq], dtype=np.float32)
            )
            scenario_seq.append(
                np.array([tr.scenario_id for tr in seq], dtype=np.int64)
            )
            env_seq.append(np.array([tr.env_id for tr in seq], dtype=np.int64))

        obs_seq = np.stack(obs_seq, axis=0)      # (B,T,p,p)
        H_seq = np.stack(H_seq, axis=0)          # (B,T,1)
        a_seq = np.stack(a_seq, axis=0)          # (B,T)
        r_seq = np.stack(r_seq, axis=0)          # (B,T)
        d_seq = np.stack(d_seq, axis=0)          # (B,T)
        death_seq = np.stack(death_seq, axis=0)  # (B,T)
        scenario_seq = np.stack(scenario_seq, axis=0)  # (B,T)
        env_seq = np.stack(env_seq, axis=0)      # (B,T)
        return obs_seq, H_seq, a_seq, r_seq, d_seq, death_seq, scenario_seq, env_seq

    def sample_sequences_with_events(
        self,
        batch_size: int,
        seq_len: int,
    ) -> Tuple[np.ndarray, ...]:
        """
        То же самое, что sample_sequences, но дополнительно возвращает
        последовательности сырых событий:
          got_food, took_damage, moved, alive.
        """
        assert len(self.buffer) >= seq_len

        obs_seq = []
        H_seq = []
        a_seq = []
        r_seq = []
        d_seq = []
        death_seq = []
        food_seq = []
        dmg_seq = []
        move_seq = []
        alive_seq = []
        scenario_seq = []
        env_seq = []

        for _ in range(batch_size):
            start = random.randint(0, len(self.buffer) - seq_len)
            seq = self.buffer[start : start + seq_len]

            obs_seq.append(np.stack([tr.obs_patch for tr in seq], axis=0))  # (T,p,p)
            H_seq.append(
                np.array([tr.energy for tr in seq], dtype=np.float32).reshape(-1, 1)
            )
            a_seq.append(np.array([tr.action for tr in seq], dtype=np.int64))
            r_seq.append(np.array([tr.reward for tr in seq], dtype=np.float32))
            d_seq.append(np.array([float(tr.done) for tr in seq], dtype=np.float32))
            death_seq.append(
                np.array([tr.death_flag for tr in seq], dtype=np.float32)
            )
            food_seq.append(
                np.array([tr.got_food for tr in seq], dtype=np.float32)
            )
            dmg_seq.append(
                np.array([tr.took_damage for tr in seq], dtype=np.float32)
            )
            move_seq.append(
                np.array([tr.moved for tr in seq], dtype=np.float32)
            )
            alive_seq.append(
                np.array([tr.alive for tr in seq], dtype=np.float32)
            )
            scenario_seq.append(
                np.array([tr.scenario_id for tr in seq], dtype=np.int64)
            )
            env_seq.append(np.array([tr.env_id for tr in seq], dtype=np.int64))

        obs_seq = np.stack(obs_seq, axis=0)       # (B,T,p,p)
        H_seq = np.stack(H_seq, axis=0)           # (B,T,1)
        a_seq = np.stack(a_seq, axis=0)           # (B,T)
        r_seq = np.stack(r_seq, axis=0)           # (B,T)
        d_seq = np.stack(d_seq, axis=0)           # (B,T)
        death_seq = np.stack(death_seq, axis=0)   # (B,T)
        food_seq = np.stack(food_seq, axis=0)     # (B,T)
        dmg_seq = np.stack(dmg_seq, axis=0)       # (B,T)
        move_seq = np.stack(move_seq, axis=0)     # (B,T)
        alive_seq = np.stack(alive_seq, axis=0)   # (B,T)
        scenario_seq = np.stack(scenario_seq, axis=0)  # (B,T)
        env_seq = np.stack(env_seq, axis=0)       # (B,T)

        return (
            obs_seq,
            H_seq,
            a_seq,
            r_seq,
            d_seq,
            death_seq,
            food_seq,
            dmg_seq,
            move_seq,
            alive_seq,
            scenario_seq,
            env_seq,
        )

    # --------- regime-aware sampling ---------
    def _valid_sequence_start_indices(self, regime_name: str, seq_len: int) -> List[int]:
        """
        Return start indices where a contiguous window of seq_len has the given regime_name.
        """
        starts: List[int] = []
        n = len(self.buffer)
        if n < seq_len:
            return starts
        for i in range(0, n - seq_len + 1):
            window = self.buffer[i : i + seq_len]
            if all(getattr(tr, "regime_name", "") == regime_name for tr in window):
                starts.append(i)
        return starts

    def _sample_sequence_at(self, start: int, seq_len: int, with_events: bool = False):
        """
        Build a sequence batch (length 1) starting at index start.
        """
        seq = self.buffer[start : start + seq_len]
        obs_seq = np.stack([tr.obs_patch for tr in seq], axis=0)  # (T,p,p)
        H_seq = np.array([tr.energy for tr in seq], dtype=np.float32).reshape(-1, 1)
        a_seq = np.array([tr.action for tr in seq], dtype=np.int64)
        r_seq = np.array([tr.reward for tr in seq], dtype=np.float32)
        d_seq = np.array([float(tr.done) for tr in seq], dtype=np.float32)
        death_seq = np.array([tr.death_flag for tr in seq], dtype=np.float32)
        scenario_seq = np.array([tr.scenario_id for tr in seq], dtype=np.int64)
        env_seq = np.array([tr.env_id for tr in seq], dtype=np.int64)

        if not with_events:
            return (
                obs_seq[None, ...],
                H_seq[None, ...],
                a_seq[None, ...],
                r_seq[None, ...],
                d_seq[None, ...],
                death_seq[None, ...],
                scenario_seq[None, ...],
                env_seq[None, ...],
            )

        food_seq = np.array([tr.got_food for tr in seq], dtype=np.float32)
        dmg_seq = np.array([tr.took_damage for tr in seq], dtype=np.float32)
        move_seq = np.array([tr.moved for tr in seq], dtype=np.float32)
        alive_seq = np.array([tr.alive for tr in seq], dtype=np.float32)
        return (
            obs_seq[None, ...],
            H_seq[None, ...],
            a_seq[None, ...],
            r_seq[None, ...],
            d_seq[None, ...],
            death_seq[None, ...],
            food_seq[None, ...],
            dmg_seq[None, ...],
            move_seq[None, ...],
            alive_seq[None, ...],
            scenario_seq[None, ...],
            env_seq[None, ...],
        )

    def sample_by_regime(self, regime_name: str, batch_size: int, seq_len: int = 1, with_events: bool = False):
        """
        Sample contiguous sequences belonging to a given regime_name.
        Falls back to uniform sampling if not enough data.
        """
        if len(self.buffer) < seq_len:
            raise AssertionError("Not enough data in buffer")

        starts = self._valid_sequence_start_indices(regime_name, seq_len)
        if not starts:
            # fallback: uniform across buffer
            return (
                self.sample_sequences_with_events(batch_size, seq_len)
                if with_events
                else self.sample_sequences(batch_size, seq_len)
            )

        batches = []
        for _ in range(batch_size):
            start = random.choice(starts)
            batches.append(self._sample_sequence_at(start, seq_len, with_events=with_events))

        # concatenate along batch dimension
        stacked = [np.concatenate([b[i] for b in batches], axis=0) for i in range(len(batches[0]))]
        return tuple(stacked)

    def sample_mixed(
        self,
        batch_size: int,
        seq_len: int = 1,
        mix_config: Optional[Dict[str, Any]] = None,
        with_events: bool = False,
    ):
        """
        Sample a batch mixing current and past regimes.

        mix_config keys (all optional):
          - current_regime: str name of active regime
          - frac_current: float in [0,1], default 0.5
          - past_regime_weights: Dict[str, float], optional weighted sampling over past regimes
          - sampling_temperature: float > 0, optional temperature for regime weights

        If regimes are missing, gracefully falls back to uniform sampling.
        """
        if len(self.buffer) < seq_len:
            raise AssertionError("Not enough data in buffer")

        if mix_config is None:
            mix_config = {}
        current_regime = mix_config.get("current_regime", "")
        try:
            frac_current = float(mix_config.get("frac_current", 0.5))
        except Exception:
            frac_current = 0.5
        frac_current = max(0.0, min(1.0, frac_current))
        raw_past_weights = mix_config.get("past_regime_weights", {}) or {}
        if not isinstance(raw_past_weights, dict):
            raw_past_weights = {}
        try:
            sampling_temperature = float(mix_config.get("sampling_temperature", 1.0))
        except Exception:
            sampling_temperature = 1.0
        sampling_temperature = max(1.0e-3, sampling_temperature)

        # collect available regimes
        regimes = set(getattr(tr, "regime_name", "") for tr in self.buffer)
        past_regimes = [r for r in regimes if r and r != current_regime]
        if not current_regime:
            past_regimes = list(regimes)

        n_current = int(round(batch_size * frac_current))
        n_past = batch_size - n_current

        batches = []
        # current regime part
        if n_current > 0:
            batches.append(
                self.sample_by_regime(current_regime, n_current, seq_len=seq_len, with_events=with_events)
            )
        # past regimes part (sample regimes uniformly)
        if n_past > 0 and past_regimes:
            weighted: List[float] = []
            for regime in past_regimes:
                try:
                    w = float(raw_past_weights.get(regime, 0.0))
                except Exception:
                    w = 0.0
                w = max(0.0, w)
                if sampling_temperature != 1.0 and w > 0.0:
                    w = float(w ** (1.0 / sampling_temperature))
                weighted.append(w)
            total_w = float(sum(weighted))
            use_weighted = total_w > 0.0
            collected = []
            for _ in range(n_past):
                if use_weighted:
                    regime = random.choices(past_regimes, weights=weighted, k=1)[0]
                else:
                    regime = random.choice(past_regimes)
                try:
                    collected.append(
                        self.sample_by_regime(regime, 1, seq_len=seq_len, with_events=with_events)
                    )
                except AssertionError:
                    continue
            if collected:
                part = [np.concatenate([c[i] for c in collected], axis=0) for i in range(len(collected[0]))]
                batches.append(tuple(part))

        if not batches:
            return (
                self.sample_sequences_with_events(batch_size, seq_len)
                if with_events
                else self.sample_sequences(batch_size, seq_len)
            )

        # merge along batch dimension
        merged = [np.concatenate([b[i] for b in batches], axis=0) for i in range(len(batches[0]))]
        B = int(merged[0].shape[0])
        if B < batch_size:
            extra_n = int(batch_size - B)
            extra = (
                self.sample_sequences_with_events(extra_n, seq_len)
                if with_events
                else self.sample_sequences(extra_n, seq_len)
            )
            merged = [np.concatenate([merged[i], extra[i]], axis=0) for i in range(len(merged))]
        return tuple(merged)

    def __len__(self):
        return len(self.buffer)

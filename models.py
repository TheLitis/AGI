"""
Neural building blocks: perception encoder, world model, self-model,
workspace combiner, value head, policy head, and small helpers.

Includes optional hashed-text instruction conditioning (no heavy NLP deps).
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
#  Models
# =========================


class Perception(nn.Module):
    """Embeds grid patches + scalar energy + scenario/env descriptors (+ optional text tokens)."""
    def __init__(
        self,
        n_cell_types: int = 5,
        emb_dim: int = 16,
        patch_size: int = 5,
        h_dim: int = 1,
        hidden_dim: int = 128,
        n_scenarios: int = 1,
        scenario_emb_dim: int = 8,
        env_desc_dim: int = 10,
        env_emb_dim: int = 8,
        text_vocab_size: int = 4096,
        text_emb_dim: int = 16,
        text_max_len: int = 24,
    ):
        super().__init__()
        self.emb = nn.Embedding(n_cell_types, emb_dim)
        self.scenario_emb = nn.Embedding(n_scenarios, scenario_emb_dim)
        self.env_desc_to_emb = nn.Sequential(
            nn.Linear(env_desc_dim, env_emb_dim),
            nn.ReLU(),
            nn.Linear(env_emb_dim, env_emb_dim),
        )
        self.text_vocab_size = int(text_vocab_size)
        self.text_emb_dim = int(text_emb_dim)
        self.text_max_len = int(text_max_len)
        self.text_emb = nn.Embedding(self.text_vocab_size, self.text_emb_dim, padding_idx=0)
        self.patch_size = patch_size
        in_dim = (
            patch_size * patch_size * emb_dim
            + h_dim
            + scenario_emb_dim
            + env_emb_dim
            + self.text_emb_dim
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        patch_ids: torch.Tensor,
        H: torch.Tensor,
        scenario_ids: torch.Tensor,
        env_desc: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        patch_ids: (B, p, p)
        H: (B, 1)
        scenario_ids: (B,)
        env_desc: (B, env_desc_dim)
        text_tokens: optional (B, L) integer ids; 0 is padding.
        """
        b = patch_ids.size(0)
        x = self.emb(patch_ids)  # (b, p, p, emb_dim)
        x = x.view(b, -1)  # (b, p*p*emb_dim)
        s_emb = self.scenario_emb(scenario_ids)  # (b, scenario_emb_dim)
        e_emb = self.env_desc_to_emb(env_desc)  # (b, env_emb_dim)
        if text_tokens is None or self.text_max_len <= 0:
            t_emb = torch.zeros(b, self.text_emb_dim, device=x.device, dtype=x.dtype)
        else:
            tt = text_tokens.to(device=x.device, dtype=torch.long)
            if tt.dim() == 1:
                tt = tt.view(b, -1)
            if tt.size(1) > self.text_max_len:
                tt = tt[:, : self.text_max_len]
            elif tt.size(1) < self.text_max_len:
                pad = torch.zeros(b, self.text_max_len - tt.size(1), device=x.device, dtype=torch.long)
                tt = torch.cat([tt, pad], dim=1)
            emb = self.text_emb(tt)  # (b, L, text_emb_dim)
            mask = (tt != 0).float().unsqueeze(-1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            t_emb = (emb * mask).sum(dim=1) / denom

        x = torch.cat([x, H, s_emb, e_emb, t_emb], dim=-1)
        z_obs = self.mlp(x)
        return z_obs


class WorldModel(nn.Module):
    """Autoregressive GRU predicting next latent observation and energy."""
    def __init__(
        self,
        obs_dim: int = 128,
        h_dim: int = 1,
        n_actions: int = 6,
        act_emb_dim: int = 8,
        w_dim: int = 128,
    ):
        super().__init__()
        self.act_emb = nn.Embedding(n_actions, act_emb_dim)
        self.gru = nn.GRU(
            input_size=obs_dim + h_dim + act_emb_dim,
            hidden_size=w_dim,
            batch_first=True,
        )
        self.head_obs = nn.Linear(w_dim, obs_dim)
        self.head_h = nn.Linear(w_dim, h_dim)
        # Event/reward heads used by planner for reward-aware imagination.
        self.head_alive = nn.Linear(w_dim, 1)
        self.head_food = nn.Linear(w_dim, 1)
        self.head_damage = nn.Linear(w_dim, 1)
        self.head_move = nn.Linear(w_dim, 1)
        self.head_reward = nn.Linear(w_dim, 1)

    def predict_event_components(self, w_t: torch.Tensor) -> dict:
        """
        Predict event probabilities and environment reward proxy from world latent.
        w_t: (..., w_dim)
        Returns dict with tensors shaped (..., 1).
        """
        alive = torch.sigmoid(self.head_alive(w_t))
        food = torch.sigmoid(self.head_food(w_t))
        damage = torch.sigmoid(self.head_damage(w_t))
        move = torch.sigmoid(self.head_move(w_t))
        reward_env = self.head_reward(w_t)
        return {
            "alive": alive,
            "food": food,
            "damage": damage,
            "move": move,
            "reward_env": reward_env,
        }

    def forward_step(
        self,
        z_obs_t: torch.Tensor,
        H_t: torch.Tensor,
        action_t: torch.Tensor,
        h_w_prev: Optional[torch.Tensor],
    ):
        """
        Один шаг GRU мирового модели.

        z_obs_t: (B, obs_dim)
        H_t:     (B, h_dim)
        action_t:(B,)
        h_w_prev:
          - (1,B,w_dim)  или
          - (B,w_dim)    или
          - None (будет инициализирован нулями)
        """
        a_emb = self.act_emb(action_t)  # (B, act_emb_dim)
        x = torch.cat([z_obs_t, H_t, a_emb], dim=-1)  # (B, obs_dim+h_dim+act_emb_dim)
        x = x.unsqueeze(1)  # (B,1,in_dim)

        B = x.size(0)
        if h_w_prev is None:
            h_w_prev = torch.zeros(
                1,
                B,
                self.gru.hidden_size,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            # допустим (B,w_dim) → приведём к (1,B,w_dim)
            if h_w_prev.dim() == 2:
                h_w_prev = h_w_prev.unsqueeze(0)
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: GRU хочет contiguous hx
            h_w_prev = h_w_prev.contiguous()

        out, h_w = self.gru(x, h_w_prev)  # out: (B,1,w_dim), h_w: (1,B,w_dim)
        w_t = h_w.squeeze(0)  # (B,w_dim)
        z_obs_hat = self.head_obs(w_t)
        H_hat = self.head_h(w_t)
        return w_t, h_w, z_obs_hat, H_hat

    def loss_supervised(
        self,
        z_obs_seq: torch.Tensor,
        H_seq: torch.Tensor,
        a_seq: torch.Tensor,
        event_targets: Optional[dict] = None,
        event_weights: Optional[dict] = None,
    ) -> torch.Tensor:
        b, T, obs_dim = z_obs_seq.shape
        z_in = z_obs_seq[:, :-1, :]
        H_in = H_seq[:, :-1, :]
        a_in = a_seq[:, :-1]

        a_emb = self.act_emb(a_in)  # (b,T-1,act_emb_dim)
        x = torch.cat([z_in, H_in, a_emb], dim=-1)

        h0 = torch.zeros(1, b, self.gru.hidden_size, device=z_obs_seq.device)
        out, _ = self.gru(x, h0)

        z_hat = self.head_obs(out)
        H_hat = self.head_h(out)

        z_target = z_obs_seq[:, 1:, :]
        H_target = H_seq[:, 1:, :]

        loss_z = F.mse_loss(z_hat, z_target)
        loss_h = F.mse_loss(H_hat, H_target)
        total = loss_z + loss_h

        if isinstance(event_targets, dict):
            weights = event_weights or {}
            w_alive = float(weights.get("alive", 0.5))
            w_food = float(weights.get("food", 0.5))
            w_damage = float(weights.get("damage", 0.5))
            w_move = float(weights.get("move", 0.3))
            w_reward = float(weights.get("reward", 0.2))

            preds = self.predict_event_components(out)
            eps = 1.0e-6

            def _event_target(name: str) -> Optional[torch.Tensor]:
                t = event_targets.get(name)
                if not isinstance(t, torch.Tensor):
                    return None
                tt = t.to(device=out.device, dtype=out.dtype)
                if tt.dim() == 2:
                    tt = tt.unsqueeze(-1)
                if tt.shape[:2] != out.shape[:2]:
                    return None
                return tt

            alive_t = _event_target("alive")
            if alive_t is not None and w_alive > 0.0:
                total = total + w_alive * F.binary_cross_entropy(
                    preds["alive"].clamp(min=eps, max=1.0 - eps),
                    alive_t.clamp(min=0.0, max=1.0),
                )

            food_t = _event_target("food")
            if food_t is not None and w_food > 0.0:
                total = total + w_food * F.binary_cross_entropy(
                    preds["food"].clamp(min=eps, max=1.0 - eps),
                    food_t.clamp(min=0.0, max=1.0),
                )

            damage_t = _event_target("damage")
            if damage_t is not None and w_damage > 0.0:
                total = total + w_damage * F.binary_cross_entropy(
                    preds["damage"].clamp(min=eps, max=1.0 - eps),
                    damage_t.clamp(min=0.0, max=1.0),
                )

            move_t = _event_target("move")
            if move_t is not None and w_move > 0.0:
                total = total + w_move * F.binary_cross_entropy(
                    preds["move"].clamp(min=eps, max=1.0 - eps),
                    move_t.clamp(min=0.0, max=1.0),
                )

            reward_t = _event_target("reward")
            if reward_t is not None and w_reward > 0.0:
                total = total + w_reward * F.mse_loss(preds["reward_env"], reward_t)

        return total

    def curiosity_error(
        self,
        z_obs_next: torch.Tensor,
        H_next: torch.Tensor,
        z_obs_hat: torch.Tensor,
        H_hat: torch.Tensor,
        lambda_h: float = 0.1,
    ) -> torch.Tensor:
        err_z = F.mse_loss(z_obs_hat, z_obs_next, reduction="none").mean(dim=-1)
        err_h = F.mse_loss(H_hat, H_next, reduction="none").mean(dim=-1)
        err = err_z + lambda_h * err_h
        return err  # (batch,)


class SelfModel(nn.Module):
    """
    SelfModel: по истории (W,H,r,a,M) предсказывает:
      - дисконтированные суммы по компонентам:
          * survival_frac_t   — оценка доли прожитой жизни,
          * food_ret_t        — ожидаемая еда,
          * damage_ret_t      — ожидаемые "удары",
          * move_ret_t        — ожидаемое движение,
      - uncertainty          — насколько свои предсказания ненадёжны,
      - и даёт проекцию S → M (to_memory) для обновления памяти,
      - и калиброванный скалярный "self-return" через маленькую голову.
    """

    def __init__(
        self,
        w_dim: int = 128,
        h_dim: int = 1,
        n_actions: int = 6,
        s_dim: int = 64,
        mem_dim: int = 32,
        env_desc_dim: int = 10,
        env_emb_dim: int = 8,
    ):
        super().__init__()
        self.act_emb = nn.Embedding(n_actions, 8)
        self.mem_dim = mem_dim

        self.env_desc_to_emb = nn.Sequential(
            nn.Linear(env_desc_dim, env_emb_dim),
            nn.ReLU(),
            nn.Linear(env_emb_dim, env_emb_dim),
        )

        # W + H + r + a_emb + M + env_emb
        in_dim = w_dim + h_dim + 1 + 8 + mem_dim + env_emb_dim
        self.gru = nn.GRU(input_size=in_dim, hidden_size=s_dim, batch_first=True)

        self.head_survival = nn.Linear(s_dim, 1)
        self.head_survival_calib = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.head_food = nn.Linear(s_dim, 1)
        self.head_damage = nn.Linear(s_dim, 1)
        self.head_move = nn.Linear(s_dim, 1)
        self.head_uncertainty = nn.Linear(s_dim, 1)

        # калибрующая голова для self-return (маленький MLP)
        self.head_return_calib = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.to_memory = nn.Linear(s_dim, mem_dim)

    def forward_seq(
        self,
        W_seq: torch.Tensor,
        H_seq: torch.Tensor,
        a_seq: torch.Tensor,
        r_seq: torch.Tensor,
        M: torch.Tensor,
        env_desc: torch.Tensor,
        h_s0: Optional[torch.Tensor] = None,
    ):
        """
        W_seq: (B,T,w_dim)
        H_seq: (B,T,h_dim)
        a_seq: (B,T)
        r_seq: (B,T)
        M: (1,mem_dim) или (B,mem_dim)
        env_desc: (B,T,env_desc_dim)
        """
        B, T, w_dim = W_seq.shape
        a_emb = self.act_emb(a_seq)  # (B,T,8)

        env_emb = self.env_desc_to_emb(env_desc)  # (B,T,env_emb_dim)

        if M.dim() == 2:
            M_b = M.unsqueeze(1).expand(B, T, -1)
        else:
            raise ValueError("M must be (1,mem_dim) or (B,mem_dim)")

        x = torch.cat(
            [W_seq, H_seq, r_seq.unsqueeze(-1), a_emb, M_b, env_emb],
            dim=-1,
        )  # (B,T,in_dim)

        if h_s0 is None:
            h_s0 = torch.zeros(
                1,
                B,
                self.gru.hidden_size,
                device=W_seq.device,
            )

        out, h_s = self.gru(x, h_s0)  # out: (B,T,s_dim)
        S_seq = out
        S_last = h_s.squeeze(0)  # (B,s_dim)

        # survival: сырое и откалиброванное [0,1]
        surv_logit = self.head_survival(S_seq)             # (B,T,1)
        surv_raw = torch.sigmoid(surv_logit)               # (B,T,1)
        surv_calib_logit = self.head_survival_calib(surv_raw)
        surv = torch.sigmoid(surv_calib_logit)             # (B,T,1)
        food = self.head_food(S_seq)                       # (B,T,1)
        damage = self.head_damage(S_seq)                   # (B,T,1)
        move = self.head_move(S_seq)                       # (B,T,1)
        unc = self.head_uncertainty(S_seq)                 # (B,T,1)

        return S_seq, S_last, surv, food, damage, move, unc, surv_raw


class Workspace(nn.Module):
    """Combines world/self/value signals, uncertainty, traits, memory into policy context."""
    def __init__(
        self,
        w_dim: int = 128,
        s_dim: int = 64,
        h_dim: int = 1,
        trait_dim: int = 4,
        mem_dim: int = 32,
        g_dim: int = 64,
    ):
        super().__init__()
        # В G_t заходит: W + S + H + V + Δ_self + U + traits + M
        in_dim = w_dim + s_dim + h_dim + 1 + 1 + 1 + trait_dim + mem_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, g_dim),
            nn.ReLU(),
        )
        self.g_dim = g_dim

    def forward(
        self,
        W_t: torch.Tensor,
        S_t: torch.Tensor,
        H_t: torch.Tensor,
        V_t: torch.Tensor,
        delta_self: torch.Tensor,
        U_t: torch.Tensor,
        traits: torch.Tensor,
        M: torch.Tensor,
    ) -> torch.Tensor:
        """
        Все входы — (B,dim), кроме traits/M — тоже (1,dim) → broadcast до B.
        """
        B = W_t.size(0)
        if traits.size(0) == 1 and B > 1:
            traits = traits.expand(B, -1)
        if M.size(0) == 1 and B > 1:
            M = M.expand(B, -1)

        x = torch.cat(
            [W_t, S_t, H_t, V_t, delta_self, U_t, traits, M],
            dim=-1,
        )
        G_t = self.mlp(x)
        return G_t


class ValueModel(nn.Module):
    """Value head over (world state, energy, traits, memory), pairs with SelfModel."""

    def __init__(
        self,
        w_dim: int = 128,
        h_dim: int = 1,
        trait_dim: int = 4,
        mem_dim: int = 32,
    ):
        super().__init__()
        in_dim = w_dim + h_dim + trait_dim + mem_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        W_t: torch.Tensor,
        H_t: torch.Tensor,
        traits: torch.Tensor,
        M: torch.Tensor,
    ) -> torch.Tensor:
        B = W_t.size(0)
        if traits.size(0) == 1 and B > 1:
            traits = traits.expand(B, -1)
        if M.size(0) == 1 and B > 1:
            M = M.expand(B, -1)
        x = torch.cat([W_t, H_t, traits, M], dim=-1)
        V_t = self.mlp(x)
        return V_t


class Policy(nn.Module):
    def __init__(self, g_dim: int = 64, n_actions: int = 6):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(g_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.pi_head = nn.Linear(64, n_actions)
        # Auxiliary head that predicts action validity mask from the same features.
        # Used for "mask internalization" in tool-like environments.
        self.mask_head = nn.Linear(64, n_actions)
        # Optional safety heads (per-action risk logits).
        self.risk_violation_head = nn.Linear(64, n_actions)
        self.risk_catastrophic_head = nn.Linear(64, n_actions)

    def forward(self, G_t: torch.Tensor):
        h = self.shared(G_t)
        return self.pi_head(h)

    def forward_with_mask(self, G_t: torch.Tensor):
        """
        Return policy logits and auxiliary mask logits.
        """
        h = self.shared(G_t)
        logits = self.pi_head(h)
        mask_logits = self.mask_head(h)
        return logits, mask_logits

    def forward_with_mask_and_risk(self, G_t: torch.Tensor):
        """
        Extended additive API: policy + mask logits + per-action risk logits.
        """
        h = self.shared(G_t)
        logits = self.pi_head(h)
        mask_logits = self.mask_head(h)
        risk_violation_logits = self.risk_violation_head(h)
        risk_catastrophic_logits = self.risk_catastrophic_head(h)
        return logits, mask_logits, risk_violation_logits, risk_catastrophic_logits


class HighLevelPolicy(nn.Module):
    """Policy over discrete skills (macro-actions)."""

    def __init__(self, g_dim: int, n_skills: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(g_dim, g_dim),
            nn.ReLU(),
            nn.Linear(g_dim, n_skills),
        )

    def forward(self, G_t: torch.Tensor) -> torch.Tensor:
        return self.mlp(G_t)


# =========================
#  Helpers: traits → weights
# =========================


def traits_to_preference_weights(traits: torch.Tensor) -> torch.Tensor:
    """
    Map latent traits -> preference weights
    (w_survive, w_food, w_danger, w_move), with a bounded tanh mapping.

    Supports both single-faction vectors ([4]) and stacked factions
    ([num_factions, 4]); preserves the input rank (1D->1D, 2D->2D).
    """
    base_survive = 0.7
    base_food = 0.7
    base_danger = -0.7
    base_move = 0.0

    if traits.dim() == 1:
        traits_in = traits.view(1, -1)
    elif traits.dim() == 2:
        traits_in = traits
    else:
        raise ValueError(f"traits_to_preference_weights expects 1D or 2D input, got shape {traits.shape}")

    t0, t1, t2, t3 = traits_in.unbind(dim=-1)

    w_survive = base_survive + 0.3 * torch.tanh(t0)
    w_food = base_food + 0.3 * torch.tanh(t1)
    w_danger = base_danger + 0.5 * torch.tanh(t2)
    w_move = base_move + 0.5 * torch.tanh(t3)

    weights = torch.stack([w_survive, w_food, w_danger, w_move], dim=-1)
    if traits.dim() == 1:
        return weights.view(-1)
    return weights


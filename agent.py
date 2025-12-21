"""
Agent wrapper that bundles perception, world model, self-model, workspace,
value head, and policy head used across the four training stages.

Numerical behavior is unchanged; this module adds documentation and small
style cleanups.
"""

from typing import Optional, List, Any, Dict
from types import SimpleNamespace

import torch
import torch.nn as nn  # ��?�>�� �?�?��-�'�? ���?�?���?�?�+��'�?�?

from models import (
    Perception,
    WorldModel,
    SelfModel,
    Workspace,
    ValueModel,
    Policy,
    HighLevelPolicy,
)
from skills import SkillLibrary


# =========================
#  Agent
# =========================


class ProtoCreatureAgent(nn.Module):
    """Container for neural modules and their optimizers."""

    def __init__(
        self,
        n_cell_types: int = 5,
        device: Optional[torch.device] = None,
        w_dim: int = 128,
        s_dim: int = 64,
        h_dim: int = 1,
        trait_dim: int = 4,
        num_factions: int = 2,
        mem_dim: int = 32,
        n_mem_slots: int = 8,
        n_scenarios: int = 1,
        env_descriptors: Optional[torch.Tensor] = None,
        use_skills: bool = False,
        n_skills: int = 0,
        skill_mode: str = "handcrafted",
        n_latent_skills: int = 0,
        n_actions: int = 6,
    ):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.n_actions = int(n_actions)
        self.w_dim = w_dim
        self.h_dim = h_dim
        self.trait_dim = trait_dim

        if env_descriptors is None:
            raise ValueError("env_descriptors tensor is required")
        self.env_descriptors = env_descriptors.to(self.device)
        self.env_desc_dim = self.env_descriptors.shape[-1]
        # context_dim = w + h + env_desc + traits + step_in_skill
        self.skill_context_dim = self.w_dim + self.h_dim + self.env_desc_dim + self.trait_dim + 1

        self.perception = Perception(
            n_cell_types=n_cell_types,
            patch_size=5,
            h_dim=h_dim,
            hidden_dim=128,
            n_scenarios=n_scenarios,
            scenario_emb_dim=8,
            env_desc_dim=self.env_desc_dim,
            env_emb_dim=8,
        ).to(self.device)

        self.world_model = WorldModel(
            obs_dim=128,
            h_dim=h_dim,
            n_actions=self.n_actions,
            act_emb_dim=8,
            w_dim=w_dim,
        ).to(self.device)

        self.self_model = SelfModel(
            w_dim=w_dim,
            h_dim=h_dim,
            n_actions=self.n_actions,
            s_dim=s_dim,
            mem_dim=mem_dim,
            env_desc_dim=self.env_desc_dim,
            env_emb_dim=8,
        ).to(self.device)

        self.workspace = Workspace(
            w_dim=w_dim,
            s_dim=s_dim,
            h_dim=h_dim,
            trait_dim=trait_dim,
            mem_dim=mem_dim,
            g_dim=64,
        ).to(self.device)

        self.value_model = ValueModel(
            w_dim=w_dim,
            h_dim=h_dim,
            trait_dim=trait_dim,
            mem_dim=mem_dim,
        ).to(self.device)

        self.policy = Policy(g_dim=64, n_actions=self.n_actions).to(self.device)
        self.use_skills = bool(use_skills)
        self.n_skills = int(n_skills) if self.use_skills else 0
        self.skill_mode = (skill_mode or "handcrafted").lower()
        self.n_latent_skills = int(n_latent_skills) if self.skill_mode in {"latent", "mixed"} else 0
        self.total_skills = self.n_skills + (self.n_latent_skills if self.use_skills else 0)
        self.high_level_policy: Optional[HighLevelPolicy] = None
        if self.use_skills and self.total_skills > 0:
            self.high_level_policy = HighLevelPolicy(g_dim=64, n_skills=self.total_skills).to(self.device)
        # latent skill defaults (can be overridden via external config)
        self.latent_skill_dim = 16
        self.latent_skill_hidden_dim = 64
        self.skill_horizon = 1
        self.skill_library: Optional[SkillLibrary] = None
        if self.use_skills and self.n_latent_skills > 0:
            lib_cfg = SimpleNamespace(
                num_latent_skills=self.n_latent_skills,
                latent_dim=self.latent_skill_dim,
                latent_skill_hidden_dim=self.latent_skill_hidden_dim,
                skill_horizon=self.skill_horizon,
            )
            self.skill_library = SkillLibrary(
                config=lib_cfg,
                num_actions=self.n_actions,
                context_dim=self.skill_context_dim,
            ).to(self.device)

        # Traits (interpretable slow values):
        #   0: survival preference (higher -> more weight on staying alive)
        #   1: food preference (higher -> more weight on food/extrinsic reward)
        #   2: danger avoidance vs seeking (negative avoids damage, positive seeks danger)
        #   3: movement/exploration vs efficiency (higher -> more exploration/movement)
        self.num_factions = max(1, int(num_factions))
        self.traits_dim = trait_dim
        traits_init = torch.zeros(self.num_factions, trait_dim, device=self.device)
        if self.num_factions >= 2:
            # safety faction: tilt toward survival / damage aversion
            traits_init[1, 0] = 0.5
            traits_init[1, 2] = -0.5
        self.traits = nn.Parameter(traits_init)
        fw_init = torch.zeros(self.num_factions, device=self.device)
        if self.num_factions >= 2:
            fw_init[0] = 0.7  # bias mixing toward the main faction at start
        self.faction_weights = nn.Parameter(fw_init)

        # �?������?�?��ؐ�?����? �����?�?�'�?: �?��?��?�>�?��? �?�>�?�'�?�? + �?����'�?�� �?����?�?��
        self.n_mem_slots = n_mem_slots
        self.mem_dim = mem_dim
        self.memory_slots = torch.zeros(1, n_mem_slots, mem_dim, device=self.device)
        self.memory = torch.zeros(1, mem_dim, device=self.device)
        self.memory_slot_ptr = 0
        # Shared reference (populated by Trainer) to trait reflection events
        self.trait_reflection_log: List[Dict[str, Any]] = []

        # �?���'��?������'�?�?�<
        self.optim_world = torch.optim.Adam(
            list(self.perception.parameters()) + list(self.world_model.parameters()),
            lr=1e-3,
        )
        policy_params = (
            list(self.policy.parameters())
            + list(self.value_model.parameters())
            + list(self.workspace.parameters())
        )
        if self.high_level_policy is not None:
            policy_params += list(self.high_level_policy.parameters())
        if self.skill_library is not None:
            policy_params += list(self.skill_library.parameters())
        self.optim_policy = torch.optim.Adam(
            policy_params,
            lr=3e-4,
        )
        self.optim_self = torch.optim.Adam(self.self_model.parameters(), lr=1e-3)

    def reset_memory(self):
        """
        Reset episodic memory slots and summary memory to zeros
        (keeps shapes and pointers intact).
        """
        self.memory_slots.zero_()
        self.memory.zero_()
        self.memory_slot_ptr = 0

    def get_env_desc(self, env_ids: torch.Tensor) -> torch.Tensor:
        """
        Map env_id tensor -> env descriptor embeddings ready for models.
        env_ids: (...,) long
        """
        return self.env_descriptors[env_ids]

    def write_memory(
        self,
        M_target: torch.Tensor,
        alpha_slot: float = 0.5,
        alpha_summary: float = 0.1,
    ):
        """
        Lightweight EMA write into episodic slots plus a slow summary memory.

        M_target: (1, mem_dim)
        """
        with torch.no_grad():
            if M_target.dim() == 3:
                # (1,1,mem_dim) -> (1,mem_dim)
                M_target = M_target.squeeze(1)
            assert (
                M_target.shape == self.memory.shape
            ), f"M_target shape {M_target.shape} != memory shape {self.memory.shape}"
            idx = self.memory_slot_ptr
            # �?�+�?�?�?�>�?��? �?�<�+�?���?�?�<�� �?�>�?�'
            self.memory_slots[:, idx, :] = self.memory_slots[:, idx, :] + alpha_slot * (
                M_target - self.memory_slots[:, idx, :]
            )
            # ����?��?�ؑ'�' �?����?�?�� ����� EMA �� �?�?��?�?��?�? ���? �?�>�?�'���?
            mem_mean = self.memory_slots.mean(dim=1)  # (1,mem_dim)
            self.memory = self.memory + alpha_summary * (mem_mean - self.memory)
            self.memory_slot_ptr = (self.memory_slot_ptr + 1) % self.n_mem_slots

    def get_fast_params(self) -> List[nn.Parameter]:
        """
        Parameters allowed to adapt during lifelong training (fast subset).
        Default: SelfModel + Workspace + Policy + ValueModel.
        """
        params: List[nn.Parameter] = []
        params.extend(self.self_model.parameters())
        params.extend(self.workspace.parameters())
        params.extend(self.policy.parameters())
        params.extend(self.value_model.parameters())
        if self.high_level_policy is not None:
            params.extend(self.high_level_policy.parameters())
        if self.skill_library is not None:
            params.extend(self.skill_library.parameters())
        return params

    def get_trait_reflection_log(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Return the structured trait reflection log (optionally truncated).
        Trainer owns the log; Agent holds a shared reference for external consumers.
        """
        if limit is None or limit <= 0:
            return list(self.trait_reflection_log)
        return list(self.trait_reflection_log[-limit:])

    def get_slow_params(self) -> List[nn.Parameter]:
        """
        Parameters kept frozen in lifelong training (slow subset).
        Default: Perception encoder + WorldModel.
        """
        params: List[nn.Parameter] = []
        params.extend(self.perception.parameters())
        params.extend(self.world_model.parameters())
        return params

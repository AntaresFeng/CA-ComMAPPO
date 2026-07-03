"""Compatibility shims for the local XuanCe version used by CA-ComMAPPO."""


def patch_xuance_marl_buffer_aliases() -> None:
    """Patch XuanCe 1.4.3 MARL buffer field-name drift in this process.

    `BaseBuffer` stores the newer names `num_envs`, `per_env_buffer_size`,
    `observation_space`, and `action_space`, but `MARL_OnPolicyBuffer` still
    reads the older names `n_envs`, `n_size`, `obs_space`, and `act_space`.
    Adding aliases after `BaseBuffer.__init__` lets MAPPO build its on-policy
    buffer without editing installed files under `.venv`.
    """
    from xuance.common.memory_tools_marl import BaseBuffer

    if getattr(BaseBuffer, "_ca_commappo_alias_patch", False):
        return

    original_init = BaseBuffer.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Keep the shim narrow: only expose the legacy names used downstream.
        self.n_envs = self.num_envs
        self.n_size = self.per_env_buffer_size
        self.obs_space = self.observation_space
        self.act_space = self.action_space

    BaseBuffer.__init__ = patched_init
    BaseBuffer._ca_commappo_alias_patch = True

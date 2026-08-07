import subprocess
import sys
from importlib.metadata import version

import numpy as np
from gymnasium.spaces import Box, Discrete

from ca_commappo.xuance_compat import (
    patch_xuance_mappo_masked_statistics,
    patch_xuance_marl_buffer_aliases,
)


def test_xuance_version_matches_buffer_alias_shim_target():
    assert version("xuance") == "1.4.3"


def test_patch_xuance_marl_buffer_aliases_fixes_current_on_policy_buffer():
    unpatched = _run_on_policy_buffer_probe(patched=False)
    assert unpatched.returncode != 0
    assert "n_envs" in unpatched.stderr

    patched = _run_on_policy_buffer_probe(patched=True)
    assert patched.returncode == 0, patched.stderr
    assert patched.stdout.strip() == "2 2"


def test_masked_advantage_normalization_ignores_inactive_values():
    from xuance.common.memory_tools_marl import MARL_OnPolicyBuffer

    patch_xuance_marl_buffer_aliases()
    patch_xuance_mappo_masked_statistics()
    sample_method = MARL_OnPolicyBuffer.sample
    patch_xuance_mappo_masked_statistics()
    assert MARL_OnPolicyBuffer.sample is sample_method

    buffer = _make_full_on_policy_buffer()
    buffer.data["advantages"]["agent_0"][:] = [[1.0, 1.0e8, 3.0, -1.0e8]]
    buffer.data["agent_mask"]["agent_0"][:] = [[True, False, True, False]]

    sample = buffer.sample(np.arange(4))

    np.testing.assert_allclose(
        sample["advantages"]["agent_0"], [-1.0, 0.0, 1.0, 0.0], atol=1e-6
    )


def test_masked_value_norm_uses_only_active_parameter_shared_targets():
    probe = _run_value_norm_probe(parameter_sharing=True)
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "7.0"


def test_masked_value_norm_uses_corresponding_non_shared_targets():
    probe = _run_value_norm_probe(parameter_sharing=False)
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "2.0 12.0"


def _run_on_policy_buffer_probe(*, patched: bool) -> subprocess.CompletedProcess[str]:
    patch_code = (
        "from ca_commappo.xuance_compat import patch_xuance_marl_buffer_aliases\n"
        "patch_xuance_marl_buffer_aliases()\n"
        if patched
        else ""
    )
    code = f"""
import numpy as np
from gymnasium.spaces import Box, Discrete
{patch_code}
from xuance.common.memory_tools_marl import MARL_OnPolicyBuffer

agent_keys = ["agent_0", "agent_1"]
obs_space = {{key: Box(-1.0, 1.0, shape=(3,), dtype=np.float32) for key in agent_keys}}
act_space = {{key: Discrete(3) for key in agent_keys}}
buffer = MARL_OnPolicyBuffer(
    agent_keys=agent_keys,
    obs_space=obs_space,
    act_space=act_space,
    n_envs=2,
    buffer_size=4,
    use_gae=False,
    use_advnorm=False,
    gamma=0.99,
    gae_lam=0.95,
)
print(buffer.n_envs, buffer.n_size)
"""
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
    )


def _make_full_on_policy_buffer():
    from xuance.common.memory_tools_marl import MARL_OnPolicyBuffer

    agent_keys = ["agent_0"]
    buffer = MARL_OnPolicyBuffer(
        agent_keys=agent_keys,
        obs_space={"agent_0": Box(-1.0, 1.0, shape=(3,), dtype=np.float32)},
        act_space={"agent_0": Discrete(3)},
        n_envs=1,
        buffer_size=4,
        use_gae=False,
        use_advnorm=True,
        gamma=0.99,
        gae_lam=0.95,
    )
    buffer.size = buffer.n_size
    return buffer


def _run_value_norm_probe(
    *, parameter_sharing: bool
) -> subprocess.CompletedProcess[str]:
    if parameter_sharing:
        model_keys = '["shared"]'
        fake_update_body = """
    targets = torch.as_tensor(
        np.stack([sample["returns"][key] for key in self.agent_keys], axis=1)
    ).reshape(-1)
    self.value_normalizer["shared"].update(targets)
"""
        output = """
print(round(learner.value_normalizer["shared"].running_mean_var()[0].item(), 6))
"""
    else:
        model_keys = '["agent_0", "agent_1"]'
        fake_update_body = """
    for key in self.model_keys:
        self.value_normalizer[key].update(torch.as_tensor(sample["returns"][key]))
"""
        output = """
print(
    round(learner.value_normalizer["agent_0"].running_mean_var()[0].item(), 6),
    round(learner.value_normalizer["agent_1"].running_mean_var()[0].item(), 6),
)
"""

    code = f"""
from types import SimpleNamespace

import numpy as np
import torch
from xuance.torch.learners.multi_agent_rl.mappo_learner import MAPPO_Learner
from xuance.torch.utils import ValueNorm

def fake_update(self, sample):
{fake_update_body}

MAPPO_Learner.update = fake_update
from ca_commappo.xuance_compat import patch_xuance_mappo_masked_statistics
patch_xuance_mappo_masked_statistics()
patched_update = MAPPO_Learner.update
patch_xuance_mappo_masked_statistics()
assert MAPPO_Learner.update is patched_update

learner = SimpleNamespace(
    use_value_norm=True,
    use_parameter_sharing={parameter_sharing!r},
    agent_keys=["agent_0", "agent_1"],
    model_keys={model_keys},
    value_normalizer={{key: ValueNorm(1) for key in {model_keys}}},
)
original_normalizers = dict(learner.value_normalizer)
sample = {{
    "returns": {{
        "agent_0": np.array([1.0, 1.0e8, 3.0, -1.0e8], dtype=np.float32),
        "agent_1": np.array([10.0, 1.0e8, 14.0, -1.0e8], dtype=np.float32),
    }},
    "agent_mask": {{
        "agent_0": np.array([True, False, True, False]),
        "agent_1": np.array([True, False, True, False]),
    }},
}}
MAPPO_Learner.update(learner, sample)
assert all(
    learner.value_normalizer[key] is normalizer
    for key, normalizer in original_normalizers.items()
)
{output}
"""
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
    )

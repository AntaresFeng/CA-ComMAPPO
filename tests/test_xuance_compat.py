import subprocess
import sys
from importlib.metadata import version


def test_xuance_version_matches_buffer_alias_shim_target():
    assert version("xuance") == "1.4.3"


def test_patch_xuance_marl_buffer_aliases_fixes_current_on_policy_buffer():
    unpatched = _run_on_policy_buffer_probe(patched=False)
    assert unpatched.returncode != 0
    assert "n_envs" in unpatched.stderr

    patched = _run_on_policy_buffer_probe(patched=True)
    assert patched.returncode == 0, patched.stderr
    assert patched.stdout.strip() == "2 2"


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

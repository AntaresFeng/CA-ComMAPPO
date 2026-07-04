from ca_commappo.evaluation.highway_metrics import (
    build_episode_record,
    episode_outcome,
    format_summary_lines,
    summarize_episode_records,
    summary_to_log_infos,
)


def test_episode_outcome_prefers_collision_over_arrival_and_truncation():
    outcome = episode_outcome(
        crashed=[False, True],
        arrived=[True, True],
        truncated=True,
    )

    assert outcome == {
        "collision": True,
        "arrival": False,
        "truncated": False,
    }


def test_episode_outcome_marks_success_only_when_all_agents_arrive():
    outcome = episode_outcome(
        crashed=[False, False, False],
        arrived=[True, True, True],
        truncated=False,
    )

    assert outcome == {
        "collision": False,
        "arrival": True,
        "truncated": False,
    }


def test_build_episode_record_computes_reward_and_agent_fractions():
    record = build_episode_record(
        phase="eval",
        episode_index=3,
        steps=7,
        agent_rewards={"agent_0": 1.0, "agent_1": 3.0},
        crashed_agents=[True, False],
        arrived_agents=[False, True],
        truncated=False,
        seed=11,
        score=2.0,
    )

    assert record == {
        "phase": "eval",
        "seed": 11,
        "episode_index": 3,
        "steps": 7,
        "episode_reward": 2.0,
        "score": 2.0,
        "agent_rewards": {"agent_0": 1.0, "agent_1": 3.0},
        "collision": True,
        "arrival": False,
        "truncated": False,
        "crashed_agents": [True, False],
        "arrived_agents": [False, True],
        "agent_collision_fraction": 0.5,
        "agent_arrival_fraction": 0.5,
    }


def test_summarize_episode_records_keeps_sanity_metric_keys():
    records = [
        build_episode_record(
            policy="random",
            episode_index=0,
            steps=5,
            agent_rewards={"agent_0": 1.0, "agent_1": 3.0},
            crashed_agents=[True, False],
            arrived_agents=[False, False],
            truncated=False,
        ),
        build_episode_record(
            policy="random",
            episode_index=1,
            steps=10,
            agent_rewards={"agent_0": 2.0, "agent_1": 2.0},
            crashed_agents=[False, False],
            arrived_agents=[True, True],
            truncated=False,
        ),
    ]

    summary = summarize_episode_records(records)

    assert summary == {
        "episodes": 2,
        "mean_episode_reward": 2.0,
        "mean_agent_reward": 2.0,
        "mean_episode_length": 7.5,
        "collision_rate": 0.5,
        "arrival_rate": 0.5,
        "truncation_rate": 0.0,
        "mean_agent_collision_fraction": 0.25,
        "mean_agent_arrival_fraction": 0.5,
    }


def test_summary_to_log_infos_prefixes_flat_scalar_keys():
    summary = {
        "episodes": 2,
        "mean_episode_reward": 2.0,
        "collision_rate": 0.5,
    }

    assert summary_to_log_infos("Eval-Highway", summary) == {
        "Eval-Highway/episodes": 2.0,
        "Eval-Highway/mean_episode_reward": 2.0,
        "Eval-Highway/collision_rate": 0.5,
    }


def test_format_summary_lines_is_shared_by_cli_and_training_output():
    lines = format_summary_lines(
        {
            "episodes": 2,
            "mean_episode_reward": 2.0,
            "mean_agent_reward": 2.0,
            "mean_episode_length": 7.5,
            "collision_rate": 0.5,
            "arrival_rate": 0.5,
            "truncation_rate": 0.0,
        },
        indent="  ",
    )

    assert lines == [
        "  episodes=2",
        "  mean_episode_reward=2.000",
        "  mean_agent_reward=2.000",
        "  mean_episode_length=7.500",
        "  collision_rate=0.500",
        "  arrival_rate=0.500",
        "  truncation_rate=0.000",
    ]

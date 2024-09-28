import time
from typing import Any, Dict, List, Tuple

from aij_multiagent_rl.agents import BaseAgent
from aij_multiagent_rl.env import AijMultiagentEnv


def sample_rollouts(
    n_rollouts: int,
    env: AijMultiagentEnv,
    agents: Dict[str, BaseAgent]
) -> Tuple[List[List[Dict[str, Any]]], float]:
    rollouts = []
    action_times = 0
    for _ in range(n_rollouts):
        rollout = []
        for agent in agents.values():
            agent.reset_state()
        observations, infos = env.reset()
        done = False
        while not done:
            start = time.perf_counter()
            actions = {name: agent.get_action(observation=observations[name])
                       for name, agent in agents.items() if name in env.agents}
            end = time.perf_counter()
            action_times += (end-start)
            next_observations, rewards, terminations, truncations, next_infos = env.step(actions)
            transition = {
                'observations': observations,
                'next_observations': next_observations,
                'actions': actions,
                'rewards': rewards,
                'terminations': terminations,
                'truncations': truncations
            }
            observations = next_observations
            done = all(truncations.values()) or all(terminations.values())
            rollout.append(transition)
        rollouts.append(rollout)
    action_time = action_times / (sum([len(e) for e in rollouts]) * 8)
    return rollouts, action_time


def compute_average_cumulative_reward(rollouts: List[List[Dict[str, Any]]], agents: Dict[str, BaseAgent]) -> float:
    total_cumulative_reward = 0.0
    total_agents = len(agents)
    total_rollouts = len(rollouts)

    for rollout in rollouts:
        for transition in rollout:
            rewards = transition['rewards']
            # Sum up rewards for all agents in this transition
            total_cumulative_reward += sum(rewards.values())

    # Calculate the average cumulative reward per agent
    # We multiply the number of rollouts by the number of agents to get the total count for averaging
    if total_rollouts > 0:
        average_cumulative_reward_per_agent = total_cumulative_reward / (total_rollouts * total_agents)
    else:
        average_cumulative_reward_per_agent = 0.0  # Prevent division by zero

    return average_cumulative_reward_per_agent



def compute_score_from_env(n_rollouts: int, agents: Dict[str, BaseAgent]) -> float:
    # Sample rollouts from the environment with the agents
    env = AijMultiagentEnv()
    rollouts, action_time = sample_rollouts(n_rollouts, env, agents)

    # Compute the average cumulative reward per agent from the rollouts
    average_reward = compute_average_cumulative_reward(rollouts, agents)

    return average_reward


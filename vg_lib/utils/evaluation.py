import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from aij_multiagent_rl.agents import BaseAgent
from aij_multiagent_rl.env import AijMultiagentEnv
import cv2


def sample_rollouts(
    n_rollouts: int,
    env: AijMultiagentEnv,
    agents: Dict[str, BaseAgent],
    video_path: Optional[str] = None,
    max_n_steps: int = 1000
) -> Tuple[List[List[Dict[str, Any]]], float]:
    rollouts = []
    action_times = 0

    # Video writer if video_path is provided
    video_writer = None
    if video_path is not None:
        video_writer = None

    for i_rollout in range(n_rollouts):
        rollout = []
        for agent in agents.values():
            agent.reset_state()
        observations, infos = env.reset()
        done = False

        with tqdm(total=max_n_steps, desc=f'eval rollout {i_rollout}') as pbar:
            while not done:
                start = time.perf_counter()
                actions = {name: agent.get_action(observation=observations[name])
                           for name, agent in agents.items() if name in env.agents}
                end = time.perf_counter()
                action_times += (end - start)

                next_observations, rewards, terminations, truncations, next_infos = env.step(actions)

                transition = {
                    'observations': observations,
                    'next_observations': next_observations,
                    'actions': actions,
                    'rewards': rewards,
                    'terminations': terminations,
                    'truncations': truncations,
                    'state': env.state(),
                }

                # Check if video_path is provided and save the current frame (state['image'])
                if video_path is not None and 'image' in transition['state']:
                    image = transition['state']['image']
                    # Convert 'image' to a numpy array if it's not already one
                    frame = np.array(image)

                    # Initialize the VideoWriter if it's the first frame
                    if video_writer is None:
                        height, width, layers = frame.shape
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
                        video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))

                    # Write the current frame to the video
                    video_writer.write(frame)

                observations = next_observations
                done = all(truncations.values()) or all(terminations.values())
                rollout.append(transition)

                pbar.update()

        rollouts.append(rollout)

    # Release the video writer when done if it was initialized
    if video_writer is not None:
        video_writer.release()

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


def compute_score_from_env(
        n_rollouts: int,
        agents: Dict[str, BaseAgent],
        video_path: Optional[str] = None,
        make_env_fn = AijMultiagentEnv,
        max_n_steps = 1000
        ) -> float:
    env = make_env_fn()
    # Sample rollouts from the environment with the agents
    rollouts, action_time = sample_rollouts(n_rollouts, env, agents, video_path, max_n_steps)

    # Compute the average cumulative reward per agent from the rollouts
    average_reward = compute_average_cumulative_reward(rollouts, agents)

    return average_reward


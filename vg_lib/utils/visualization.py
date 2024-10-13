import cv2
import numpy as np
from typing import Dict
from aij_multiagent_rl.agents import BaseAgent
from copy import deepcopy

def visualize_global_rollout(env, agents: Dict[str, BaseAgent], rollouts=1, video_name='global_rollout.mp4'):
    """
    Creates a global video visualization of the environment using env.state(), with agents mapped by names.

    Parameters:
    - env: The environment object.
    - agents: A dictionary mapping agent names to BaseAgent instances.
    - rollouts: Number of rollout episodes to visualize.
    - video_name: The name of the saved video file.
    """
    height, width, _ = env.state()['image'].shape
    video_writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    for _ in range(rollouts):
        obs, infos = env.reset()
        done = {agent_id: False for agent_id in agents}

        while not all(done.values()):
            actions = {}
            for agent_id, agent in agents.items():
                action = agent.get_action(obs[agent_id])  # Get action from the agent
                actions[agent_id] = action

            obs, rewards, terminations, truncations, infos = env.step(actions)
            global_image = env.state()['image']  # Global image from the environment's state
            video_writer.write(global_image)

            done.update({agent_id: terminations[agent_id] or truncations[agent_id] for agent_id in agents})

    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Global rollout video saved as {video_name}")


def write_vid(obs_list, video_name='global_rollout.mp4'):
    """
    Creates a global video visualization of the environment using env.state(), with agents mapped by names.

    Parameters:
    - env: The environment object.
    - agents: A dictionary mapping agent names to BaseAgent instances.
    - rollouts: Number of rollout episodes to visualize.
    - video_name: The name of the saved video file.
    """
    agents = sorted(list(obs_list[0].keys()))
    pad_obs = {a: {'image': np.zeros_like(obs_list[0][a]['image'][0])} for a in agents}

    img = np.concatenate([pad_obs[a]['image'] for a in agents])
    height, width, _ = img.shape
    video_writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    for obs in (obs_list):
        _obs = deepcopy(pad_obs)
        _obs.update(obs)
        _img = np.concatenate([_obs[a]['image'][0] for a in agents])
        video_writer.write(_img)


    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Global rollout video saved as {video_name}")

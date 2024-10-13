import random
import multiprocessing as mp
from typing import Type, List, Tuple, Dict, Any
import numpy as np
from aij_multiagent_rl.env import AijMultiagentEnv, ParallelEnv

# Helper function to create and run an environment in a separate process
def worker_process(env_cls: Type[ParallelEnv], conn, env_idx, env_kwargs, seed):
    # Set a unique seed for each process using the combination of base seed and env_idx
    unique_seed = seed + env_idx
    np.random.seed(unique_seed)
    random.seed(unique_seed)

    env = env_cls(**env_kwargs)

    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            actions = data
            obs, rewards, dones, truncs, infos = env.step(actions)

            # Get all agent IDs from 'dones' and 'truncs' keys
            agent_ids = set(dones.keys()).union(set(truncs.keys()))

            # Automatically reset the environment if all agents are done or truncated
            all_done = all(
                dones.get(agent_id, False) or truncs.get(agent_id, False)
                for agent_id in agent_ids
            )

            if all_done:
                # Reset the environment and get the new observations
                reset_obs = env.reset()
                # Update the observations with the reset observations
                obs.update(reset_obs)
                # Optionally, indicate in 'infos' that a reset has occurred
                for agent_id in infos:
                    infos[agent_id]['env_reset'] = True

            conn.send((obs, rewards, dones, truncs, infos))

        elif cmd == "reset":
            conn.send(env.reset())  # Send reset results
        elif cmd == "close":
            conn.close()
            break
        elif cmd == "render":
            conn.send(env.render())  # Send render result
        elif cmd == "state":
            conn.send(env.state())  # Send global state result

class VectorParallelEnv:
    def __init__(self, env_cls: Type[ParallelEnv], num_envs: int, env_kwargs: Dict[str, Any] = None, seed: int = 0):
        self.env_cls = env_cls
        self.num_envs = num_envs
        self.env_kwargs = env_kwargs if env_kwargs is not None else {}
        self.seed = seed
        self.envs = []
        self.conns = []
        self.worker_conns = []

        # Start the worker processes
        for i in range(num_envs):
            parent_conn, child_conn = mp.Pipe()
            self.conns.append(parent_conn)
            self.worker_conns.append(child_conn)
            process = mp.Process(
                target=worker_process,
                args=(env_cls, child_conn, i, self.env_kwargs, seed)
            )
            process.daemon = True  # Ensure the process ends when the main process exits
            process.start()
            self.envs.append(process)

    def step(self, actions: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], List[Dict]]:
        assert len(self.conns) == len(actions)
        for conn, action in zip(self.conns, actions):
            conn.send(("step", action))
        results = [conn.recv() for conn in self.conns]
        obs, rewards, dones, truncs, infos = zip(*results)
        return list(obs), list(rewards), list(dones), list(truncs), list(infos)

    def reset(self) -> List[Dict[str, Any]]:
        for conn in self.conns:
            conn.send(("reset", None))
        return [conn.recv() for conn in self.conns]

    def render(self) -> List[Any]:
        for conn in self.conns:
            conn.send(("render", None))
        return [conn.recv() for conn in self.conns]

    def state(self) -> List[np.ndarray]:
        for conn in self.conns:
            conn.send(("state", None))
        return [conn.recv() for conn in self.conns]

    def close(self):
        for conn in self.conns:
            conn.send(("close", None))
        for env in self.envs:
            env.join()

# Utility function to create a vectorized environment
def create_vec_env(env_cls: Type[ParallelEnv], num_envs: int, env_kwargs: Dict[str, Any] = None, seed: int = 0) -> VectorParallelEnv:
    return VectorParallelEnv(env_cls=env_cls, num_envs=num_envs, env_kwargs=env_kwargs, seed=seed)

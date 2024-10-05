import numpy as np
from collections import deque
from gymnasium import Wrapper


class FrameStack(Wrapper):
    def __init__(self, env, stack_size: int = 4, keys: list = ["image"]):
        """FrameStack wrapper for multi-agent environments.

        Args:
            env: The multi-agent environment to wrap.
            stack_size: The number of frames to stack.
            keys: A list of observation keys to stack.
        """
        super(FrameStack, self).__init__(env)
        self.stack_size = stack_size
        self.keys = keys
        self.frames = {
            key: {agent: deque(maxlen=stack_size) for agent in env.agents}
            for key in keys
        }

    def reset(self, **kwargs):
        for k in self.frames:
            for agent in self.frames[k]:
                self.frames[k][agent].clear()
        observations, infos = self.env.reset(**kwargs)
        stacked_observations = self._stack_frames(observations)
        return stacked_observations, infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        stacked_observations = self._stack_frames(observations)
        return stacked_observations, rewards, terminations, truncations, infos

    def _stack_frames(self, observations):
        """Stack the frames for the selected keys.

        Arg:
            observations: The current observations from the environment.

        Returns:
            A dictionary of stacked observations for specified keys,
            maintaining the structure of the original observations.
        """
        stacked_observations = {}
        for agent in observations.keys():
            if agent not in stacked_observations:
                stacked_observations[agent] = {}

            # Stack specified keys
            for key in self.keys:
                if key in observations[agent]:
                    # Append the current observation to the deque
                    self.frames[key][agent].append(observations[agent][key])
                    # Pad with the last frame if necessary
                    if len(self.frames[key][agent]) < self.stack_size:
                        last_frame = self.frames[key][agent][-1]
                        # Pad with the last frame until we reach stack_size
                        while len(self.frames[key][agent]) < self.stack_size:
                            self.frames[key][agent].append(last_frame)

                    stacked_observations[agent][key] = np.array(self.frames[key][agent])

            # Add all other keys from the original observations that are not stacked
            for key in observations[agent].keys():
                if key not in self.keys:
                    stacked_observations[agent][key] = observations[agent][key]

        return stacked_observations

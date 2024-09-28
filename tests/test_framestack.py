import pytest
import numpy as np
from collections import deque
from unittest.mock import MagicMock
from vg_lib.env.framestack import FrameStack  # Assumed relative import for the `FrameStack` class.
import numpy as np
from gym import Env
from collections import deque

class MockParallelEnv(Env):
    """
    A simple mock of a multi-agent ParallelEnv that returns different observations
    at each step for the purpose of testing FrameStack.
    """
    def __init__(self, agents, obs_sequence):
        super(MockParallelEnv, self).__init__()
        self.agents = agents
        self.obs_sequence = obs_sequence
        self.current_step = 0

    def reset(self):
        """Resets the environment and returns the initial observation for each agent."""
        self.current_step = 0
        observations = {agent: self.obs_sequence[agent][self.current_step]
                        for agent in self.agents}
        return observations, {}

    def step(self, actions):
        """
        Moves to the next step in the mock environment. If step exceeds the max number
        of steps, it repeats the last observation.
        """
        # Cap the current_step to not go beyond the available observation sequence.
        if self.current_step < len(next(iter(self.obs_sequence.values()))) - 1:
            self.current_step += 1

        observations = {agent: self.obs_sequence[agent][self.current_step]
                        for agent in self.agents}
        return observations, {}, {}, {}, {}  # Dummy reward, terminations, truncations, infos

    def render(self, mode='human'):
        pass

    def close(self):
        pass


@pytest.fixture
def mock_env():
    """
    This fixture sets up a mock environment for two agents with sequential observations.
    The observations for each agent change at each step.
    """
    agents = ['agent_0', 'agent_1']

    # Define sequential observations for each agent across 3 steps
    obs_sequence = {
        'agent_0': [
            {'image': np.array([[0, 0], [0, 0]])},  # Step 0
            {'image': np.array([[1, 1], [1, 1]])},  # Step 1
            {'image': np.array([[2, 2], [2, 2]])},  # Step 2
        ],
        'agent_1': [
            {'image': np.array([[3, 3], [3, 3]])},  # Step 0
            {'image': np.array([[4, 4], [4, 4]])},  # Step 1
            {'image': np.array([[5, 5], [5, 5]])},  # Step 2
        ]
    }

    # Create the mock environment
    env = MockParallelEnv(agents, obs_sequence)

    # Wrap it in the FrameStack with `stack_size = 3`
    framestack_env = FrameStack(env, stack_size=3, keys=['image'])

    return framestack_env, obs_sequence


def test_framestack_first_reset(mock_env):
    """
    Test that after reset, the observation is stacked correctly by repeating
    the initial frame for the stack_size.
    """
    framestack_env, obs_sequence = mock_env

    # Reset the env, and check if frames are stacked with padding (repeating initial obs)
    stacked_obs, _ = framestack_env.reset()

    for agent in framestack_env.env.agents:
        expected_stack = np.stack([obs_sequence[agent][0]['image']] * 3)
        np.testing.assert_array_equal(stacked_obs[agent]['image'], expected_stack)


def test_framestack_after_first_step(mock_env):
    """
    Test that after the first step, stacking includes the current and previous step observations.
    """
    framestack_env, obs_sequence = mock_env

    framestack_env.reset()

    # Take the first step, expect the new observation to be added and old frames kept
    stacked_obs, _, _, _, _ = framestack_env.step({'agent_0': 0, 'agent_1': 0})

    expected_agent_0 = np.stack([
        obs_sequence['agent_0'][0]['image'],  # Step 0 (initial)
        obs_sequence['agent_0'][0]['image'],  # Step 0 (still padded)
        obs_sequence['agent_0'][1]['image'],  # Step 1 (new)
    ])
    np.testing.assert_array_equal(stacked_obs['agent_0']['image'], expected_agent_0)

    expected_agent_1 = np.stack([
        obs_sequence['agent_1'][0]['image'],
        obs_sequence['agent_1'][0]['image'],
        obs_sequence['agent_1'][1]['image'],
    ])
    np.testing.assert_array_equal(stacked_obs['agent_1']['image'], expected_agent_1)


def test_framestack_second_step(mock_env):
    """
    Test that after two steps, stacking includes the last three observations.
    """
    framestack_env, obs_sequence = mock_env

    framestack_env.reset()
    framestack_env.step({'agent_0': 0, 'agent_1': 0})  # First step
    stacked_obs, _, _, _, _ = framestack_env.step({'agent_0': 0, 'agent_1': 0})  # Second step

    expected_agent_0 = np.stack([
        obs_sequence['agent_0'][0]['image'],  # Step 0
        obs_sequence['agent_0'][1]['image'],  # Step 1
        obs_sequence['agent_0'][2]['image'],  # Step 2 (new)
    ])
    np.testing.assert_array_equal(stacked_obs['agent_0']['image'], expected_agent_0)

    expected_agent_1 = np.stack([
        obs_sequence['agent_1'][0]['image'],  # Step 0
        obs_sequence['agent_1'][1]['image'],  # Step 1
        obs_sequence['agent_1'][2]['image'],  # Step 2 (new)
    ])
    np.testing.assert_array_equal(stacked_obs['agent_1']['image'], expected_agent_1)




def test_framestack_third_step(mock_env):
    """
    After the third step, we expect that the frame stack stays at the last available observation
    and doesn't move beyond the available steps in the environment.
    """
    framestack_env, obs_sequence = mock_env

    # Reset the environment (put Step 0 into all stack positions)
    framestack_env.reset()

    # First step -> Stack should now be: [Step 0, Step 0, Step 1]
    framestack_env.step({'agent_0': 0, 'agent_1': 0})

    # Second step -> Stack should now be: [Step 0, Step 1, Step 2]
    framestack_env.step({'agent_0': 0, 'agent_1': 0})

    # Third step -> Stack should contain only Step 1 and Step 2:
    #              [Step 1, Step 2, Step 2] (Step 0 is pushed out)
    stacked_obs, _, _, _, _ = framestack_env.step({'agent_0': 0, 'agent_1': 0})  # Third step (edge case)

    # Expected stacks for 'agent_0'
    expected_agent_0_stack = np.stack([
        obs_sequence['agent_0'][1]['image'],  # Step 1
        obs_sequence['agent_0'][2]['image'],  # Step 2
        obs_sequence['agent_0'][2]['image'],  # Step 2 (repeated on current step)
    ])
    np.testing.assert_array_equal(stacked_obs['agent_0']['image'], expected_agent_0_stack)

    # Expected stacks for 'agent_1'
    expected_agent_1_stack = np.stack([
        obs_sequence['agent_1'][1]['image'],  # Step 1
        obs_sequence['agent_1'][2]['image'],  # Step 2
        obs_sequence['agent_1'][2]['image'],  # Step 2 (repeated on current step)
    ])
    np.testing.assert_array_equal(stacked_obs['agent_1']['image'], expected_agent_1_stack)


def test_framestack_resets_after_multiple_steps(mock_env):
    """
    Test that after taking multiple steps, if reset() is called, the frame stack is correctly cleared
    and filled with the initial observation for each agent, repeated 'stack_size' times.
    """
    framestack_env, obs_sequence = mock_env

    # Step 0 (Reset should initialize the stack with Step 0 observations)
    stacked_obs, _ = framestack_env.reset()

    ### Step Forward Twice ###

    # Step 1 -> Take First Step (stack updates with Step 1)
    framestack_env.step({'agent_0': 0, 'agent_1': 0})

    # Step 2 -> Take Second Step (stack updates with Step 2)
    framestack_env.step({'agent_0': 0, 'agent_1': 0})

    ### Now Call Reset ###

    # Reset the stack, should clear out any steps, and reinitialize with Step 0 observations
    stacked_obs, _ = framestack_env.reset()

    # After reset, for both agents, the stack should contain only Step 0 repeated stack_size times.
    for agent in framestack_env.env.agents:
        expected_stack = np.stack([obs_sequence[agent][0]['image']] * framestack_env.stack_size)
        np.testing.assert_array_equal(stacked_obs[agent]['image'], expected_stack)

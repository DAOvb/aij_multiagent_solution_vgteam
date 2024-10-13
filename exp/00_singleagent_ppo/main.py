from absl import app
from absl import flags
from ml_collections import config_flags
import tensorflow as tf
from ppo_lib import train
from flax.training import train_state
import os

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'workdir',
    default='/tmp/ppo_training',
    help='Directory to save checkpoints and logging info.',
)

config_flags.DEFINE_config_file(
    'config',
    'configs/default.py',
    'File path to the default configuration file.',
    lock_config=True,
)


def main(argv):
    from config import get_config
    from agent import ActorCritic, TheFramestackPolicy
    from vg_lib.utils.evaluation import compute_score_from_env
    from env import AijMultiagentEnv
    config = get_config()


    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], 'GPU')
    model = ActorCritic(num_outputs=8)


    def eval_train_state_fn(state: train_state.TrainState, step):
        policies = {f'agent_{i}': TheFramestackPolicy(state) for i in range(8)}

        # return {'score': -1}
        video_name = os.path.join(FLAGS.workdir, f"{step:05}.mp4")
        return {'score': compute_score_from_env(1, policies, video_path=video_name, make_env_fn=AijMultiagentEnv)}

    train(
        model, config, FLAGS.workdir,
        eval_train_state_fn=eval_train_state_fn
    )

if __name__ == '__main__':
    app.run(main)

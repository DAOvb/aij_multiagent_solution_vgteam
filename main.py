from aij_multiagent_rl.env import AijMultiagentEnv
import yaml
import flax.linen as nn
import numpy as np
import jax
from vg_lib.modules import nets
from vg_lib.utils.training import *
from vg_lib.modules.loss import actor_loss_fn, critic_loss_fn

# jax.config.update("jax_traceback_filtering", "off")


def make_train(config):
    env = AijMultiagentEnv()
    initial_agents_state, initial_area_state = env.reset()
    initial_world_state = env.state()
    config["NUM_ACTORS"] = NUM_AGENTS * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )
    rng = jax.random.PRNGKey(config["SEED"])
    rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
    obsv, area = env.reset(seed=rng[0].item())
    env_state = env.state()

    def train(rng):
        # gather agent initial state
        actor_init_image = jnp.zeros(
            (1, config["NUM_ENVS"], *initial_agents_state[AGENT_KEYS[0]]["image"].shape)
        )
        actor_init_proprio = jnp.zeros(
            (
                1,
                config["NUM_ENVS"],
                *initial_agents_state[AGENT_KEYS[0]]["proprio"].shape,
            )
        )
        actor_init_obs = (actor_init_image, actor_init_proprio)
        actor_init_x = (
            actor_init_obs,
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        # create actor
        actor = nets.ActorRNN(env.action_space(AGENT_KEYS[0]).n, config=config)
        ac_init_hstate = nets.ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        actor_network_params = actor.init(_rng_actor, ac_init_hstate, actor_init_x)

        # gather critic initial state
        image, feats = process_world_state(initial_world_state, initial_area_state)
        critic_image = jnp.zeros((1, config["NUM_ENVS"], *image.shape))
        critic_feats = feats.reshape(1, config["NUM_ENVS"], -1)
        cr_init_x = (
            (critic_image, critic_feats),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        # create critic
        critic = nets.CriticRNN(config=config)
        cr_init_hstate = nets.ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        critic_network_params = critic.init(_rng_critic, cr_init_hstate, cr_init_x)
        # create train states
        actor_train_state = create_train_state(actor, actor_network_params, config)
        critic_train_state = create_train_state(critic, critic_network_params, config)
        # initialize actor and critic with starting parameters
        rng, _rng = jax.random.split(rng)
        ac_init_hstate = nets.ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )
        cr_init_hstate = nets.ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )

        def _update_step(update_runner_state, _):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            # method to collect experience from environment
            def _env_step(runner_state, _):
                (
                    train_states,
                    env_state,
                    last_obs,
                    last_area,
                    last_done,
                    hstates,
                    rng,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_p_batch = batchify(last_obs, "proprio", config)
                obs_i_batch = batchify_image(last_obs, config)
                ac_in = (
                    (obs_i_batch[np.newaxis, :], obs_p_batch[np.newaxis, :]),
                    last_done[np.newaxis, :],
                )
                ac_hstate, pi = actor.apply(train_states[0].params, hstates[0], ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify_acts(action, config["NUM_ENVS"])
                # VALUE
                # output of wrapper is (num_envs, num_agents, world_state_size)
                # swap axes to (num_agents, num_envs, world_state_size) before reshaping to (num_actors, world_state_size)
                world_image, world_feats = process_world_state(env_state, last_area)
                world_feats = world_feats.reshape((config["NUM_ENVS"], -1))
                world_image = world_image.reshape(
                    (config["NUM_ENVS"], *world_image.shape[-3:])
                )
                cr_in = (
                    (world_image[None, :], world_feats[None, :]),
                    last_done[np.newaxis, :],
                )
                cr_hstate, value = critic.apply(
                    train_states[1].params, hstates[1], cr_in
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, reward, truncated, terminated, info = env.step(env_act)

                area = info
                done = jnp.logical_or(truncated, terminated)
                env_state = env.state()
                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, NUM_AGENTS, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_i_batch,
                    obs_p_batch,
                    world_image,
                    world_feats,
                    info,
                )
                runner_state = (
                    train_states,
                    env_state,
                    obsv,
                    area,
                    done_batch,
                    (ac_hstate, cr_hstate),
                    rng,
                )
                return runner_state, transition

            initial_hstates = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_done, hstates, rng = runner_state

            world_image, world_feats = process_world_state(env_state)
            world_feats = world_feats.reshape((config["NUM_ACTORS"], -1))
            world_image = world_image.reshape(
                (config["NUM_ACTORS"], *world_image.shape[-3:])
            )
            cr_in = (
                (world_image[None, :], world_feats[None, :]),
                last_done[np.newaxis, :],
            )
            _, last_val = critic.apply(train_states[1].params, hstates[1], cr_in)
            last_val = last_val.squeeze()
            advantages, targets = scan_adv(
                traj_batch, last_val, config["GAMMA"], config["GAE_LAMBDA"]
            )

            def _update_epoch(update_state, _):
                def _update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = (
                        batch_info
                    )

                    actor_grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params, ac_init_hstate, traj_batch, advantages
                    )
                    critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params, cr_init_hstate, traj_batch, targets
                    )
                    actor_train_state = actor_train_state.apply_gradients(
                        grads=actor_grads
                    )
                    critic_train_state = critic_train_state.apply_gradients(
                        grads=critic_grads
                    )

                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                    }
                    return (actor_train_state, critic_train_state), loss_info

                train_states, init_hstates, traj_batch, advantages, targets, rng = (
                    update_state
                )
                rng, _rng = jax.random.split(rng)
                init_hstates = jax.tree_map(
                    lambda x: jnp.reshape(x, (1, config["NUM_ACTORS"], -1)),
                    init_hstates,
                )
                batch = (
                    init_hstates[0],
                    init_hstates[1],
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )
                train_states, loss_info = jax.lax.scan(
                    _update_minbatch, train_states, minibatches
                )
                update_state = (
                    train_states,
                    jax.tree_map(lambda x: x.squeeze(), init_hstates),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_states,
                initial_hstates,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, config["UPDATE_EPOCHS"]
            )
            loss_info["ratio_0"] = loss_info["ratio"].at[0, 0].get()
            loss_info = jax.tree_map(lambda x: x.mean(), loss_info)

            train_states = update_state[0]
            metric = traj_batch.info
            metric["loss"] = loss_info
            rng = update_state[-1]

            # def callback(metric):

            #     wandb.log(
            #         {
            #             "returns": metric["returned_episode_returns"][-1, :].mean(),
            #             "env_step": metric["update_steps"]
            #             * config["NUM_ENVS"]
            #             * config["NUM_STEPS"],
            #             **metric["loss"],
            #         }
            #     )

            metric["update_steps"] = update_steps
            # jax.experimental.io_callback(callback, None, metric)
            update_steps = update_steps + 1
            runner_state = (train_states, env_state, last_obs, last_done, hstates, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            area,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train


def main():
    with open("config.yaml") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    # wandb.init(
    #     entity=config["ENTITY"],
    #     project=config["PROJECT"],
    #     tags=["MAPPO", "RNN", config["ENV_NAME"]],
    #     config=config,
    #     mode=config["WANDB_MODE"],
    # )
    rng = jax.random.PRNGKey(config["SEED"])
    with jax.disable_jit(False):
        train_jit = jax.jit(make_train(config))
        out = train_jit(rng)


if __name__ == "__main__":
    main()

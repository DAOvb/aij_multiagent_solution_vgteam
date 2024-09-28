import jax
import jax.numpy as jnp

def actor_loss_fn(actor_params, actor_network, init_hstate, traj_batch, gae, clip_eps, ent_cf):
    # RERUN NETWORK
    _, pi = actor_network.apply(
        actor_params,
        init_hstate.squeeze(),
        (traj_batch.obs, traj_batch.done),
    )
    log_prob = pi.log_prob(traj_batch.action)

    # CALCULATE ACTOR LOSS
    logratio = log_prob - traj_batch.log_prob
    ratio = jnp.exp(logratio)
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = (
        jnp.clip(
            ratio,
            1.0 - clip_eps,
            1.0 + clip_eps,
        )
        * gae
    )
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()
    entropy = pi.entropy().mean()
    
    # debug
    approx_kl = ((ratio - 1) - logratio).mean()
    clip_frac = jnp.mean(jnp.abs(ratio - 1) > clip_eps)
    
    actor_loss = (
        loss_actor
        - ent_cf * entropy
    )
    return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)

def critic_loss_fn(critic_params, critic_network, init_hstate, traj_batch, targets, clip_eps, vf_cf):
    # RERUN NETWORK
    _, value = critic_network.apply(critic_params, init_hstate.squeeze(), (traj_batch.world_state,  traj_batch.done)) 
    
    # CALCULATE VALUE LOSS
    value_pred_clipped = traj_batch.value + (
        value - traj_batch.value
    ).clip(-clip_eps, clip_eps)
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = (
        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    )
    critic_loss = vf_cf * value_loss
    return critic_loss, (value_loss)
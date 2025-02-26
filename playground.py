import equinox as eqx
import jax
import jax.numpy as jnp
from jaxvmas.make_env import make_env
from jaxvmas.simulator.environment.environment import RenderObject

# Create a random key for initialization
key = jax.random.PRNGKey(0)

# Create vectorized environments
num_envs = 32
env = make_env(
    scenario="football",  # or "simple" from MPE scenarios
    num_envs=num_envs,
    PRNG_key=key,
    continuous_actions=True,
)
n_steps = 100

# Reset environment
env, obs = env.reset(PRNG_key=key)

actions = [None] * len(obs)
for i in range(len(obs)):
    n_envs = obs[i].shape[0]
    actions[i] = jnp.zeros((n_envs, 2))


render_object = RenderObject()
total_reward = 0
step = 0
for _ in range(n_steps):
    PRNG_key, key_step = jax.random.split(key)
    step += 1
    actions = [None] * len(obs)
    for i in range(len(obs)):
        key_step, key_step_i = jax.random.split(key_step)
        actions[i] = jnp.zeros((n_envs, 2))

    jitted_step = eqx.filter_jit(env.step)
    PRNG_key, key_step_i = jax.random.split(PRNG_key)
    env, (obs, rews, dones, info) = jitted_step(PRNG_key=key_step_i, actions=actions)

    rewards = jnp.stack(rews, axis=1)
    global_reward = rewards.mean(axis=1)
    mean_global_reward = global_reward.mean(axis=0)
    total_reward += mean_global_reward
    render_object, rgb_array = env.render(
        render_object=render_object,
        mode="rgb_array",
        agent_index_focus=None,
        visualize_when_rgb=True,
    )

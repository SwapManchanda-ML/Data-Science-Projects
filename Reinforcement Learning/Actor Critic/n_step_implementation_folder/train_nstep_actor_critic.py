import gym
import numpy as np
import time
from tqdm import tqdm

from rl2020.exercise4.n_step_implementation_folder.agents import Reinforce
from rl2020.exercise4.n_step_implementation_folder.networks import FCNetwork


LUNARLANDER_CONFIG = {
    "env": "LunarLander-v2",
    "episode_length": 250,
    "target_return": 190.0,
    "max_timesteps": 1e6,
    "eval_freq": 50000,
    "eval_episodes": 40,
    "max_time": 60 * 60,
    "gamma": 0.99,
    "hidden_size": (32,64,), #(32,64),
    "learning_rate": 10e-5, #0.001
    "save_filename": "reinf_lunarlander_latest.pt",
}

CARTPOLE_CONFIG = {
    "env": "CartPole-v1",
    "episode_length": 250,
    "target_return": 195.0,
    "max_timesteps": 300000,
    "eval_freq": 2000,
    "eval_episodes": 20,
    "max_time": 30 * 60,
    "gamma": 0.99,
    "hidden_size": (32,),
    "learning_rate": 0.001,
    "save_filename": None,
}

#CONFIG = CARTPOLE_CONFIG
CONFIG = LUNARLANDER_CONFIG

RENDER = False


def play_episode(
    env: gym.Env,
    agent: Reinforce,
    train: bool = True,
    explore=True,
    render=False,
    max_steps=200,
):
    """
    Play out a single episode with REINFORCE

    :param env (gym.Env): gym environment to use
    :param agent (Reinforce): REINFORCE agent to train
    :param train (bool): flag whether training should be executed
    :param explore (bool): flag whether agent should use exploration
    :param render (bool): flag whether environment steps should be rendered
    :param max_steps (int): maximum number of steps to take for the episode
    :return (Tuple[int, float]): number of timesteps completed during episode, episode return
    """
    obs = env.reset()

    if render:
        env.render()

    done = False
    num_steps = 0
    total_rewards = 0

    observations = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    critic_states = []
    p_update = 0

    while not done and num_steps < max_steps:
        action,critic_state,log_prob = agent.act(np.array(obs), explore=explore)
        nobs, rew, done, _ = env.step(action)

        observations.append(obs)
        actions.append(action)
        rewards.append(rew)
        dones.append(done)

        log_probs.append(log_prob)
        critic_states.append(critic_state)
        
        if render:
            env.render()

        num_steps += 1
        total_rewards += rew

        obs = nobs

        if done or (num_steps - p_update ==100):
            loss = agent.update(log_probs,critic_states,rewards, dones, nobs)
            p_update = num_steps

            rewards = []
            log_probs = []
            critic_states = []
            dones = []

    return num_steps, total_rewards


def train(env, config, output=True):
    """
    Execute training of REINFORCE on given environment using the provided configuration

    :param env (gym.Env): environment to train on
    :param config: configuration dictionary mapping configuration keys to values
    :param output (bool): flag whether evaluation results should be printed
    :return (Tuple[List[float], List[float]]): mean average returns during training, times of evaluation
    """
    timesteps_elapsed = 0

    agent = Reinforce(
        action_space=env.action_space, observation_space=env.observation_space, **config
    )

    total_steps = config["max_timesteps"]
    eval_returns_all = []
    eval_times_all = []

    start_time = time.time()
    with tqdm(total=total_steps) as pbar:
        while timesteps_elapsed < total_steps:
            elapsed_seconds = time.time() - start_time
            if elapsed_seconds > config["max_time"]:
                pbar.write(f"Training ended after {elapsed_seconds}s.")
                break
            agent.schedule_hyperparameters(timesteps_elapsed, total_steps)
            num_steps, _ = play_episode(
                env,
                agent,
                train=True,
                explore=True,
                render=False,
                max_steps=config["episode_length"],
            )
            timesteps_elapsed += num_steps
            pbar.update(num_steps)

            if timesteps_elapsed % config["eval_freq"] < num_steps:
                eval_return = 0
                for _ in range(config["eval_episodes"]):
                    _, total_reward = play_episode(
                        env,
                        agent,
                        train=False,
                        explore=False,
                        render=RENDER,
                        max_steps=env._max_episode_steps,
                    )
                    eval_return += total_reward / (config["eval_episodes"])
                if output:
                    pbar.write(
                        f"Evaluation at timestep {timesteps_elapsed} returned a mean return of {eval_return}"
                    )
                eval_returns_all.append(eval_return)
                eval_times_all.append(time.time() - start_time)
                if eval_return >= config["target_return"]:
                    pbar.write(
                        f"Reached return {eval_return} >= target return of {config['target_return']}"
                    )
                    break

    if config["save_filename"]:
        print("Saving to: ", agent.save(config["save_filename"]))

    return np.array(eval_returns_all), np.array(eval_times_all)


if __name__ == "__main__":
    random_seed = 543
    env = gym.make(CONFIG["env"])
    env.seed(random_seed)
    returns, times = train(env, CONFIG)
    env.close()

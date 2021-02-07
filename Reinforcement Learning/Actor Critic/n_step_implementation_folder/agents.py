from abc import ABC, abstractmethod
from copy import deepcopy
import gym
import numpy as np
import os.path
from torch import Tensor
from torch.distributions.categorical import Categorical
import torch.nn
from torch.optim import Adam
from typing import Dict, Iterable, List

from rl2020.exercise4.n_step_implementation_folder.networks import FCNetwork, ActorCritic
from rl2020.exercise4.n_step_implementation_folder.networks2 import Critic, Actor

class Agent(ABC):
    """Base class for Deep RL Exercise 3 Agents

    **DO NOT CHANGE THIS CLASS**

    Note:
        see http://gym.openai.com/docs/#spaces for more information on Gym spaces
    """

    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        """The constructor of the Agent Class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space

        :attr saveables (Dict[str, torch.nn.Module]):
            mapping from network names to PyTorch network modules
        """
        self.action_space = action_space
        self.observation_space = observation_space

        self.saveables = {}

    def save(self, path: str) -> str:
        """Saves saveable PyTorch models under given path

        The models will be saved in directory found under given path in file "{path}"

        :param path (str): path to directory where to save models
        :return (str): path to file of saved models file
        """
        torch.save(self.saveables, path)
        return path

    def restore(self, save_path: str):
        """Restores PyTorch models from models file given by path

        :param save_path (str): path to file containing saved models
        """
        dirname, _ = os.path.split(os.path.abspath(__file__))
        save_path = os.path.join(dirname, save_path)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    @abstractmethod
    def act(self, obs: np.ndarray):
        """Returns an action to select in given observation

        **DO NOT CHANGE THIS FUNCTION**
        """
        ...

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **DO NOT CHANGE THIS FUNCTION**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def update(self):
        """Updates model parameters

        **DO NOT CHANGE THIS FUNCTION**
        """
        ...


class Reinforce(Agent):
    """ The Reinforce Agent for Ex 3

    **YOU MUST COMPLETE THIS CLASS**
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        gamma: float,
        **kwargs,
    ):
        """
        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param gamma (float): discount rate gamma

        :attr policy (FCNetwork): fully connected actor network for policy
        :attr policy_optim (torch.optim): PyTorch optimiser for policy network
        """
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        self.actor = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=None
        )

        self.critic = FCNetwork(
            (STATE_SIZE, *hidden_size, 1), output_activation=None
        )

        #self.actor =Actor(STATE_SIZE,ACTION_SIZE)
        #self.critic = Critic(STATE_SIZE,ACTION_SIZE)

        self.actor_optim = Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optim = Adam(self.critic.parameters(), lr=learning_rate)
        
        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_acts = gym.spaces.utils.flatdim(action_space)
        self.update_counter = 0
        self.epsilon_start = 0.99
        self.epsilon_end = 0.002
        self.epsilon_decay = 200
        self.gamma_start = 0.99
        self.gamma_end = 0.96
        self.gamma_decay =100
        self.lr_end = 3e-3
        self.lr_start = 1e-2
        # ############################### #
        # WRITE ANY AGENT PARAMETERS HERE #
        # ############################### #
        self.saved_log_probs = []
        self.soft_max = torch.nn.Softmax(dim=-1)
        self.sigmoid = torch.nn.Sigmoid()
        # ###############################################
        self.saveables.update({"actor": self.actor})
        self.saveables.update({"critic": self.critic})

    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):
        """Updates the hyperparameters 

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        #self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.update_counter/ self.epsilon_decay)
        #self.gamma = self.gamma_end + (self.gamma_start - self.gamma_end) * np.exp(-1. * self.update_counter/ self.gamma_decay)
        #self.learning_rate = self.lr_end + (self.lr_start - self.lr_end) * np.exp(-1. * self.update_counter/ self.gamma_decay)
   
        self.update_counter +=1
        

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        Select an action from the model's stochastic policy by sampling a discrete action
        from the distribution specified by the model output

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        state = torch.from_numpy(obs).type(torch.FloatTensor)
        probs = self.actor(state)
        probs = self.soft_max(probs)

        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)

        critic_state = self.critic(state)
        
        return action.item(), critic_state, log_prob
    

    def update(
        self, log_probs, critic_states, rewards ,dones, nobs
    ) -> Dict[str, float]:
        """Update function for REINFORCE

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        :param rewards (List[float]): rewards of episode (from first to last)
        :param observations (List[np.ndarray]): observations of episode (from first to last)
        :param actions (List[int]): applied actions of episode (from first to last)
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
        
        p_loss = 0.0
        critic_loss = 0.0

        #critic_states_rev = critic_states[::-1]
        #values = torch.stack(critic_states_rev)
        values = torch.stack(critic_states)
      
      
        #q_vals = np.zeros((len(log_probs), 1))
        
        nobs = torch.from_numpy(nobs).type(torch.FloatTensor)
        last_q_val = self.critic(nobs)
        
        #rewards_rev = rewards[::-1]
        #critic_states_rev = critic_states[::-1]
        #log_probs_rev = log_probs[::-1]
        #dones_rev = dones[::-1]

        q_val = last_q_val.detach().data.numpy()

        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
   
        for i in range(len(rewards)):
            pos = len(rewards)-1 - i
            #q_val = rewards_rev[i] + self.gamma*q_val*(1.0-dones_rev[i])
            q_val = rewards[pos] + self.gamma*q_val*(1.0-dones[pos])
            #q_vals[len(rewards)-1 - i] = q_val

            #advantage = torch.as_tensor(q_val) - values[i]
            advantage = torch.as_tensor(q_val) - values[pos]
            p_loss = (-torch.as_tensor(log_probs[pos])*advantage.detach())
            critic_loss = advantage.pow(2)
            
            self.actor.train(True)
            p_loss.backward(retain_graph=True)
            self.actor.train(False)

            self.critic.train(True)
            critic_loss.backward(retain_graph=True)
            self.critic.train(False)

        self.actor_optim.step()
        self.critic_optim.step()
        #advantage = torch.as_tensor(q_vals) - values
        
        #critic_loss = advantage.pow(2).mean()
        #self.critic_optim.zero_grad()
        #critic_loss.backward()
        #self.critic_optim.step()

        #p_loss = (-torch.stack(log_probs)*advantage.detach()).mean()
        #self.actor_optim.zero_grad()
        #p_loss.backward()
        #self.actor_optim.step()











        '''
        for t in range(len(rewards)):
            Gt = 0 
            pw = 0
            for r in rewards[t:]:
                #Gt = Gt + self.gamma**pw * r
                Gt = r + (self.gamma)*Gt
                pw = pw + 1
            ret.append(Gt)
        ret = torch.as_tensor(ret, dtype = torch.float32)  
        ret = (ret - ret.mean()) / (ret.std() + 1e-9)

        self.policy_optim.zero_grad()
            
        for i in range(len(ret)):
            state = torch.from_numpy(observations[i]).float().unsqueeze(0)
            action = torch.as_tensor(actions[i], dtype=torch.float32)
            reward = ret[i]
            
            probs = self.policy(state)
            probs = self.soft_max(probs)
            m = Categorical(probs)
            
            loss = -m.log_prob(action) * reward 
            policy_loss1.append(loss)
        



        p_loss = torch.stack(policy_loss1).mean()
        p_loss.backward(retain_graph=True)
        self.policy_optim.step()
      
        self.saved_log_probs = []
        '''

        return {"p_loss": p_loss}

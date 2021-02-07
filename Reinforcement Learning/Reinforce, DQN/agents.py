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

from rl2020.exercise3.networks import FCNetwork
from rl2020.exercise3.replay import Transition, ReplayBuffer


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


class DQN(Agent):
    """The DQN agent for exercise 3

    **YOU MUST COMPLETE THIS CLASS**
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        target_update_freq: int,
        batch_size: int,
        gamma: float,
        **kwargs,
    ):
        """The constructor of the DQN agent class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param target_update_freq (int): update frequency (number of iterations after which the target
            networks should be updated)
        :param batch_size (int): size of sampled batches of experience
        :param gamma (float): discount rate gamma

        :attr critics_net (FCNetwork): fully connected DQN to compute Q-value estimates
        :attr critics_target (FCNetwork): fully connected DQN target network
        :attr critics_optim (torch.optim): PyTorch optimiser for DQN critics_net
        :attr update_counter (int): counter of updates for target network updates
        """
        super().__init__(action_space, observation_space)

        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        self.critics_net = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=None
        )

        self.critics_target = deepcopy(self.critics_net)

        self.critics_optim = Adam(
            self.critics_net.parameters(), lr=learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.critics_optim, mode='max', factor=0.1, patience=5, verbose=True)
        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.update_counter = 0
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = 0.99
        self.epsilon_end = 0.05
        self.epsilon_decay = 200
        self.n_acts = gym.spaces.utils.flatdim(action_space)
        self.TAU = 1e-3
        # ######################################### #
        self.saveables.update(
            {
                "critics_net": self.critics_net,
                "critics_target": self.critics_target,
                "critic_optim": self.critics_optim,
            }
        )

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
       
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.update_counter/ self.epsilon_decay)
        self.update_counter +=1

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        When explore is False you should select the best action possible (greedy). However, during
        exploration, you should be implementing an exploration strategy

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore (or act greedily)
        :return (sample from self.action_space): action the agent should perform
        """

        state = torch.from_numpy(obs).float().unsqueeze(0)
        self.critics_net.eval()
        with torch.no_grad():
            action = self.critics_net(state)
        self.critics_net.train()

        if explore:
            if np.random.random(1)[0] > self.epsilon:
                return np.argmax(action.cpu().data.numpy())
            else:
                return np.random.choice(np.arange(self.n_acts))
            
        else:
            return np.argmax(action.cpu().data.numpy())


    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for DQN

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**
        
        This function is called after storing a transition in the replay buffer. This happens
        every timestep. It should update your network and return the Q-loss in the form of a
        dictionary.

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to value of losses
        """
        q_loss = 0.0

        state, action, next_state, reward, done  = batch

        action = torch.as_tensor(action, dtype=torch.long)
 
        criterion = torch.nn.MSELoss()

        self.critics_net.train()
        self.critics_target.eval()

        predicted_targets = self.critics_net(state).gather(1,action)
       
        with torch.no_grad():
            labels_next = self.critics_target(next_state).detach().max(1)[0].unsqueeze(1)

        labels = reward + (self.gamma*(1-done)*labels_next)
        
        q_loss = criterion(predicted_targets,labels)
        
        self.critics_optim.zero_grad()
        q_loss.backward()
        self.critics_optim.step()
        
        self.critics_target.soft_update(self.critics_net,self.TAU)
        
        return {"q_loss": q_loss}


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
        self.policy = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=None
        )

        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)
        
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
        self.saveables.update({"policy": self.policy})

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
        probs = self.policy(state)
        probs = self.soft_max(probs)
        m = Categorical(probs)
        action = m.sample()

        return action.item()

    def update(
        self, rewards: List[float], observations: List[np.ndarray], actions: List[int]
    ) -> Dict[str, float]:
        """Update function for REINFORCE

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        :param rewards (List[float]): rewards of episode (from first to last)
        :param observations (List[np.ndarray]): observations of episode (from first to last)
        :param actions (List[int]): applied actions of episode (from first to last)
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
        clip_grads = True
        p_loss = 0.0
        policy_loss = []
        policy_loss1 = []
        
        ret = []

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
        if (clip_grads):
            for param in self.policy.parameters():
                param.grad.data.clamp_(-1.5, 1.5)
        
        self.policy_optim.step()
      
        self.saved_log_probs = []

        return {"p_loss": p_loss}

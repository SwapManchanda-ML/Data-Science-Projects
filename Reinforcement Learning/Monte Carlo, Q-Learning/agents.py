from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
import random
from typing import List, Dict, DefaultDict
from gym.spaces import Space
from gym.spaces.utils import flatdim


class Agent(ABC):
    """Base class for Q-Learning agent

    **ONLY CHANGE THE BODY OF THE act() FUNCTION**
    """

    def __init__(
        self,
        action_space: Space,
        obs_space: Space,
        gamma: float,
        epsilon: float,
        **kwargs
    ):
        """Constructor of base agent for tabular RL agents

        Initializes basic variables of the agent namely the epsilon and discount
        rate (gamma).

        :param action_space (int): action space of the environment
        :param obs_space (int): observation space of the environment
        :param gamma (float): discount factor (gamma)
        :param epsilon (float): epsilon for epsilon-greedy action selection

        :attr n_acts (int): number of actions
        :attr q_table (DefaultDict): table for Q-values mapping (OBS, ACT) pairs of observations
            and actions to respective Q-values
        """
        self.action_space = action_space
        self.obs_space = obs_space

        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.update_counter = 0

        self.n_acts = flatdim(action_space)

        self.q_table: DefaultDict = defaultdict(lambda: 0)
        #self.q_table: DefaultDict = defaultdict(lambda: np.zeros(self.n_acts))

    def act(self, obs: np.ndarray) -> int:
        """Implement the epsilon-greedy action selection here

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        :param obs (np.ndarray of float with dim (observation size)):
            received observation representing the current environmental state
        :return (int): index of selected action
        """
        '''
        def test(self,obs):
            A = np.ones(self.n_acts, dtype=float) * self.epsilon / self.n_acts
            best_action = np.argmax(self.q_table[obs])
            A[best_action] += (1.0 - self.epsilon)
            return A

        policy = test(self,obs)
        print(policy,obs)
        action_probs = policy[obs]
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action
        '''

        act_vals = [self.q_table[(obs, act)] for act in range(self.n_acts)]
        max_val = max(act_vals)
        max_acts = [idx for idx, act_val in enumerate(act_vals) if act_val == max_val]

        if random.random() < self.epsilon:
            return random.randint(0, self.n_acts - 1)
        else:
            return random.choice(max_acts)
        
        

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def learn(self):
        """Updates the Q-table based on agent experience
        
        **DO NOT CHANGE THIS FUNCTION**
        """
        ...


class QLearningAgent(Agent):
    """Agent using the Q-Learning algorithm
    
    **YOU MUST COMPLETE THIS CLASS**
    """

    def __init__(self, alpha: float, **kwargs):
        """Constructor of QLearningAgent

        Initializes some variables of the Q-Learning agent, namely the epsilon, discount rate
        and learning rate (alpha).

        :param alpha (float): learning rate alpha for Q-learning updates
        """

        super().__init__(**kwargs)
        self.alpha: float = alpha

    def learn(self, obs: np.ndarray, action: int, reward: float, n_obs: np.ndarray, done: bool) -> float:
        """Updates the Q-table based on agent experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        :param obs (np.ndarray of float with dim (observation size)):
            received observation representing the current environmental state
        :param action (int): index of applied action
        :param reward (float): received reward
        :param n_obs (np.ndarray of float with dim (observation size)):
            received observation representing the next environmental state
        :param done (bool): flag indicating whether a terminal state has been reached
        :return (float): updated Q-value for current observation-action pair
        """

        act_vals = [self.q_table[(n_obs, act)] for act in range(self.n_acts)]
        max_val = max(act_vals)

        target_value = reward + self.gamma * (1 - done)* max_val
        self.q_table[(obs, action)] += self.alpha * (target_value - self.q_table[(obs, action)])

        return self.q_table[(obs, action)]

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        max_deduct, decay = 0.9999, 0.04
        self.epsilon = 1 - (min(1, timestep/(decay * max_timestep))) * max_deduct
        #self.alpha = 0.1

class MonteCarloAgent(Agent):
    """Agent using the first-visit Monte-Carlo algorithm for training
    
    **YOU MUST COMPLETE THIS CLASS**
    """

    def __init__(self, **kwargs):
        """Constructor of MonteCarloAgent

        Initializes some variables of the Monte-Carlo agent, namely epsilon,
        discount rate and an empty observation-action pair dictionary.

        :attr sa_counts (Dict[(Obs, Act), int]): dictionary to count occurrences observation-action pairs
        """
        super().__init__(**kwargs)
        self.sa_counts = defaultdict(float)
        self.updated_values = defaultdict(float)

    def learn(
        self, obses: List[np.ndarray], actions: List[int], rewards: List[float]
    ) -> Dict:
        """Updates the Q-table based on agent experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        :param obses (List[np.ndarray] with numpy arrays of float with dim (observation size)):
            list of received observations representing environmental states of trajectory (in
            the order they were encountered)
        :param actions (List[int]): list of indices of applied actions in trajectory (in the
            order they were applied)
        :param rewards (List[float]): list of received rewards during trajectory (in the order
            they were received)
        :return (Dict): A dictionary containing the updated Q-value of all the updated state-action pairs
            indexed by the state action pair.
        """

        episode = []
        
        returns = defaultdict(float)
    
        for state,action,rewards in zip(obses,actions,rewards):
            episode.append((state,action,rewards))

        sa_in_episode = set([(x[0], x[1]) for x in episode])
        
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            first_occurence_idx = next(i for i,x in enumerate(episode)
                                       if x[0] == state and x[1] == action)
           
            G = sum([x[2]*(self.gamma**i) for i,x in enumerate(episode[first_occurence_idx:])])
            self.updated_values[sa_pair] += G 
            self.sa_counts[sa_pair] += 1
            self.q_table[sa_pair] = self.updated_values[sa_pair] / self.sa_counts[sa_pair] 
        return self.q_table
        

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
       
        """
  
        max_deduct, decay = 0.6, 0.03
        self.epsilon =  1 - (min(1, timestep/(decay * max_timestep))) * max_deduct
        


class WolfPHCAgent:
    """Agent using the Wolf-PHC algorithm
    **YOU MUST COMPLETE THIS CLASS**
    DISCLAIMER: This is NOT inheriting from the Agent baseclass!
    """

    def __init__(
        self,
        gamma: float,
        num_acts: int,
        alpha: float,
        win_delta: float,
        lose_delta: float,
        init_policy: List[float],
        **kwargs
    ):
        """Constructor of WolfPHCAgent
        Initializes variables of the Wolf-PhC, such as discount rate, Q-value learning rate
        (alpha), win and lose policy update rates, initial policy, an empty average policy
        dictionary, an empty policy dictionary, and an empty visitation table.
        :param gamma (float): discount factor (gamma)
        :param num_acts (int): number of possible actions
        :param alpha (float): learning rate for the Q-Value table update
        :param win_delta (float): stochastic policy update rate when winning
        :param lose_delta (float): stochastic policy update rate when losing
        :param init_policy (List[float]): initial probability of choosing actions in
            unvisited states
        :attr avg_pi_table (Dict(State, List[float])): average probability of choosing
            actions at states
        :attr pi_table (Dict(State, List[float])): probability of choosing actions
            for each state
        :attr vis_table (Dict(State, int)): state visitation counts for every state
        """
        self.gamma: float = gamma

        self.alpha = alpha
        self.win_delta = win_delta
        self.lose_delta = lose_delta

        self.n_acts: int = num_acts
        self.q_table: DefaultDict = defaultdict(lambda: 0)

        if init_policy is None:
            init_policy = [1.0 / (self.n_acts) for _ in range(self.n_acts)]
        self.init_policy = init_policy

        self.avg_pi_table: DefaultDict = defaultdict(
            lambda: [0 for _ in range(self.n_acts)]
        )
        self.pi_table: DefaultDict = defaultdict(lambda: init_policy.copy())
        self.vis_table: DefaultDict = defaultdict(lambda: 0)
        

    def learn(
        self, obs: int, action: int, reward: float, n_obs: int, done: bool
    ) -> List[float]:
        """Updates the Q-table, policy, average policy, and visitation table
         based on agent experience
        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**
        DISCLAIMER: For RPS we only have a single state and always use 0 as the state/ observation
        :param obs (int): observation representing environmental state
        :param action (int): index of the applied action according to experience
        :param reward (float): reward received according to experience
        :param n_obs (int): next observation representing reached environmental state
        :param done (bool): flag indicating whether a terminal state has been reached
        :return (List[float]): list containing the updated probabilities of choosing each
            valid action in obs
        """
        #Update Q-table
        act_vals = [self.q_table[(n_obs, act)] for act in range(self.n_acts)]
        max_Q_nobs = max(act_vals)
        update_q_table = reward + (1-done)*max_Q_nobs*self.gamma - self.q_table[(obs,action)]
        self.q_table[(obs,action)] = self.q_table[(obs,action)] + self.alpha*update_q_table


        #Update average policy 
        self.vis_table[(obs)] += 1.0

        sum_pi = 0
        sum_avg_pi = 0

        for act in range(self.n_acts):
            self.avg_pi_table[obs][act] += (1/self.vis_table[obs])*(self.pi_table[obs][act] - self.avg_pi_table[obs][act]) 
   
        for act in range(self.n_acts):
            sum_pi += self.pi_table[obs][act]*self.q_table[(obs,act)]
            sum_avg_pi += self.avg_pi_table[obs][act]*self.q_table[(obs,act)]
        
        if sum_pi >= sum_avg_pi:
            win = self.win_delta
        else:
            win = self.lose_delta  
        
        Q_list = [self.q_table[(obs,act)] for act in range(self.n_acts)]
        max_Q_val = max(Q_list)
        A_suboptimal = [idx for idx in range(len(Q_list)) if Q_list[idx] != max_Q_val]

        A_suboptimal_mod = len(A_suboptimal)
        p_moved = 0

        for act in A_suboptimal:
            p_moved = p_moved + min((win/A_suboptimal_mod),self.pi_table[obs][act])
            self.pi_table[obs][act] = self.pi_table[obs][act] - min((win/A_suboptimal_mod),self.pi_table[obs][act])

        for act in range(self.n_acts):
            if act not in A_suboptimal:
                self.pi_table[obs][act] = self.pi_table[obs][act] + p_moved/(self.n_acts - A_suboptimal_mod)
      
        return self.pi_table[obs]

    def act(self, obs: int) -> int:
        """Implement the stochastic policy action selection here
        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**
        :param obs (int): received observation representing the current environmental state
        :return (int): index of selected action
        """
        prob = np.array(self.pi_table[(obs)])
        action = np.random.choice(self.n_acts,size = 1,p=prob)
        return int(action)
 
   

import matplotlib
from torch import optim
import torch.nn.functional as F
from itertools import count

from agent import Agent
from cart_pole_manager import CartPoleEnvManager
from epsilon_greedy_strategy import EpsilonGreedyStrategy
from helper_funcs import *
from qvalues import QValues
from replay_memory import ReplayMemory

is_ipython = 'inline' in matplotlib.get_backend()

# define hyper parameters
batch_size = 256
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 100000
lr = 0.001
num_episodes = 200

# define instances
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU or CPU?
em = CartPoleEnvManager(device)  # manages the environment
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)  # manages exploration vs exploitation
agent = Agent(strategy, em.num_actions_available(), device)  # makes decisions
memory = ReplayMemory(memory_size)  # stores past experiences
policy_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)  # NN for the policy func
target_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)  # NN for the target func
target_net.load_state_dict(policy_net.state_dict())  # set target net to be the same as the policy net
target_net.eval()  # target net is set to evaluation mode
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)  # create optimizer

episode_durations = []

# For each episode...
for episode in range(num_episodes):
    # init starting state
    em.reset()
    state = em.get_state()

    # for each time step
    for timestep in count():
        action = agent.select_action(state, policy_net)
        reward = em.take_action(action)
        next_state = em.get_state()
        memory.push(Experience(state, action, next_state, reward))
        state = next_state

        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)

            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if em.done:
            episode_durations.append(timestep)
            plot(episode_durations, 100)
            break

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    em.close()

finish_training_save(policy_net, target_net)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

import numpy as np

import gymnasium as gym

# %%

class DQN(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DQN, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = opt.Adam(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        
        return actions
    
# %%

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size = 100000, eps_end = 0.01, eps_dec=5e-4):
        self.gamma = gamma
        
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec

        self.lr = lr
        
        self.action_space = [i for i in range(n_actions)]

        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_control = 0

        self.q_eval = DQN(self.lr, n_actions = n_actions, input_dims = input_dims, fc1_dims=256, fc2_dims=256)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_new, done):
        index = self.mem_control % self.mem_size

        self.state_memory[index] = state[0]
        self.new_state_memory[index] = state_new
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_control += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
        # exploitation
            state = torch.tensor([observation]).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
        # exploration
            action = np.random.choice(self.action_space) # can also use env.action_space.sample() from gym

        return action
    
    def learn(self):
        if self.mem_control < self.batch_size:
            return
        
        self.q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_control, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False) #replace = False so that we don't choose the same things again

        batch_index = np.arange(self. batch_size, dtype=np.int32)

        state_batch  = torch.tensor(self.state_memory[batch]).to(self.q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.q_eval.device)

        action_batch = self.action_memory[batch] # doesn't need to be a tensor

        q_eval = self.q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.q_eval.loss(q_target, q_eval).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min


# %%
if __name__ == "__main__":
    env  = gym.make("LunarLander-v2", render_mode="human")
    agent = Agent(gamma=0.99,
                  epsilon = 1.0,
                  batch_size=64,
                  n_actions =4,
                  eps_end=0.01,
                  input_dims=[8],
                  lr = 0.003)
    scores, eps_history = [], []
    n_games = 500

    for i in range (n_games):
        score = 0
        done = False
        # observation = env.reset(options={"randomize": False})
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            
            new_observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                done = True

            score += reward
            agent.store_transition(observation, action, reward, new_observation, done)

            agent.learn()
            observation = new_observation

        scores.append(score)
        eps_history.append(agent.epsilon)

        average_score = np.mean(scores[-100:])

        print("episode", i, "score %.2f" % score, "average_score %0.2f" % average_score, "epsilon %.2f" % agent.epsilon)

        X = [i + 1 for i in range (n_games)]

        # plot_learning_curve(x, scores, eps_history, filename)
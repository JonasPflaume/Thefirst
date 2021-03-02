import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

GAMMA = 0.95

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.affine1 = nn.Linear(4, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.affine2 = nn.Linear(256, 2)

        self.log_probs = []
        self.reward = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.affine2(x)
        return F.softmax(x)

policy = Policy()
env = gym.make('CartPole-v0')
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

def select_action(obs):
    state = torch.from_numpy(obs).float().unsqueeze(0)
    prob_action = policy(state)
    m = Categorical(prob_action)
    action = m.sample()
    policy.log_probs.append(m.log_prob(action))
    return action.item()
    
def episode_updata():
    G = 0
    policy_loss = []
    returns = []
    for r in policy.reward[::-1]:
        G = r + GAMMA * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 0.0001)

    for log_prob, G in zip(policy.log_probs, returns):
        policy_loss.append(-log_prob * G)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).mean()
    policy_loss.backward()
    optimizer.step()
    policy.reward = []
    policy.log_probs = []


def learn():
    id = False
    counter = 0
    for episode in range(1,10000):
        obs, ep_reward = env.reset(), 0
        for t in range(1, 10000):
            action = select_action(obs)
            obs, reward, done, _ = env.step(action)
            if id:
                env.render()
            policy.reward.append(reward)
            ep_reward += reward
            if done:
                break
        episode_updata()
        if ep_reward > 199.9:
            counter += 1
        else:
            counter -= 1
        if counter >= 10:
            id = True
        else:
            id = False

        if counter > 50:
            print("Problem solved!")
            break

        print(ep_reward)

if __name__ == '__main__':
    learn()
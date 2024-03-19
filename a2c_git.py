import multiprocessing
from collections import namedtuple

from tqdm import tqdm
from torch.optim import Adam
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from parallel_environments import ParallelEnvironments
from storage import Storage

LEFT = [-1.0, 0.0, 0.0]
RIGHT = [1.0, 0.0, 0.0]
GAS = [0.0, 1.0, 0.0]
BRAKE = [0.0, 0.0, 1.0]

ACTIONS = [LEFT, RIGHT, GAS, BRAKE]


Params = namedtuple(
    "Params",
    [
        "stack_size",
        "lr",
        "discount_factor",
        "value_loss_coef",
        "entropy_coef",
        "max_norm",
        "episodes",
        "steps_per_ep",
        # "steps_per_update",
        # "num_of_steps",
        # "steps_per_update"
    ]
)


def get_action_space():
    return len(ACTIONS)


def get_actions(probs):
    values, indices = probs.max(1)
    actions = np.zeros((probs.size(0), 3))
    for i in range(probs.size(0)):
        action = ACTIONS[indices[i]]
        actions[i] = float(values[i]) * np.array(action)
    return actions


class ActorCritic(nn.Module):
    def __init__(self, num_of_inputs, num_of_actions):
        super().__init__()

        self.conv1 = nn.Conv2d(num_of_inputs, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.linear1 = nn.Linear(32*9*9, 256)
        self.policy = nn.Linear(256, num_of_actions)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))

        flattened = torch.flatten(conv2_out, start_dim=1)  # N x 9*9*32
        linear1_out = self.linear1(flattened)

        policy_output = self.policy(linear1_out)
        value_output = self.value(linear1_out)

        probs = F.softmax(policy_output)
        log_probs = F.log_softmax(policy_output)
        return probs, log_probs, value_output


# def actor_critic_inference(params, path):
#     model = ActorCritic(params.stack_size, get_action_space())
#     model.load_state_dict(torch.load(path))
#     model.eval()
#
#     env = gym.make('CarRacing-v2')
#     env_wrapper = EnvironmentWrapper(env, params.stack_size)
#
#     state = env_wrapper.reset()
#     state = torch.Tensor([state])
#     done = False
#     total_score = 0
#     while not done:
#         probs, _, _ = model(state)
#         action = get_actions(probs)
#         print(action)
#         state, reward, terminated, truncated, _ = env_wrapper.step(action[0])
#         state = torch.Tensor([state])
#         total_score += reward
#         env_wrapper.render()
#     return total_score
#

class A2CTrainer:
    def __init__(self, params: Params, model_path: str):
        self.params = params
        self.model_path = model_path
        self.num_of_processes = multiprocessing.cpu_count() - 2
        self.parallel_environments = ParallelEnvironments(
            self.params.stack_size,
            number_of_processes=self.num_of_processes
        )
        self.actor_critic = ActorCritic(self.params.stack_size, get_action_space())
        self.optimizer = Adam(self.actor_critic.parameters(), lr=self.params.lr)
        self.storage = Storage(self.params.steps_per_ep, self.num_of_processes)
        self.current_observations = torch.zeros(
            self.num_of_processes,
            *self.parallel_environments.get_state_shape()
        )
        self.rewards = []

    def run(self):
        for episode in range(int(self.params.episodes)):
            print("Episode started")
            self.storage.reset_storage()
            self.current_observations = self.parallel_environments.reset()
            # print(self.current_observations.size())
            for step in tqdm(range(self.params.steps_per_ep)):
                probs, log_probs, value = self.actor_critic(self.current_observations)
                actions = get_actions(probs)
                action_log_probs, entropies = self.compute_action_logs_and_entropies(probs, log_probs)

                states, rewards, dones = self.parallel_environments.step(actions)
                rewards = rewards.view(-1, 1)
                dones = dones.view(-1, 1)
                self.current_observations = states
                self.storage.add(step, value, rewards, action_log_probs, entropies, dones)

            _, _, last_values = self.actor_critic(self.current_observations)
            expected_rewards = self.storage.compute_expected_rewards(last_values,
                                                                     self.params.discount_factor)
            advantages = torch.tensor(expected_rewards) - self.storage.values
            value_loss = advantages.pow(2).mean()
            policy_loss = -(advantages * self.storage.action_log_probs).mean()

            self.optimizer.zero_grad()
            loss = policy_loss - self.params.entropy_coef * self.storage.entropies.mean() + \
                self.params.value_loss_coef * value_loss
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm(self.actor_critic.parameters(), self.params.max_norm)
            self.optimizer.step()
            ep_mean_reward = self.storage.get_mean_reward()
            self.rewards.append(ep_mean_reward)
            print(f"Episode done, mean reward across envs: {ep_mean_reward}")

    def plot(self):
        # Generate recent 50 interval average
        interval_size = 100
        average_reward = []
        for idx in range(len(self.rewards)):
            avg_list = np.empty(shape=(1,), dtype=int)
            if idx < interval_size:
                avg_list = self.rewards[:idx + 1]
            else:
                avg_list = self.rewards[idx - (interval_size - 1):idx + 1]
            average_reward.append(np.average(avg_list))

        # Plot
        plt.plot(self.rewards, label='reward')
        plt.plot(average_reward, label='average reward (100 eps)')
        plt.xlabel('N episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig("Results.png")
        # plt.show()

    def save_models(self):
        torch.save(self.actor_critic.policy, "./Policy")
        torch.save(self.actor_critic.value, "./Value")

    def compute_action_logs_and_entropies(self, probs, log_probs):
        values, indices = probs.max(1)
        indices = indices.view(-1, 1)
        action_log_probs = log_probs.gather(1, indices)

        entropies = -(log_probs * probs).sum(-1)

        return action_log_probs, entropies


def run_training():
    params = Params(**{
        "stack_size": 5,
        "lr": 0.0001,
        "discount_factor": 0.99,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.1,
        "max_norm": 0.5,
        # "num_of_steps": 125000,
        # "steps_per_update": 5,  # batching
        "episodes": 5000,
        "steps_per_ep": 200
    })
    trainer = A2CTrainer(params, "TrainedModel")
    trainer.run()
    trainer.plot()
    trainer.save_models()


if __name__ == '__main__':
    run_training()

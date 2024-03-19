import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from improved_env import create_car_racing_v2
from nn import CNNForEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = create_car_racing_v2()  # среда
state, info = env.reset()
print(state.shape)
print(env.action_space)

actor_net = CNNForEnv(
    state_dim=state.shape[0],
    action_dim=env.action_space.n
).to(device)

critic_net = CNNForEnv(
    state_dim=state.shape[0],
    action_dim=1,
).to(device)


def pick_sample(s):
    with torch.no_grad():
        #   --> size : (1, 4)
        s_batch = np.expand_dims(s, axis=0)
        s_batch = torch.tensor(s_batch, dtype=torch.float).to(device)
        # Get logits from state
        #   --> size : (1, 2)
        logits = actor_net(s_batch)
        #   --> size : (2)
        logits = logits.squeeze(dim=0)
        # From logits to probabilities
        probs = F.softmax(logits, dim=-1)
        # Pick up action's sample
        a = torch.multinomial(probs, num_samples=1)
        # Return
        return a.tolist()[0]


gamma = 0.99  # дисконтирование
reward_records = []  # массив наград

# Оптимизаторы
opt1 = torch.optim.AdamW(critic_net.parameters(), lr=0.001)
opt2 = torch.optim.AdamW(actor_net.parameters(), lr=0.001)

# количество циклов обучения
num_episodes = 200
#
for i in tqdm(range(num_episodes)):
    # в начале эпизода обнуляем массивы и сбрасываем среду
    done = False
    states = []
    actions = []
    rewards = []
    s, _ = env.reset()

    # пока не достигнем конечного состояния продолжаем выполнять действия
    while not done:
        # добавить состояние в список состояний
        states.append(s.tolist())
        # по текущей политике получить действие
        a = pick_sample(s)
        # выполнить шаг, получить награду (r), следующее состояние (s) и флаги конечного состояния (term, trunc)
        s, r, term, trunc, _ = env.step(a)
        # если конечное состояние - устанавливаем флаг окончания в True
        done = term or trunc
        # добавляем действие и награду в соответствующие массивы
        actions.append(a)
        rewards.append(r)

    #
    # Если траектория закончилась (достигли финального состояния)
    #
    # формируем массив полной награды для каждого состояния
    cum_rewards = np.zeros_like(rewards)
    reward_len = len(rewards)
    for j in reversed(range(reward_len)):
        cum_rewards[j] = rewards[j] + (cum_rewards[j + 1] * gamma if j + 1 < reward_len else 0)

    #
    # Оптимизируем параметры сетей
    #

    # Оптимизируем value loss (Critic)
    # Обнуляем градиенты в оптимизаторе
    opt1.zero_grad()
    # преобразуем состояния и суммарные награды для каждого состояния в тензор
    states = torch.tensor(states, dtype=torch.float).to(device)
    cum_rewards = torch.tensor(cum_rewards, dtype=torch.float).to(device)

    # Вычисляем лосс
    values = critic_net(states)
    values = values.squeeze(dim=1)
    vf_loss = F.mse_loss(
        values,
        cum_rewards,
        reduction="none")
    # считаем градиенты
    vf_loss.sum().backward()
    # делаем шаг оптимизатора
    opt1.step()

    # Оптимизируем policy loss (Actor)
    with torch.no_grad():
        values = critic_net(states)

    # Обнуляем градиенты
    opt2.zero_grad()
    # преобразуем к тензорам
    actions = torch.tensor(actions, dtype=torch.int64).to(device)
    # считаем advantage функцию
    advantages = cum_rewards - values

    # считаем лосс
    logits = actor_net(states)
    log_probs = -F.cross_entropy(logits, actions, reduction="none")
    pi_loss = -log_probs * advantages

    # считаем градиент
    pi_loss.sum().backward()
    # делаем шаг оптимизатора
    opt2.step()

    # Выводим итоговую награду в эпизоде (max 500)
    reward_records.append(sum(rewards))

    if i % 100 == 0:
        print("Run episode {} with average reward {}".format(i, np.mean(reward_records[-100:])), end="\r")

    # stop if mean reward for 100 episodes > 475.0
    if np.average(reward_records[-100:]) > 475.0:
        break

print("\nDone")
env.close()

# Generate recent 50 interval average
average_reward = []
for idx in range(len(reward_records)):
    avg_list = np.empty(shape=(1,), dtype=int)
    if idx < 50:
        avg_list = reward_records[:idx+1]
    else:
        avg_list = reward_records[idx-49:idx+1]
    average_reward.append(np.average(avg_list))

# Plot
plt.plot(reward_records, label='reward')
plt.plot(average_reward, label='average reward')
plt.xlabel('N episode')
plt.ylabel('Reward')
plt.legend()
plt.savefig("Results.png")
plt.show()

torch.save(actor_net, "actor")
torch.save(critic_net, "critic")
print("Saved")

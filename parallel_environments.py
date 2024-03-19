import multiprocessing
import gymnasium as gym
import torch

from multiprocessing import Process, Pipe
from env_wrapper import EnvironmentWrapper


def worker(connection, stack_size):
    env = make_environment(stack_size)
    while True:
        command, data = connection.recv()
        if command == 'step':
            state, reward, terminated, truncated, _ = env.step(data)
            done = terminated or truncated
            if done:
                state, _ = env.reset()
            connection.send((state, reward, done))
        elif command == 'reset':
            state, _ = env.reset()
            connection.send(state)


def make_environment(stack_size):
    env = gym.make('CarRacing-v2')
    env_wrapper = EnvironmentWrapper(env, stack_size)
    return env_wrapper


class ParallelEnvironments:
    def __init__(self, stack_size, number_of_processes=multiprocessing.cpu_count()):
        self.number_of_processes = number_of_processes
        self.stack_size = stack_size

        # pairs of connections in duplex connection
        self.parents, self.childs = zip(*[Pipe() for _
                                          in range(number_of_processes)])

        self.processes = [Process(target=worker, args=(child, self.stack_size,), daemon=True)
                          for child in self.childs]

        for process in self.processes:
            process.start()

    def step(self, actions):
        for action, parent in zip(actions, self.parents):
            parent.send(('step', action))
        results = [parent.recv() for parent in self.parents]
        states, rewards, dones = zip(*results)
        return torch.Tensor(states), torch.Tensor(rewards), torch.Tensor(dones)

    def reset(self):
        for parent in self.parents:
            parent.send(('reset', None))
        results = [parent.recv() for parent in self.parents]
        return torch.Tensor(results)

    def get_state_shape(self):
        return (self.stack_size, 84, 84)

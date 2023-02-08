# AI for Doom
#
# Installation
###############
# Install VizDoom and vizdoomgym
# See:
# https://towardsdatascience.com/building-the-ultimate-ai-agent-for-doom-using-dueling-double-deep-q-learning-ea2d5b8cdd9f


# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# Importing the packages for OpenAI and Doom
from vizdoom import gym_wrapper  # noqa
import gym
# Importing the other Python files
import experience_replay, image_preprocessing

# Part 1 - Building the AI

# Making the brain
class CNN(nn.Module):
    def __init__(self, number_actions):
        super(CNN, self).__init__()
        """
        - in_channels: Because we are working with white/black images,
                       we only need 1 channel
        - out_channels: Number of features that we want to detect (common
                        practice is to start with 32 new images with detect
                        features)
        - kernel_size: For the first convulation layer we will start by 5x5 and
                       for the rest of the convulations, we will reduce the size

        Notice that the amount of channels can be considered as the amount of
        images that the convulation layer will input/output.
        """
        self.convolution1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.convolution3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        # Eventually the output of the convolution3 layer will be flatten and passed
        # to fc1. self.count_neurons((1, 80, 80)) has been calculated in a
        # special way, where 1 is the channels (white/black) and 80x80 is the
        # size
        self.fc1 = nn.Linear(in_features=self.count_neurons((1, 80, 80)), out_features=40)
        self.fc2 = nn.Linear(in_features=40, out_features=number_actions)

    def count_neurons(self, image_dim):
        """
        This function will flatten a random image to get the actual number of
        neurons that we need for the full connection 1.
        """
        x = Variable(torch.rand(1, *image_dim))
        """
        max_pool2 takes 3 parameters:
            1. the output of the convolution
            2. the kernel size: 3 is a common value
            3. the strides: by how many pixels is going to slide the kernel. 2
                            is a common value
        """
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        return x.data.view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

"""
Making the body
It inherites nn.Module because we can use the forward method to call the object
as a function. However, we could replace the forward() function by the __call__()
function, similarly to the AI class.
"""
class SoftmaxBody(nn.Module):
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T  # Temperature parameter

    def forward(self, outputs):
        probs = F.softmax(outputs * self.T)
        actions = probs.multinomial(num_samples=1)
        return actions


# Making the AI
class AI:
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs):
        input = Variable(torch.from_numpy(np.array(inputs, dtype=np.float32)))
        output = self.brain(input)
        actions = self.body(output)
        return actions.data.numpy()


# Part 2 - Training the AI with Deep Convolutional Q-Learning

# Getting the Doom environment
doom_env = image_preprocessing.PreprocessImage(
    gym.make("VizdoomCorridor-v0", frame_skip=4, render_mode="human"), width=80, height=80, grayscale=True)
"""
For video recording check this out:
https://www.anyscale.com/blog/an-introduction-to-reinforcement-learning-with-openai-gym-rllib-and-google
"""
#doom_env = gym.wrappers.Monitor(doom_env, "videos", force=True)
number_actions = doom_env.action_space.n

# Building an AI
cnn = CNN(number_actions)
softmax_body = SoftmaxBody(T=100.0)
ai = AI(brain=cnn, body=softmax_body)

# Setting up Experience Replay
# This Experience Replay is combined with the Elegibility Trace to
# uses the last 10 steps to calculate the rewards
n_steps = experience_replay.NStepProgress(env=doom_env, ai=ai, n_step=10)
# capacity: The last 10000 steps
memory = experience_replay.ReplayMemory(n_steps=n_steps, capacity=10000)


# Implementing Eligibility Trace
# Consists of accumulating rewards over several steps (in our case 10 steps)
# Based on this paper https://arxiv.org/pdf/1602.01783.pdf, more specifically
# based on the algorithm of "Asynchronous n-step Q-learning" (page 4)
# See:
#   https://ai.stackexchange.com/a/12909
#   https://arxiv.org/abs/2007.01839
#
# The below implementation will iterate through each serie in the batch, which
# contains a whole game episode of 10 steps. Then it will initialise the
# cumulitve reward to 0 if the game is done (checking the last step) otherwise, it will take the maximum
# Q value of the last output (output[1] which is related to the last step).
# Once the cumulative reward has been initialised, it will iterate backwards
# through each reward with the discount factor (gamma) and it will accumulate it.
# This last accumulated reward will be assigned to the output[0] (related to the
# first step).
#
# As per arXiv:2007.01839: As an example, suppose we collect a key to open a
# door, which leads to an unexpected reward. Using standard one-step TD
# learning, we would update the state in which the door opened.
# Using eligibility traces, we would also update the preceding
# trajectory, including the acquisition of the key.
#
# With the below elegibility trace system, the first step accumulated reward is
# updated, but the rest 9 steps don't get updated and also are not considered
# for the training purposes either. So, for instance the last step (state) will
# never be considered for training. I think the below algorithm should have used
# used all the steps with accumulated rewards.
#
# However, according to the tutor, this is the right thing to return...
def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []

    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype=np.float32)))
        output = cnn(input)
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cumul_reward
        rewards = [x.reward for x in series]
        is_done = "[D]" if series[-1].done else "[R]"
        print(f"{is_done} Action: {series[0].action} | cumR: {cumul_reward} | rewards: {rewards} | o: {output[1].data}")
        inputs.append(state)
        targets.append(target)
    import pdb
    pdb.set_trace()
    pass
    return torch.from_numpy(np.array(inputs, dtype=np.float32)), torch.stack(targets)


# Making the moving average on 100 steps
class MA:
    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size

    def add(self, rewards):
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]

    def average(self):
        return np.mean(self.list_of_rewards)


ma = MA(100)

# Training the AI
loss = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)
nb_epochs = 100
for epoch in range(1, nb_epochs + 1):
    memory.run_steps(200)
    for batch in memory.sample_batch(128):
        inputs, targets = eligibility_trace(batch)
        inputs, targets = Variable(inputs), Variable(targets)
        predictions = cnn(inputs)
        loss_error = loss(predictions, targets)
        optimizer.zero_grad()
        loss_error.backward()
        optimizer.step()
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)
    avg_reward = ma.average()
    print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))
    # The average reward of 150 has been calculated taking into account that
    # getting to the vest gives a reward of 100 points
    if avg_reward >= 150:
        print("Congratulations, your AI wins")
        break

# Closing the Doom environment
doom_env.close()

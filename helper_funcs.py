import os

import matplotlib.pyplot as plt
import torch

from dqn import DQN
from experience import Experience


def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    plt.pause(0.001)
    print("Episode" + str(len(values)) + "\n" + str(moving_avg_period) + "episode moving avg:" + str(moving_avg[-1]))


def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period - 1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()


def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return t1, t2, t3, t4


def finish_training_save(policy_net: DQN, target_net: DQN) -> None:
    save_dir = "saves/"
    save_folder_name = "trial"
    counter = 1

    while os.path.isdir(save_dir + save_folder_name + str(counter)):
        counter += 1

        assert counter < 100, "problem making dir"

    os.mkdir(save_dir + save_folder_name + str(counter))
    torch.save(policy_net, save_dir + save_folder_name + str(counter) + "/policy.pt")
    torch.save(target_net, save_dir + save_folder_name + str(counter) + "/target.pt")
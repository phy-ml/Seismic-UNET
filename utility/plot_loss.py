import numpy as np
# import torch
# import os
import matplotlib.pyplot as plt


# import matplotlib.colors as mcolors

def plot_train_loss_v1(mini_batch_loss, num_epochs, avg_iter=100, custom_label=''):
    iter_per_epoch = len(mini_batch_loss) // num_epochs

    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(len(mini_batch_loss)), mini_batch_loss, label=f'Minibatch Loss{custom_label}')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    if len(mini_batch_loss) < 1000:
        num_loss = len(mini_batch_loss) // 2
    else:
        num_loss = 1000

    ax1.set_ylim([
        0, np.max(mini_batch_loss[num_loss]) * 1.5
    ])

    ax1.plot(np.convolve(mini_batch_loss, np.ones(avg_iter, ) / avg_iter,
                         mode='valid'),
             label=f'Running Average{custom_label}')
    ax1.legend()

    # set second x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs + 1))
    newpos = [i * iter_per_epoch for i in newlabel]
    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())

    plt.tight_layout()

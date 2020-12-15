import os
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
import dataset_handler as ds

def line_activities(states, r, centers, max_cells = 10, save_loc = './results/lineplot'):
    check_dir(save_loc)
    # only show first max_cells cells
    show_path = 0
    rows = max_cells // 5
    cols = max_cells // rows

    fig = plt.figure(constrained_layout = True, figsize = (20, 20))
    spec = fig.add_gridspec(rows + 1, cols)
    count = 0

    axs = []
    axs.append(fig.add_subplot(spec[0, 2]))
    axs[0].plot(r[show_path, 0, 0], r[show_path, 0, 1], 'ok')
    axs[0].plot(r[show_path, :, 0], r[show_path, :, 1], 'r', linewidth = 0.75)
    axs[0].set_xlim([-1.2, 1.2])
    axs[0].set_ylim([-1.2, 1.2])
    axs[0].set_xlabel('x', fontsize = 14)
    axs[0].set_ylabel('y', fontsize = 14)
    axs[0].plot(centers[show_path,:, 0], centers[show_path, :, 1], 'g*', markersize = 1)

    d = np.linalg.norm(centers[:, None,:] - r[:,:, None,:], axis = -1)

    for i in range(1, rows + 1):
        for j in range(cols):
            axs.append(fig.add_subplot(spec[i,j]))
            axs[-1].plot(states[show_path, :, count], linewidth = 0.75)
            axs[-1].set_xlabel('t', fontsize = 14)
            axs[-1].set_ylabel('$f_{pc}$', fontsize = 14)
            ax2 = axs[-1].twinx()
            ax2.plot(d[show_path, :, count], '--m', linewidth = 0.75)
            ax2.set_ylabel('d', fontsize = 14)
            count += 1

    plt.savefig(save_loc + '_line')
    plt.close()


def visualize_activities(activations, r, max_cells = 10, save_loc = './results/heatmap'):
    check_dir(save_loc)

    x = np.ravel(r[:,:,0])
    y = np.ravel(r[:,:,1])

    rows = max_cells // 10
    cols = max_cells // rows

    fig, axs = plt.subplots(rows, cols , figsize = (20, 20))
    count = 0
    for i in range(rows):
        for j in range(cols):
            states = np.ravel(activations[:,:,:max_cells][:,:,count])

            hists, binx, biny, binno = stats.binned_statistic_2d(x, y, states, bins = 50)

            axs[i,j].imshow(hists, cmap = 'jet', origin = 'upper')
            axs[i,j].set_xticklabels([])
            axs[i,j].set_yticklabels([])
            axs[i,j].set_xlabel('x', fontsize = 12)
            axs[i,j].set_ylabel('y', fontsize = 12)
            count += 1

    plt.tight_layout()
    plt.savefig(save_loc + '_activations')
    plt.close()

def check_dir(path):
    # check that path is a directory, if not, create it!
    if not os.path.isdir(path):
        os.makedirs(path)

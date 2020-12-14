import os
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
import dataset_handler as ds

def plot_place_cell():
    # Plot theoretical example of place cell activation
    n = 200 # number of bins
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)

    xx, yy = np.meshgrid(x,y)

    center = (0.8, 0.4) # example place cell center
    std = 0.05 # place cell width
    # 2d Gaussian
    pc = np.exp( -(xx-center[0])**2/(2*std**2) - (yy-center[1])**2/(2*std**2) )
    plt.imshow(pc, extent = [0,1,0,1], cmap = 'jet', origin = 'lower')
    plt.xlabel('x', fontsize = 12)
    plt.ylabel('y', fontsize = 12)
    plt.savefig('./results/visualize_pc.png')
    plt.close()

def plot_grid_cell():
    # plot theoretical example of grid cell
    n = 200 # bins
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)

    xx, yy = np.meshgrid(x,y)

    gc = 0

    theta = np.pi/3 # initial wave vector;
    wave_vectors = 50*np.array([ [np.cos(theta*i), np.sin(theta*i)] for i in range(3)])
    # grid cell modelled as sum of 3 plane waves with pi/3 phase difference
    # frequency determined by prefactor (25 looks nice)
    for k in wave_vectors:
        gc = gc + np.cos(k[0]*xx + k[1]*yy)

    plt.imshow(gc, extent = [0,1,0,1], cmap = 'jet', origin = 'lower')
    plt.xlabel('x', fontsize = 12)
    plt.ylabel('y', fontsize = 12)
    plt.savefig('./results/visualize_gc.png')
    plt.close()

def plot_paths():
    dataset = './datasets/cartesian1000steps.npz'
    x_train, x_test, y_train, y_test = ds.load_dataset(dataset)
    r0 = y_test[:,0]
    r = y_test
    for i in range(1):
        plt.plot(r0[i,0], r0[i, 1], 'o')
        plt.plot(r[i,:, 0], r[i,:, 1], '-ko', linewidth = 0.75, markersize = 0.6)

    #plt.axis([-1, 1, -1, 1])
    plt.xlabel('x', fontsize = 12)
    plt.ylabel('y', fontsize = 12)
    plt.savefig('./results/visualize_paths.png')
    plt.close()

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
    axs[0].plot(centers[show_path,:, 0], centers[show_path, :, 1], 'g*', markersize = 1)

    d = np.linalg.norm(centers[:, None,:] - r[:,:, None,:], axis = -1)

    for i in range(1, rows + 1):
        for j in range(cols):
            axs.append(fig.add_subplot(spec[i,j]))
            axs[-1].plot(states[show_path, :, count], linewidth = 0.75)
            axs[-1].plot(d[show_path, :, count], '--m', linewidth = 0.75)
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

            axs[i,j].imshow(hists, cmap = 'jet')
            count += 1

    plt.tight_layout()
    plt.savefig(save_loc + '_activations')
    plt.close()

def check_dir(path):
    # check that path is a directory, if not, create it!
    if not os.path.isdir(path):
        os.makedirs(path)

if __name__ == '__main__':
    check_dir('./results/')
    plot_place_cell()
    plot_grid_cell()
    plot_paths()

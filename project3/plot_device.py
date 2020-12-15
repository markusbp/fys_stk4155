import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

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


def gaussian(r, s, mu):
    return np.exp(-0.5*(r-mu)**2/s**2)

def plot_rbfs():
    fig = plt.figure(constrained_layout = True, figsize = (20, 10))
    spec = fig.add_gridspec(2, 3)

    count = 0

    axs = []

    c = np.array([0, 0.1, 0.5, 0.8])*5
    cols = ['c', 'r', 'm', 'blue']
    x = np.linspace(0, 1 , 1000)*5
    t = np.linspace(0, 1, 1000)

    axs.append(fig.add_subplot(spec[:,0]))
    axs[0].plot(t, x, linewidth = 0.75, color = 'grey')
    axs[0].set_xlabel('t', fontsize = 14)
    axs[0].set_ylabel('v(t)', fontsize = 14)

    count = 0
    for i in range(2):
        for j in range(1,3):
            axs.append(fig.add_subplot(spec[i,j]))
            axs[-1].plot(t, gaussian(x, 0.1, c[count]), color = cols[count])
            axs[-1].set_xlabel('t',fontsize = 14)
            axs[-1].set_ylabel('$f(v)$', fontsize = 14)
            count += 1
    plt.savefig('./results/visualize_rbf_pc')
    plt.close()

    hd_centers = [-5/6*np.pi]
    hd_sd = 2*np.pi

    hd = tfp.distributions.VonMises(hd_centers, hd_sd)

    theta = np.linspace(-np.pi, np.pi, 1000)

    y = hd.prob(theta)

    #plt.plot(theta, gaussian(theta, 1/(np.sqrt(2*np.pi)), -5/6*np.pi) )
    plt.plot(theta, y, '--r', linewidth = 0.75)
    plt.plot(theta[0], y[0], 'ob', markersize = 4)
    plt.plot(theta[-1], y[-1], 'ob', markersize = 4)

    #plt.plot(theta[46:-46], y[46:-46])
    plt.xlabel('$\\theta$', fontsize = 12)
    plt.ylabel('$f(\\theta)$', fontsize = 12)
    plt.savefig('./results/visualize_rbf_vonMises')
    plt.close()


def plot_decoding():

    t = np.linspace(0, 2*np.pi, 1000)
    c1 = np.array([np.sqrt(2)/2, np.sqrt(2)/2]).reshape(1,2)
    c2 = np.array([0, 1]).reshape(1, 2)
    centers = [c1, c2]
    r = np.array([np.cos(t), np.sin(t)]).T

    pc1 = np.exp( (-0.5*np.sum((r-c1)**2, axis = -1)/0.1**2))
    pc2 = np.exp( (-0.5*np.sum((r-c2)**2, axis = -1)/0.1**2))
    fig, axs = plt.subplots(1, 2, constrained_layout = True, figsize = (10, 5))

    cols = ['k', 'r']
    axs[1].plot(r[:,0], r[:,1], linewidth = 0.75)
    for i, (pc, col) in enumerate(zip([pc1, pc2], cols)):
        axs[0].plot(t, pc, linewidth = 0.75, color = col, label = '$pc_{%.d}$' %i)
        axs[1].plot(centers[i][0,0], centers[i][0,1], 'o', color = col, label = '$pc_{%.d}$' %i)

    axs[0].set_xlabel('t', fontsize = 12)
    axs[0].set_ylabel('$f_{pc}(t)$', fontsize = 12)
    axs[1].set_xlabel('x', fontsize = 12)
    axs[1].set_ylabel('y', fontsize = 12)
    axs[1].set_aspect('equal')
    axs[0].legend(frameon = False)
    axs[1].legend(frameon = False)
    plt.show()

def check_dir(path):
    # check that path is a directory, if not, create it!
    if not os.path.isdir(path):
        os.makedirs(path)


if __name__ == '__main__':
    check_dir('./results/')
    plot_place_cell()
    plot_grid_cell()
    plot_paths()
    plot_rbfs()
    plot_decoding()

import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from scipy.misc import imread

def inspect_data():
    # Load the terrain
    terrain = imread('./datasets/slottsfjell.tif')
    full_tjome = terrain[3000:3100, 700:800] # select square Tjome grid
    full_tjome = full_tjome/np.amax(full_tjome) # scale to between 0 and 1
    tjome = full_tjome[0:-1:3, 0:-1:3] # downsample to finish this century

    np.save('./datasets/full_tjome.npy', full_tjome)
    np.save('./datasets/tjome.npy', tjome)
    # Show the terrain
    plt.imshow(tjome, cmap='coolwarm')
    plt.colorbar()
    plt.xlabel('x',fontsize = 12)
    plt.ylabel('y', fontsize = 12)
    plt.savefig('./results/tjome.png')
    plt.close()
    plt.imshow(full_tjome, cmap='coolwarm')
    plt.colorbar()
    plt.xlabel('x',fontsize = 12)
    plt.ylabel('y', fontsize = 12)
    plt.savefig('./results/full_tjome.png')

def get_dataset():
    labels = np.load('./datasets/tjome.npy') #square grid
    xy = np.linspace(0, 1, len(labels)) # square coordinate grid
    xx, yy = np.meshgrid(xy, xy)
    r = np.stack((np.ravel(xx), np.ravel(yy)), axis = -1) # coordinate vector
    return r, labels.ravel()

def get_full_dataset():
    labels = np.load('./datasets/full_tjome.npy') #square grid
    xy = np.linspace(0, 1, len(labels)) # square coordinate grid
    xx, yy = np.meshgrid(xy, xy)
    r = np.stack((np.ravel(xx), np.ravel(yy)), axis = -1) # coordinate vector
    return r, labels.ravel()



if __name__ == '__main__':
    inspect_data()
    get_dataset()
    get_full_dataset()

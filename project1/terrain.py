import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from scipy.misc import imread

def inspect_data():
    # Load the terrain
    terrain = imread('./datasets/slottsfjell.tif')
    terrain = terrain[3000:3100, 700:800] # select square Vestfold grid
    np.save('./datasets/tjome.npy', terrain)
    # Show the terrain
    plt.imshow(terrain, cmap='coolwarm')
    plt.colorbar()
    plt.xlabel('x',fontsize = 12)
    plt.ylabel('y', fontsize = 12)
    plt.savefig('./results/vestfold.png')

def get_dataset():
    labels = np.load('./datasets/tjome.npy') #square grid
    xy = np.linspace(-1, 1, len(labels))
    xx, yy = np.meshgrid(xy, xy)
    r = np.stack((np.ravel(xx), np.ravel(yy)), axis = -1)
    print(labels.ravel().shape, r.shape)
    return r, labels.ravel()

if __name__ == '__main__':
    inspect_data()
    get_dataset()

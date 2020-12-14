import os
import numpy as np
from sklearn.model_selection import train_test_split

def distance_to_wall(r, bound = 1):
    # distance to nearest walls; exploit cartesian coordinates and square chamber
    x = r[:, 0] # x and y coordinates
    y = r[:, 1]
    # take coordinate - wall coordinate --> distance
    d = [bound - x, bound - y, -bound - x, -bound - y]
    return np.abs(d).T

def create_path(samples, timesteps, dt = 0.1):
    # create samples paths of length timesteps, with stepsize dt
    vmin = 0 # min speed
    vmax = 0.5 # max speed

    s = np.zeros((samples, timesteps)) # speed
    hd = np.zeros((samples, timesteps)) # head direction
    r = np.zeros((samples, timesteps, 2)) # position

    bound = 1 # chamber size

    # Initial position, speed, head direction
    r[:, 0] = np.random.uniform(-bound*0.95, bound*0.95, (samples, 2))
    hd[:,0] = np.random.uniform(0, 2*np.pi, samples)
    s[:,0] = np.random.uniform(0, vmax, samples)

    acc = 0 # acceleration
    stddev = 0.25 # standard deviation, head direction

    # wall normal directions
    wall_angles = np.array([i*np.pi/2 for i in range(4)]).reshape(4)

    proximity_limit = 0.05 # distance to wall when agent should start turning
    turn = 0  # initial turn angle

    indices = np.arange(samples) # just to slice more easily

    for i in range(timesteps-1):
        # compute distances to each wall
        distances = distance_to_wall(r[:,i], bound = bound)
        closest_wall = np.argmin(distances, axis = -1)
        # mask out timesteps where too close to wall, or outside
        too_close = np.where(distances[indices, closest_wall] < proximity_limit, 1, 0)
        outside = np.sum(np.abs(r[:,i]) > bound, axis = -1) # x or y > boundary
        should_turn = too_close + outside # turn when too close or outside
        # difference between head direction and wall normal
        hd_diff = np.where(should_turn, hd[:,i] - wall_angles[closest_wall], 0)
        hd_diff = np.mod(hd_diff + np.pi, 2*np.pi) - np.pi # fix for first wall
        # bounds hd diff between -pi and pi
        turn = np.where(should_turn, np.sign(hd_diff), 0)


        s[:,i+1] =  s[:,i] + np.random.normal(0, 0.1, samples)
        hd[:,i+1] =  hd[:,i] + np.random.normal(0, stddev, (samples)) +  turn
        hd[:,i+1] = np.mod(hd[:,i+1], 2*np.pi)

        s[:,i+1] = np.clip(s[:,i+1], vmin, vmax) # clip speed between 0, vmax
        r[:,i+1] = r[:,i] + dt*s[:,i+1, None]*np.stack([np.cos(hd[:,i+1]), np.sin(hd[:,i+1])], axis = -1)

    return r, hd, s

def create_datasets(save_loc = './datasets/', samples = 10000, timesteps = 1000,
                    n_pc = 500, stddev = 0.1):
    if not os.path.isdir(save_loc):
        os.makedirs(save_loc)

    r, hd, s = create_path(samples, timesteps)

    # save a little bit of space
    r = r.astype('float32')
    v = s[:,:,None]*np.stack((np.cos(hd), np.sin(hd)), axis = -1) # velocity
    # create Cartesian dataset
    x = v.astype('float32')
    np.savez(f'{save_loc}/cartesian{timesteps}steps', x = x, y = r)

    # create HD/speed dataset
    z = np.stack((s, hd), axis = -1)
    np.savez(f'{save_loc}/hd_s_{timesteps}steps', x = z, y = r)

def load_dataset(name):
    data = np.load(name)
    x = data['x']
    y = data['y']
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)
    # repackage to include initial position and predict correct timesteps
    y_train = y_train[:,1:] # agent does not stand still during first step!
    x_train = (x_train[:, 1:], y_train)
    y_test = y_test[:,1:]
    x_test = (x_test[:, 1:], y_test)
    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
    print('Creating Datasets...')
    create_datasets(timesteps = 100)
    create_datasets(timesteps = 1000)
    print('Success!')

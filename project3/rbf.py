import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

def gaussian(r, s, mu):
    return np.exp(-0.5*(r-mu)**2/s**2)

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

plt.plot(theta, y, '--r', linewidth = 0.75)
plt.plot(theta[0], y[0], 'ob', markersize = 4)
plt.plot(theta[-1], y[-1], 'ob', markersize = 4)

#plt.plot(theta[46:-46], y[46:-46])
plt.xlabel('$\\theta$', fontsize = 12)
plt.ylabel('$f(\\theta)$', fontsize = 12)
plt.savefig('./results/visualize_rbf_vonMises')
plt.close()

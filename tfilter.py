import random
import numpy as np
import matplotlib.pyplot as plt

def moving_window_matrix(x,window_size):
    # Fork from https://qiita.com/bauer/items/48ef4a57ff77b45244b6
    n = x.shape[0]
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, shape=(n-window_size+1, window_size), strides=(stride,stride) ).copy()

def hsvd(x, window, rank):
    m = moving_window_matrix(x, window)
    u, s, vh = np.linalg.svd(m)
    h = u[:,:rank] @ np.diag(s[:rank]) @ vh[:rank,:]
    c = h[:,0]
    c = np.append(c, h[-1,1:])
    return c, x-c

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    N = 500
    x = np.sin(np.arange(N) * np.pi/50.0)
    x = x + np.random.normal(0, 0.3, size=N)
    plt.plot(x, label='raw')
    plt.legend()
    plt.savefig('raw.png')
    plt.clf()

    window = 100
    rank = 2
    l, h = hsvd(x, window, rank)

    plt.plot(l, label='low')
    plt.legend()
    plt.savefig('low.png')
    plt.clf()

    plt.plot(h, label='high')
    plt.legend()
    plt.savefig('high.png')
    plt.clf()

    plt.plot(x, label='raw')
    plt.plot(l, label='low')
    plt.plot(h, label='high')
    plt.legend()
    plt.savefig('summary.png')
    plt.clf()

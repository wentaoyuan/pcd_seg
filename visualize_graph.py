import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import sparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='data path')
    parser.add_argument('output', help='output path')
    parser.add_argument('-l', type=int, help='level to show')
    args = parser.parse_args()

    f = np.load(args.data)
    X = f['X'][0]
    Y = f['Y'][0]
    G = f['G'][0]
    G = G[args.l]
    n = X.shape[0]
    for l in range(args.l):
        n = n // 2
        X_sub = np.empty((n,3))
        Y_sub = np.empty((n,))
        for i in range(n):
            if np.sum(X[2*i,:]) == 0:
                X_sub[i,:] = X[2*i+1,:]
                Y_sub[i] = Y[2*i+1]
            elif np.sum(X[2*i+1,:]) == 0:
                X_sub[i,:] = X[2*i,:]
                Y_sub[i] = Y[2*i]
            else:
                X_sub[i,:] = (X[2*i,:] + X[2*i+1,:]) / 2
                Y_sub[i] = Y[2*i]
        X = X_sub
        Y = Y_sub

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], X[:,2], c=Y, zdir='y')
    for i in range(n):
        for j in range(i,n):
            if G[i,j] > 0:
                ax.plot([X[i,0], X[j,0]], [X[i,1], X[j,1]], [X[i,2], X[j,2]],
                    'b', zdir='y', linewidth=0.5)
    plt.savefig(args.output)


if __name__ == '__main__':
    main()
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from util import synset_ids


def main():
    fig = plt.figure(1, figsize=(8, 12))
    for i, (cat, synset_id) in enumerate(sorted(synset_ids.items())):
        pred_dir = os.path.join('../results/level-2_order-1', synset_id)
        if not os.path.exists(pred_dir):
            print(pred_dir, 'does not exist.')
            continue
        data_dir = os.path.join('../data/test_data', synset_id)
        label_dir = os.path.join('../data/test_label', synset_id)
        model_ids = [os.path.splitext(entry)[0]
            for entry in os.listdir(label_dir) if entry.endswith('.seg')]
        file_to_vis = model_ids[np.random.randint(len(model_ids))]
        x = np.loadtxt(os.path.join(data_dir, file_to_vis + '.pts'))
        y = np.loadtxt(os.path.join(label_dir, file_to_vis + '.seg'))
        p = np.loadtxt(os.path.join(pred_dir, file_to_vis + '.seg'))
        lim = [x.min(), x.max()]
        ax = fig.add_subplot(8, 4, 2*i+1, projection='3d')
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], zdir='y', c=p, s=2)
        ax.set_title(cat, x=0, y=0.7, rotation='vertical', transform=ax.transAxes)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_zlim(lim)
        ax.axis('off')
        ax = fig.add_subplot(8, 4, 2*i+2, projection='3d')
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], zdir='y', c=y, s=2)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_zlim(lim)
        ax.axis('off')
    plt.subplots_adjust(left=0.1, bottom=0, right=1, top=1, wspace=0.1, hspace=0)
    plt.show()


if __name__ == '__main__':
    main()

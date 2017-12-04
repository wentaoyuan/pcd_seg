import argparse
import numpy as np
import os
from scipy import sparse
from tensorpack import DataFlow, dftools
from third_party import coarsening, graph


synset_ids = {
    "Airplane": "02691156",
    "Bag": "02773838",
    "Cap": "02954340",
    "Car": "02958343",
    "Chair": "03001627",
    "Earphone": "03261776",
    "Guitar": "03467517",
    "Knife": "03624134",
    "Lamp": "03636649",
    "Laptop": "03642806",
    "Motorbike": "03790512",
    "Mug": "03797390",
    "Pistol": "03948459",
    "Rocket": "04099429",
    "Skateboard": "04225987",
    "Table": "04379243"
}
lmdb_dir = '../data/lmdb'


def sparse_to_tuple(mx):
    mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    values = mx.data
    shape = mx.shape
    return coords, values, shape


def chebyshev(L, k):
    """Return T_k L where T_k are the Chebyshev polynomials of order up to K."""
    m, m = L.shape
    L = graph.rescale_L(L, lmax=2)
    T = [sparse.identity(m, format='csr', dtype=L.dtype)]
    if k >= 1:
        T.append(L)
    # T_k = 2 L T_k-1 - T_k-2.
    for k in range(2, k+1):
        T.append(2 * L.dot(T[k-1]) - T[k-2])
    for k in range(k+1):
        T[k] = sparse_to_tuple(T[k])
    return T


class graph_df(DataFlow):
    def __init__(self, file_list, data_dir, label_dir, args):
        self.file_list = file_list
        # we apply a global shuffling here because later we'll only use local shuffling
        np.random.shuffle(self.file_list)
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.nn = args.n
        self.levels = args.l
        self.orders = args.k
        self.sample_num = args.m

    def get_data(self):
        for file_name in self.file_list:
            try:
                x = np.loadtxt(os.path.join(self.data_dir, file_name + '.pts'))
                y = np.loadtxt(os.path.join(self.label_dir, file_name + '.seg'))
            except ValueError:
                print('Unable to read', file_name)
                continue
            # Normalize point cloud into a 1x1x1 cube
            # x -= x.min(axis=0)
            # x /= x.max(axis=0)
            dist, idx = graph.distance_scipy_spatial(x, self.nn)
            A = graph.adjacency(dist, idx)
            graphs, perm = coarsening.coarsen(A, levels=self.levels)
            x = coarsening.perm_data(x.T, perm).T
            y = coarsening.perm_data(y[None,:], perm)[0,:]
            # Upsample by adding args.m-n fake nodes
            m = self.sample_num
            n = x.shape[0]
            if n > m:
                print('{0} has {1} points which exceeds maximum {2}'.format(file_name, n, m))
            else:
                x = np.pad(x, [(0, m - n), (0,0)], 'constant')
                y = np.pad(y, [(0, m - n)], 'constant')
                for l in range(self.levels + 1):
                    g = graphs[l].tocoo()
                    graphs[l] = sparse.csr_matrix((g.data, (g.row, g.col)),
                        shape=(m, m))
                    m = m // 2
                mask = y > 0   # mask for real nodes
                y[mask] -= 1
                cheby = [chebyshev(graph.laplacian(A), self.orders) for A in graphs]
                # for i in range(cheby.shape[1]):
                #     print(cheby[0][i].nnz, cheby[0][i].nnz / cheby[0][i].shape[0] ** 2)
                yield [x, y, mask, cheby]

    def size(self):
        return len(self.file_list)


def process_dataset(args):
    sample_num = args.m
    split = args.dataset
    for cat_name in args.c:
        synset_id = synset_ids[cat_name]
        print('Processing', synset_id, cat_name)
        data_dir = os.path.join('../data', split + '_data', synset_id)
        label_dir = os.path.join('../data', split + '_label', synset_id)
        file_names = [os.path.splitext(entry)[0] for entry in os.listdir(data_dir)
            if entry.endswith('.pts')]
        df = graph_df(file_names, data_dir, label_dir, args)
        output_path = os.path.join(lmdb_dir, '%s_%d_%s.lmdb' % (cat_name, sample_num, split))
        if os.path.exists(output_path):
            os.system('rm -f %s' % output_path)
        dftools.dump_dataflow_to_lmdb(df, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='dataset to process')
    parser.add_argument('-n', type=int, default=6, help='number of neighbours')
    parser.add_argument('-l', type=int, default=4, help='number of coarsening levels')
    parser.add_argument('-k', type=int, default=3, help='order of Chebyshev polynomial')
    parser.add_argument('-m', type=int, default=4096, help='number of output points')
    parser.add_argument('-c', nargs='+', help='categories to process')
    args = parser.parse_args()
    process_dataset(args)


if __name__ == '__main__':
    main()

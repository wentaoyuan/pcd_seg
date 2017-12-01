import argparse
import json
import numpy as np
import os
from scipy import sparse
from tensorpack import DataFlow
from third_party import coarsening, graph


categories = {
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
"Table": "04379243"}


def chebyshev(L, K):
    """Return T_k L where T_k are the Chebyshev polynomials of order up to K."""
    M, M = L.shape
    L = graph.rescale_L(L, lmax=2)
    T = np.empty((K, M, M), L.dtype)
    T = [sparse.identity(M, format='csr', dtype=L.dtype)]
    if K >= 1:
        T.append(L)
    # T_k = 2 L T_k-1 - T_k-2.
    for k in range(2, K+1):
        T.append(2 * L.dot(T[k-1]) - T[k-2])
    for k in range(K+1):
        T[k] = T[k].tocoo()
    return T


class graph_df(DataFlow):
    def __init__(self, file_list):
        self.file_list = file_list
        # we apply a global shuffling here because later we'll only use local shuffling
        np.random.shuffle(self.file_list)

    def get_data(self):
        for file_name in self.file_list:
            try:
                x = np.loadtxt(os.path.join(data_dir, entries[j] + '.pts'))
                y = np.loadtxt(os.path.join(label_dir, entries[j] + '.seg'))
            except ValueError:
                print '++++++ Unable to read %s' % file_name
                continue
            # Normalize point cloud into a 1x1x1 cube
            x -= x.min(axis=0)
            # x /= x.max(axis=0)
            dist, idx = graph.distance_scipy_spatial(x, args.n)
            A = graph.adjacency(dist, idx)
            graphs, perm = coarsening.coarsen(A, levels=args.l)
            x = coarsening.perm_data(x.T, perm).T
            y = coarsening.perm_data(y[None,:], perm)[0,:]
            # Upsample by adding args.m-n fake nodes
            m = args.m
            n = x.shape[0]
            if n > m:
                print('{0}: {1} points exceed maximum'.format(entries[j], n))
            else:
                x = np.pad(x, [(0, m - n), (0,0)], 'constant')
                y = np.pad(y, [(0, m - n)], 'constant')
                for l in range(args.l+1):
                    g = graphs[l].tocoo()
                    graphs[l] = sparse.csr_matrix((g.data, (g.row, g.col)),
                        shape=(m,m))
                    m = m // 2
                m = y > 0   # mask for real nodes
                y[m] -= 1
                cheby = [chebyshev(graph.laplacian(A), args.k) for A in graphs]
                yield [x, y, m] + cheby

    def size(self):
        return len(self.file_list)
        


def process_dataset(args):
    if args.c:
        categories = {args.c: categories[args.c]}
    for cat_name, synset_id in categories.items():
        print('Processing', synset_id, cat_name)
        data_dir = os.path.join('../data', args.dataset + '_data', synset_id)
        label_dir = os.path.join('../data', args.dataset + '_label', synset_id)
        output_dir = os.path.join('../data', args.dataset + '_lmdb', cat_name)
        os.makedirs(output_dir, exist_ok=True)
        file_names = [os.path.splitext(entry)[0] for entry in os.listdir(data_dir)
            if not entry.startswith('.')]

        df = graph_df(file_names)
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
    parser.add_argument('-b', type=int, default=128, help='size of output batch per file')
    parser.add_argument('-c', help='category to process')
    args = parser.parse_args()
    process_dataset(args)


if __name__ == '__main__':
    main()

import os
from tensorpack import dataflow
from termcolor import cprint


def print_emph(content):
    cprint(content, 'blue', attrs=['bold'])


def create_dir(dir_name):
    if os.path.exists(dir_name):
        delete_key = input('===== %s exists. Delete? [y (or enter)/n] ' % dir_name)
        if delete_key == 'y' or delete_key == "":
            os.system('rm -rf %s' % dir_name)
    else:
        os.makedirs(dir_name)


def batch_generator(lmdb_path, batch_size, nproc):
    df = dataflow.LMDBDataPoint(lmdb_path, shuffle=True)
    num_samples = df.size()
    df = dataflow.PrefetchDataZMQ(df, nproc)
    df = dataflow.BatchData(df, batch_size, use_list=True)
    df = dataflow.RepeatedData(df, -1)
    df.reset_state()
    return df.get_data(), num_samples


def get_feed_dict(placeholders, points, labels, mask, cheby):
    feed_dict = {
        placeholders['points']: points,
        placeholders['labels']: labels,
        placeholders['mask']: mask
    }
    for i, level in enumerate(placeholders['cheby']):
        for j, order in enumerate(level):
            for k, pl in enumerate(order):
                # cheby.shape = (batch, level, order)
                indices, values, shape = cheby[k][i][j]
                feed_dict[pl] = tf.SparseTensorValue(indices, values, shape)
    return feed_dict

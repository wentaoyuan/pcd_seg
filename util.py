import os
import tensorflow as tf
from tensorpack import dataflow


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


def create_dir(dir_name):
    if os.path.exists(dir_name):
        delete_key = input('%s exists. Delete? [y (or enter)/n] ' % dir_name)
        if delete_key == 'y' or delete_key == "":
            os.system('rm -rf %s' % dir_name)
        else:
            exit(1)
    os.makedirs(dir_name, exist_ok=True)


def batch_generator(lmdb_path, batch_size, nproc, repeat):
    df = dataflow.LMDBDataPoint(lmdb_path, shuffle=True)
    num_samples = df.size()
    df = dataflow.PrefetchDataZMQ(df, nproc)
    df = dataflow.BatchData(df, batch_size, use_list=True)
    if repeat:
        df = dataflow.RepeatedData(df, -1)
    df.reset_state()
    return df.get_data(), num_samples


def get_cheby_feed_dict(cheby_pl, cheby):
    feed_dict = {}
    for i, level in enumerate(cheby_pl):
        for j, order in enumerate(level):
            for k, pl in enumerate(order):
                # cheby.shape = (batch, level, order)
                indices, values, shape = cheby[k][i][j]
                feed_dict[pl] = tf.SparseTensorValue(indices, values, shape)
    return feed_dict

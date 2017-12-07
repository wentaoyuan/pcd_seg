import argparse
import numpy as np
import tensorflow as tf
import tf_ops
from termcolor import colored
from util import *


def test(args):
    checkpoint_dir = os.path.join('../log', args.task_name)
    if not os.path.exists(checkpoint_dir):
        print('Checkpoint %s does not exist!' % checkpoint_dir)
        exit(1)
    output_dir = os.path.join('../data/test_pred', synset_ids[args.category])
    create_dir(output_dir)

    test_lmdb_path = '../data/lmdb/%s_%d_%s.lmdb' % (args.category, args.num_points, 'test')
    test_gen, num_test_samples = batch_generator(test_lmdb_path, batch_size=1, nproc=1, repeat=False)

    with tf.Graph().as_default():
        print(colored('Creating model...', on_color='on_blue'))
        points_pl = tf.placeholder(tf.float32, shape=(1, args.num_points, 3), name='points')
        # labels_pl = tf.placeholder(tf.int32, shape=(1, args.num_points), name='labels')
        # mask_pl = tf.placeholder(tf.bool, shape=(1, args.num_points), name='mask')
        cheby_pl = [[[tf.sparse_placeholder(tf.float32, name='cheby_l%d_o%d' % (k, j))]
            for j in range(args.order+1)]
            for k in range(args.level+1)]

        output = tf_ops.gcn(points_pl, cheby_pl, args.num_points, args.num_parts, args.level)
        # accuracy = tf_ops.masked_accuracy(output, labels_pl, mask_pl)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session()

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        restorer = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        print(colored("Model restoring from %s..." % latest_checkpoint, on_color='on_red'))
        restorer.restore(sess, latest_checkpoint)
        print(colored("Restored from %s." % latest_checkpoint, on_color='on_red'))

        for file_name, points, perm, cheby in test_gen:
            feed_dict = {points_pl: points}
            feed_dict.update(get_cheby_feed_dict(cheby_pl, cheby))
            pred = sess.run(output, feed_dict=feed_dict)
            pred = np.argmax(pred[0], axis=1)
            pred = pred[perm[0]] + 1    # labels count from 1 instead of 0
            output_path = os.path.join(output_dir, '%s.seg' % file_name[0])
            with open(output_path, 'w') as f:
                for p in pred:
                    print(p, file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--category')
    parser.add_argument('--task_name')
    parser.add_argument('--num_points', type=int, default=4096)
    parser.add_argument('--num_parts', type=int, default=4)
    parser.add_argument('--level', type=int, default=2)
    parser.add_argument('--order', type=int, default=1)
    args = parser.parse_args()
    test(args)

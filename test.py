import argparse
import numpy as np
import tensorflow as tf


def test(args):
    checkpoint_dir = os.path.join('../checkpoint', args.task_name)
    pred_dir = os.path.join('../prediction', args.task_name)
    if not os.path.exist(checkpoint_dir):
        print('Checkpoint %s does not exist!' % checkpoint_dir)
        exit(1)
    create_dir(pred_dir)

    test_lmdb_path = '../data/lmdb/%s_%d_%s.lmdb' % (args.category, args.num_points, 'test')
    test_gen, num_test_samples = batch_generator(test_lmdb_path, batch_size=1, nproc=1)

    with tf.Graph().as_default():
        print_emph('Creating model...')
        points = tf.placeholder(tf.float32,shape=(1, args.num_points, 3), name='points')
        labels = tf.placeholder(tf.int32,shape=(1, args.num_points), name='labels')
        mask = tf.placeholder(tf.bool,shape=(1, args.num_points), name='mask')
        cheby = [[[tf.sparse_placeholder(tf.float32, name='cheby_l%d_o%d_b%d' % (k, j, i))]
            for j in range(args.order+1)]
            for k in range(args.level+1)]
        placeholders = {'points': points, 'labels': labels, 'masks': mask, 'cheby': cheby}

        output = tf_ops.gcn(points, cheby, args.num_points, args.num_parts)

        sess = tf.Session()

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        restorer = tf.train.Saver()
        print_emph("Model restoring from %s..." % latest_checkpoint)
        restorer.restore(sess, latest_checkpoint)
        print_emph("Restored from %s." % latest_checkpoint)
        for points, labels, mask, cheby in test_gen:
            feed_dict = get_feed_dict(placeholders, points, labels, mask, cheby)
            pred = sess.run(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--category')
    parser.add_argument('--task_name')
    parser.add_argument('--num_points', type=int, default=4096)
    parser.add_argument('--num_parts', type=int, default=4)
    args = parser.parse_args()
    test(args)

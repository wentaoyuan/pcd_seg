import argparse
import numpy as np
import os
import tensorflow as tf
from tensorpack import dataflow
from termcolor import cprint
import tf_ops
import time


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


def get_feed_dict(placeholders, points, labels, masks, cheby):
    feed_dict = {
        placeholders['points']: points,
        placeholders['labels']: labels,
        placeholders['masks']: masks
    }
    for i, level in enumerate(placeholders['cheby']):
        for j, order in enumerate(level):
            for k, pl in enumerate(order):
                # cheby.shape = (batch, level, order)
                indices, values, shape = cheby[k][i][j]
                feed_dict[pl] = tf.SparseTensorValue(indices, values, shape)
    return feed_dict


def get_learning_rate(global_step, args):
    learning_rate = tf.train.exponential_decay(
        args.base_lr,
        global_step,
        args.lr_decay_steps,
        args.lr_decay_rate,
        staircase=True,
        name='learning_rate')
    learning_rate = tf.maximum(learning_rate, args.lr_clip) # CLIP THE LEARNING RATE!
    return learning_rate


def train(args):
    log_dir = os.path.join('../log', args.task_name)
    checkpoint_dir = os.path.join('../checkpoint', args.task_name)
    create_dir(log_dir)
    create_dir(checkpoint_dir)

    train_lmdb_path = '../data/lmdb/%s_%d_%s.lmdb' % (args.category, args.num_points, 'train')
    val_lmdb_path = '../data/lmdb/%s_%d_%s.lmdb' % (args.category, args.num_points, 'val')
    train_gen, num_train_samples = batch_generator(train_lmdb_path, args.batch_size, args.nproc)
    val_gen, num_val_samples = batch_generator(val_lmdb_path, args.batch_size, args.nproc)
    num_val_batches = num_val_samples // args.batch_size

    with tf.Graph().as_default():
        print_emph('Creating model...')
        points = tf.placeholder(tf.float32,shape=(args.batch_size, args.num_points, 3), name='points')
        labels = tf.placeholder(tf.int32,shape=(args.batch_size, args.num_points), name='labels')
        masks = tf.placeholder(tf.bool,shape=(args.batch_size, args.num_points), name='masks')
        cheby = [[[tf.sparse_placeholder(tf.float32, name='cheby_l%d_o%d_b%d' % (k, j, i))
            for i in range(args.batch_size)]
            for j in range(args.order+1)]
            for k in range(args.level+1)]
        # is_training = tf.placeholder(tf.bool, shape=())
        placeholders = {'points': points, 'labels': labels, 'masks': masks, 'cheby': cheby}

        output = tf_ops.gcn(points, cheby, args.num_points, args.num_parts)
        xentropy = tf_ops.masked_sparse_softmax_cross_entropy(output, labels, masks)
        accuracy = tf_ops.masked_accuracy(output, labels, masks)

        sess = tf.Session()

        global_step = tf.train.create_global_step(sess.graph)
        learning_rate = get_learning_rate(global_step, args)
        lr_summary = tf.summary.scalar('learning_rate', learning_rate)
        trainer = tf.train.AdamOptimizer(learning_rate)
        train_op = trainer.minimize(xentropy, global_step, tf.trainable_variables())
        saver = tf.train.Saver()

        train_loss = tf.summary.scalar('training_loss', xentropy)
        train_acc = tf.summary.scalar('training_accuracy', accuracy)
        train_summary = tf.summary.merge([train_loss, train_acc, lr_summary])
        val_loss_pl = tf.placeholder(tf.float32, shape=())
        val_acc_pl = tf.placeholder(tf.float32, shape=())
        val_loss = tf.summary.scalar('validation_loss', val_loss_pl)
        val_acc = tf.summary.scalar('validation_accuracy', val_acc_pl)
        val_summary = tf.summary.merge([val_loss, val_acc])
        writer = tf.summary.FileWriter(log_dir, sess.graph)

        init = tf.global_variables_initializer()
        sess.run(init)

        print_emph('Training...')
        start = time.time()
        step = 0
        while step < args.max_steps:
            step = tf.train.global_step(sess, global_step) + 1
            epoch = step * args.batch_size // num_train_samples + 1

            points, labels, masks, cheby = next(train_gen)
            feed_dict = get_feed_dict(placeholders, points, labels, masks, cheby)
            _, loss, acc, summary = sess.run([train_op, xentropy, accuracy, train_summary],
                feed_dict=feed_dict)
            writer.add_summary(summary, step)
            if step % args.print_steps == 0:
                print('Epoch:', epoch, ' step:', step, ' loss:', loss, ' accuracy:', acc,
                    ' time:', time.time() - start)

            if step % args.eval_steps == 0:
                print_emph('Evaluating...')
                total_loss = 0.
                total_acc = 0.
                for i in range(num_val_batches):
                    points, labels, masks, cheby = next(val_gen)
                    feed_dict = get_feed_dict(placeholders, points, labels, masks, cheby)
                    loss, acc = sess.run([xentropy, accuracy], feed_dict=feed_dict)
                    total_loss += loss * args.batch_size
                    total_acc += acc * args.batch_size
                mean_loss = total_loss / (num_val_batches * args.batch_size)
                mean_acc = total_acc / (num_val_batches * args.batch_size)
                summary = sess.run(val_summary, feed_dict={val_loss_pl: mean_loss, val_acc_pl: mean_acc})
                writer.add_summary(summary, step)
                print('Epoch:', epoch, ' step:', step, ' loss:', mean_loss, ' accuracy:', mean_acc,
                    ' time:', time.time() - start)
                save_path = os.path.join(checkpoint_dir, 'Epoch-%d_step-%d' % (epoch, step))
                saver.save(sess, save_path)
                print_emph('Model saved at %s' % save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--category')
    parser.add_argument('--task_name')
    parser.add_argument('--nproc', type=int, default=8)
    parser.add_argument('--num_points', type=int, default=4096)
    parser.add_argument('--num_parts', type=int, default=4)
    parser.add_argument('--level', type=int, default=2)
    parser.add_argument('--order', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=1e7)
    parser.add_argument('--print_steps', type=int, default=100)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay_steps', type=int, default=1e5)
    parser.add_argument('--lr_decay_rate', type=float, default=0.7)
    parser.add_argument('--lr_clip', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    args = parser.parse_args()
    train(args)

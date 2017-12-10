import argparse
import numpy as np
import os
import tensorflow as tf
import tf_ops
import time
from termcolor import colored
from util import *


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
    def log(content):
        log_file.write(content)
        log_file.write('\n')
        log_file.flush()
        print(content)

    log_dir = os.path.join('../log', args.task_name)
    create_dir(log_dir)
    log_file = open(os.path.join(log_dir, 'log.txt'), 'w')
    log(str(args))

    train_lmdb_path = '../data/lmdb/%s_%d_%s.lmdb' % (args.category, args.num_points, 'train')
    val_lmdb_path = '../data/lmdb/%s_%d_%s.lmdb' % (args.category, args.num_points, 'val')
    train_gen, num_train_samples = batch_generator(train_lmdb_path, args.batch_size, args.nproc, repeat=True)
    val_gen, num_val_samples = batch_generator(val_lmdb_path, args.batch_size, args.nproc, repeat=True)
    num_val_batches = num_val_samples // args.batch_size

    with tf.Graph().as_default():
        print(colored('Creating model...', on_color='on_blue'))
        with tf.device('/gpu:0'):
            points_pl = tf.placeholder(tf.float32, shape=(args.batch_size, args.num_points, 3), name='points')
            labels_pl = tf.placeholder(tf.int32, shape=(args.batch_size, args.num_points), name='labels')
            mask_pl = tf.placeholder(tf.bool, shape=(args.batch_size, args.num_points), name='mask')
            cheby_pl = [[[tf.sparse_placeholder(tf.float32, name='cheby_l%d_o%d_%d' % (l, k, j))
                for j in range(args.batch_size)]
                for k in range(args.order+1)]
                for l in range(args.level+1)]
            cheby_pl = [cheby_pl[0]]

            logits = tf_ops.gcn_nopool(points_pl, cheby_pl, args.num_points, args.num_parts, args.level)
            xentropy = tf_ops.masked_sparse_softmax_cross_entropy(labels_pl, logits, mask_pl)
            predictions = tf.argmax(logits, axis=2, output_type=tf.int32)
            accuracy, update_acc = tf_ops.masked_accuracy(labels_pl, predictions, mask_pl)
            mean_iou, update_iou = tf_ops.masked_iou(labels_pl, predictions, args.num_parts, mask_pl)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        if args.restore:
            latest_checkpoint = tf.train.latest_checkpoint(log_dir)
            print(colored("Model restoring from %s..." % latest_checkpoint, on_color='on_red'))
            restorer.restore(sess, latest_checkpoint)
            print(colored("Restored from %s." % latest_checkpoint, on_color='on_red'))

        global_step = tf.train.create_global_step(sess.graph)
        learning_rate = get_learning_rate(global_step, args)
        lr_summary = tf.summary.scalar('learning_rate', learning_rate)
        trainer = tf.train.AdamOptimizer(learning_rate)
        train_op = trainer.minimize(xentropy, global_step, tf.trainable_variables())

        train_loss = tf.summary.scalar('training_loss', xentropy)
        train_acc = tf.summary.scalar('training_accuracy', accuracy)
        train_iou = tf.summary.scalar('training_iou', mean_iou)
        train_summary = tf.summary.merge([train_loss, train_acc, train_iou, lr_summary])
        val_loss_pl = tf.placeholder(tf.float32, shape=())
        val_loss = tf.summary.scalar('validation_loss', val_loss_pl)
        val_acc = tf.summary.scalar('validation_accuracy', accuracy)
        val_iou = tf.summary.scalar('validation_iou', mean_iou)
        val_summary = tf.summary.merge([val_loss, val_acc, val_iou])
        writer = tf.summary.FileWriter(log_dir, sess.graph)

        sess.run(tf.global_variables_initializer())

        log(colored('Training...', on_color='on_red'))
        start = time.time()
        step = 0
        max_iou = 0.
        while step < args.max_steps:
            step = tf.train.global_step(sess, global_step) + 1
            epoch = step * args.batch_size // num_train_samples + 1

            points, labels, mask, cheby = next(train_gen)
            feed_dict = {points_pl: points, labels_pl: labels, mask_pl: mask}
            feed_dict.update(get_cheby_feed_dict(cheby_pl, cheby))

            sess.run(tf.local_variables_initializer())
            sess.run([train_op, update_acc, update_iou], feed_dict=feed_dict)
            loss, acc, iou, summary = sess.run([xentropy, accuracy, mean_iou, train_summary],
                feed_dict=feed_dict)
            writer.add_summary(summary, step)
            if step % args.print_steps == 0:
                log('Epoch: %d  step: %d  loss: %f  accuracy: %f  iou: %f  time: %f' % (
                    epoch, step, loss, acc, iou, time.time() - start))

            if step % args.eval_steps == 0:
                log(colored('Evaluating...', on_color='on_green'))
                loss = 0.
                for i in range(num_val_batches):
                    points, labels, mask, cheby = next(val_gen)
                    feed_dict = {points_pl: points, labels_pl: labels, mask_pl: mask}
                    feed_dict.update(get_cheby_feed_dict(cheby_pl, cheby))
                    loss += sess.run(xentropy, feed_dict=feed_dict)
                    sess.run([update_acc, update_iou], feed_dict=feed_dict)
                loss /= num_val_batches
                acc, iou = sess.run([accuracy, mean_iou], feed_dict=feed_dict)
                summary = sess.run(val_summary, feed_dict={val_loss_pl: loss})
                writer.add_summary(summary, step)
                log('Epoch: %d  step: %d  loss: %f  accuracy: %f  iou: %f  time: %f' % (
                    epoch, step, loss, acc, iou, time.time() - start))
                log(colored('Done', on_color='on_green'))
                if iou > max_iou:
                    max_iou = iou
                    saver.save(sess, os.path.join(log_dir, 'model.ckpt'), step)
                    log(colored('Model saved at %s' % log_dir, on_color='on_red'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--category')
    parser.add_argument('--task_name')
    parser.add_argument('--restore', action=store_true)
    parser.add_argument('--nproc', type=int, default=4)
    parser.add_argument('--num_points', type=int, default=4096)
    parser.add_argument('--num_parts', type=int, default=4)
    parser.add_argument('--level', type=int, default=2)
    parser.add_argument('--order', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=1e6)
    parser.add_argument('--print_steps', type=int, default=100)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay_steps', type=int, default=1e5)
    parser.add_argument('--lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--lr_clip', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    args = parser.parse_args()
    train(args)

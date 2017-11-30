from __future__ import print_function
import numpy as np
import os
import tensorflow as tf
import tf_ops
import time


# def cheby_to_tuple(C):
#     """Convert Chebyshev coefficients to tuple representation."""
#     def sparse_to_tuple(mx):
#         mx = mx.tocoo()
#         coords = np.vstack((mx.row, mx.col)).transpose()
#         values = mx.data
#         shape = mx.shape
#         return coords, values, shape
#     return [[[sparse_to_tuple(L) for L in level]
#         for level in cheby] for cheby in C]


def load_data(filename, level, order):
    f = np.load(filename)
    return f['X'], f['Y'], f['M'], f['C'][:, :level+1, :order+1]


def get_coef(C):
    # (batch, level, order) => (level, order, batch)
    C = np.transpose(C, [1, 2, 0])
    C = [np.stack([np.stack([mx.toarray() for mx in order])
            for order in level]) for level in C]
    return C


def train():
    category = 'Airplane'
    train_dir = os.path.join('../data/train_npz', category)
    val_dir = os.path.join('../data/val_npz', category)
    num_train_files = len([f for f in os.listdir(train_dir) if not f.startswith('.')])
    num_val_files = len([f for f in os.listdir(val_dir) if not f.startswith('.')])
    log_dir = '../log/Airplane_2level'
    if os.path.exists(log_dir):
        delete_key = input('===== %s exists. Delete? [y (or enter)/n] ' % log_dir)
        if delete_key == 'y' or delete_key == "":
            os.system('rm -rf %s' % log_dir)
    else:
        os.makedirs(log_dir)
    batch_size = 16
    num_point = 4096
    num_parts = 4
    level = 2
    order = 1
    learning_rate = 0.001
    epochs = 20

    with tf.Graph().as_default():
        inputs_ph = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
        labels_ph = tf.placeholder(tf.int32, shape=(batch_size, num_point))
        masks_ph = tf.placeholder(tf.bool, shape=(batch_size, num_point))
        cheby_ph = [tf.placeholder(tf.float32, shape=(order+1, batch_size,
            num_point // 2**i, num_point // 2**i)) for i in range(level+1)]
        is_training_ph = tf.placeholder(tf.bool, shape=())

        logits = tf_ops.gcn(inputs_ph, cheby_ph, is_training_ph, num_parts)
        loss_op = tf_ops.masked_sparse_softmax_cross_entropy(logits, labels_ph, masks_ph)
        accuracy_op = tf_ops.masked_accuracy(logits, labels_ph, masks_ph)

        train_loss_ph = tf.placeholder(tf.float32, shape=())
        test_loss_ph = tf.placeholder(tf.float32, shape=())
        train_acc_ph = tf.placeholder(tf.float32, shape=())
        test_acc_ph = tf.placeholder(tf.float32, shape=())

        train_loss_summary_op = tf.summary.scalar('training_loss', train_loss_ph)
        test_loss_summary_op = tf.summary.scalar('testing_loss', test_loss_ph)
        train_acc_summary_op = tf.summary.scalar('training_accuracy', train_acc_ph)
        test_acc_summary_op = tf.summary.scalar('testing_accuracy', test_acc_ph)

        train_variables = tf.trainable_variables()
        trainer = tf.train.AdamOptimizer(learning_rate)
        train_op = trainer.minimize(loss_op, var_list=train_variables)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(log_dir, 'test'))

        for epoch in range(epochs):
            steps = 0
            train_loss = 0.
            train_acc = 0.
            for i in np.random.permutation(num_train_files):
                start = time.process_time()
                X, Y, M, C = load_data(os.path.join(train_dir, str(i) + '.npz'), level, order)
                perm = np.random.permutation(X.shape[0])
                X = X[perm]
                Y = Y[perm]
                M = M[perm]
                C = C[perm]
                for j in range(X.shape[0] // batch_size):
                    begin = j * batch_size
                    end = (j+1) * batch_size
                    C_batch = get_coef(C[begin:end])
                    feed_dict = {inputs_ph: X[begin:end],
                                 labels_ph: Y[begin:end],
                                 masks_ph: M[begin:end],
                                 is_training_ph: True}
                    feed_dict.update({ph: c for ph,c in zip(cheby_ph, C_batch)})
                    _, loss, acc = sess.run([train_op, loss_op, accuracy_op],
                        feed_dict=feed_dict)
                    train_loss += loss
                    train_acc += acc
                steps += X.shape[0] // batch_size
                end = time.process_time()
                print('Step:', steps, ' Time:', end - start)
            train_loss /= steps * batch_size
            train_acc /= steps * batch_size
            train_loss_summary, train_acc_summary = sess.run(
                [train_loss_summary_op, train_acc_summary_op],
                feed_dict={train_loss_ph: train_loss, train_acc_ph:train_acc})
            train_writer.add_summary(train_loss_summary, epoch)
            train_writer.add_summary(train_acc_summary, epoch)
            print('Epoch:', epoch)
            print('Train loss:', train_loss)
            print('Train accuracy:', train_acc)

            steps = 0
            test_loss = 0.
            test_acc = 0.
            for i in range(num_val_files):
                X, Y, M, C = load_data(os.path.join(val_dir, str(i) + '.npz'), level, order)
                for j in range(X.shape[0] // batch_size):
                    begin = j * batch_size
                    end = (j+1) * batch_size
                    C_batch = get_coef(C[begin:end])
                    feed_dict = {inputs_ph: X[begin:end],
                                 labels_ph: Y[begin:end],
                                 masks_ph: M[begin:end],
                                 is_training_ph: False}
                    feed_dict.update({ph: c for ph,c in zip(cheby_ph, C_batch)})
                    loss, acc = sess.run([loss_op, accuracy_op], feed_dict=feed_dict)
                    test_loss += loss
                    test_acc += acc
                steps += X.shape[0] // batch_size
            test_loss /= steps * batch_size
            test_acc /= steps * batch_size
            test_loss_summary, test_acc_summary = sess.run(
                [test_loss_summary_op, test_acc_summary_op],
                feed_dict={test_loss_ph: test_loss, test_acc_ph:test_acc})
            test_writer.add_summary(test_loss_summary, epoch)
            test_writer.add_summary(test_acc_summary, epoch)

            print('Test loss:', test_loss)
            print('Test accuracy:', test_acc)


if __name__ == '__main__':
    train()

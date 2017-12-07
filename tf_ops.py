import numpy as np
import tensorflow as tf
from third_party import tf_util


def sparse_tensor_dense_matmul_3D(A, B):
    return tf.stack([tf.sparse_tensor_dense_matmul(a, b)
        for a,b in zip(A, tf.unstack(B))])


def graph_conv(inputs,
               cheby,
               num_output_channels,
               scope,
               weight_decay=0.0,
               activation_fn=tf.nn.relu,
               bn=False,
               bn_decay=None,
               is_training=None):
    with tf.variable_scope(scope) as sc:
        num_input_channels = inputs.shape[2].value
        filters = [tf.get_variable(
                'weights_' + str(i),
                shape=[1, num_input_channels, num_output_channels],
                dtype=tf.float32,
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            for i in range(len(cheby))]
        terms = [sparse_tensor_dense_matmul_3D(c, inputs) for c in cheby]
        outputs = [tf.nn.conv1d(terms[i], filters[i], stride=1, padding='SAME')
            for i in range(len(cheby))]
        outputs = tf.add_n(outputs)
        if activation_fn:
            outputs = activation_fn(outputs)
        return outputs


def gcn(points, cheby, num_points, num_parts, num_levels):
    gc = [None for i in range(2 * (num_levels + 1))]
    pool = [None for i in range(num_levels + 1)]
    upsamp = [None for i in range(num_levels + 1)]
    print('Input:', points)
    gc[0] = graph_conv(points, cheby[0], 64, 'gc1')
    print('After gc1:', gc[0])
    for i in range(num_levels):
        pool[i] = tf.nn.pool(gc[i], [2], 'MAX', 'SAME', strides=[2])
        gc[i+1] = graph_conv(pool[i], cheby[i+1], 64, 'gc%d' % (i+2))
        print('After gc%d:' % (i+2), gc[i+1])

    pool[num_levels] = tf.reduce_max(pool[num_levels-1], axis=[1])
    upsamp[0] = tf.stack([pool[-1] for i in range(num_points // 2 ** num_levels)], axis=1)

    for i in range(num_levels):
        j = num_levels + 1 + i
        gc[j] = graph_conv(tf.concat([gc[num_levels-i], upsamp[i]], axis=2),
            cheby[num_levels-i], 64, 'gc%d' % (j+1))
        print('After gc%d:' % (j+1), gc[j])
        upsamp[i+1] = tf.keras.layers.UpSampling1D(2)(gc[j])

    gc[-1] = graph_conv(tf.concat([gc[0], upsamp[num_levels]], axis=2),
        cheby[0], num_parts, 'gc%d' % len(gc), activation_fn=None)
    print('After gc%d' % len(gc), gc[-1])
    return gc[-1]


def masked_sparse_softmax_cross_entropy(logits, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
        labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    loss = tf.reduce_sum(loss * mask, 1) / tf.reduce_sum(mask, 1)
    return tf.reduce_mean(loss)


def masked_accuracy(logits, labels, mask):
    """Accuracy with masking."""
    preds = tf.cast(tf.argmax(logits, axis=2), dtype=tf.int32)
    correct_prediction = tf.equal(preds, labels)
    accuracy = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    accuracy = tf.reduce_sum(accuracy * mask, 1) / tf.reduce_sum(mask, 1)
    return tf.reduce_mean(accuracy)

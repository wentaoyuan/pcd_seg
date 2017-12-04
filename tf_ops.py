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


def gcn(points, cheby, num_points, num_parts):
    with tf.device('/gpu:0'):
        gc1 = graph_conv(points, cheby[0], 64, 'gc1')
        print('+++ gc1:', gc1)
        pool1 = tf.nn.pool(gc1, [2], 'MAX', 'SAME', strides=[2])
        gc2 = graph_conv(pool1, cheby[1], 64, 'gc2')
        print('+++ gc2:', gc2)
        pool2 = tf.nn.pool(gc2, [2], 'MAX', 'SAME', strides=[2])
        gc3 = graph_conv(pool2, cheby[2], 64, 'gc3')
        print('+++ gc3:', gc3)
        bottleneck = tf.reduce_max(pool2, axis=[1])
        glob = tf.stack([bottleneck for i in range(num_points // 4)], axis=1);
        gc4 = graph_conv(tf.concat([glob, gc3], axis=2), cheby[2], 64, 'gc4')
        print('+++ gc4:', gc4)
        up1 = tf.keras.layers.UpSampling1D(2)(gc4)
        gc5 = graph_conv(tf.concat([gc2, up1], axis=2), cheby[1], 64, 'gc5')
        print('+++ gc5:', gc5)
        up2 = tf.keras.layers.UpSampling1D(2)(gc5)
        gc6 = graph_conv(tf.concat([gc1, up2], axis=2), cheby[0], num_parts, 'gc6',
            activation_fn=None)
        print('+++ gc6:', gc6)
    return gc6


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

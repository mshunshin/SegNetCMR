import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.framework import ops

def make_conv2d_layer(input, conv_size, output_layers, is_training=True, scope=None, batch_norm_decay_rate=0.99):
    with tf.variable_scope(scope, 'conv'):
        input_layers = input.get_shape().as_list()[3]

        weights = tf.Variable(tf.truncated_normal(shape=[conv_size, conv_size, input_layers, output_layers], stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(value=0.1, shape=[output_layers]), name='biases')

        conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
        conv_bias = tf.nn.bias_add(conv, biases)

        bn = tf.contrib.layers.batch_norm(conv_bias,
                                          center=True,
                                          scale=False,
                                          is_training=is_training,
                                          scope='bn',
                                          decay=batch_norm_decay_rate,
                                          updates_collections=ops.GraphKeys.UPDATE_OPS)
        result = tf.nn.relu(bn)

    return result

def unpool(value, name='unpool'):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat(values=[out, tf.zeros_like(out)], axis=i)
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out


########
#This is unpool_with_max_args
########
def unpool_with_argmax(updates, mask, ksize=[1, 2, 2, 1]):
    input_shape = updates.get_shape().as_list()
    #  calculation new shape
    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
    # calculation indices for batch, height, width and feature maps

    one_like_mask = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    f = one_like_mask * feature_range
    # transpose indices & reshape update values to one dimension
    updates_size = tf.size(updates)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(updates, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return ret



######
#As there is not a version of max_pool_with_args available for CPU you need an unpool that doesn't need it
######


@ops.RegisterGradient("MaxPoolWithArgmax")
def _MaxPoolGradWithArgmax(op, grad, unused_argmax_grad):
    return gen_nn_ops._max_pool_grad_with_argmax(op.inputs[0],
                                                 grad,
                                                 op.outputs[1],
                                                 op.get_attr("ksize"),
                                                 op.get_attr("strides"),
                                                 padding=op.get_attr("padding"))


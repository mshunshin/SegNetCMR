import tensorflow as tf

from .layers import make_conv2d_layer, unpool_with_argmax, unpool

def inference(images, is_training, conv_size=3, have_gpu=False, batch_norm_decay_rate=0.9):

    tf.summary.image('input', images, max_outputs=3)

    with tf.variable_scope('pool1'):
        result1a = make_conv2d_layer(images, conv_size, 64, is_training=is_training, scope='conva', batch_norm_decay_rate=batch_norm_decay_rate)
        result1b = make_conv2d_layer(result1a, conv_size, 64, is_training=is_training, scope='convb', batch_norm_decay_rate=batch_norm_decay_rate)
        if have_gpu:
            result1, arg1 = tf.nn.max_pool_with_argmax(result1b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')
        else:
            result1 = tf.nn.max_pool(result1b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')

    with tf.variable_scope('pool2'):
        result2a = make_conv2d_layer(result1, conv_size, 128, is_training=is_training, scope='conva', batch_norm_decay_rate=batch_norm_decay_rate)
        result2b = make_conv2d_layer(result2a, conv_size, 128, is_training=is_training, scope='convb', batch_norm_decay_rate=batch_norm_decay_rate)
        if have_gpu:
            result2, arg2 = tf.nn.max_pool_with_argmax(result2b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')
        else:
            result2 = tf.nn.max_pool(result2b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')

    with tf.variable_scope('pool3'):
        result3a = make_conv2d_layer(result2, conv_size, 256, is_training=is_training, scope='conva', batch_norm_decay_rate=batch_norm_decay_rate)
        result3b = make_conv2d_layer(result3a, conv_size, 256, is_training=is_training, scope='convb', batch_norm_decay_rate=batch_norm_decay_rate)
        result3c = make_conv2d_layer(result3b, conv_size, 256, is_training=is_training, scope='convc', batch_norm_decay_rate=batch_norm_decay_rate)
        if have_gpu:
            result3, arg3 = tf.nn.max_pool_with_argmax(result3c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')
        else:
            result3 = tf.nn.max_pool(result3c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')

    with tf.variable_scope('pool4'):
        result4a = make_conv2d_layer(result3, conv_size, 512, is_training=is_training, scope='conva', batch_norm_decay_rate=batch_norm_decay_rate)
        result4b = make_conv2d_layer(result4a, conv_size, 512, is_training=is_training, scope='convb', batch_norm_decay_rate=batch_norm_decay_rate)
        result4c = make_conv2d_layer(result4b, conv_size, 512, is_training=is_training, scope='convc', batch_norm_decay_rate=batch_norm_decay_rate)
        if have_gpu:
            result4, arg4 = tf.nn.max_pool_with_argmax(result4c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')
        else:
            result4 = tf.nn.max_pool(result4c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')

    with tf.variable_scope('pool5'):
        result5a = make_conv2d_layer(result4, conv_size, 512, is_training=is_training, scope='conva', batch_norm_decay_rate=batch_norm_decay_rate)
        result5b = make_conv2d_layer(result5a, conv_size, 512, is_training=is_training, scope='convb', batch_norm_decay_rate=batch_norm_decay_rate)
        result5c = make_conv2d_layer(result5b, conv_size, 512, is_training=is_training, scope='convc', batch_norm_decay_rate=batch_norm_decay_rate)
        if have_gpu:
            result5, arg5 = tf.nn.max_pool_with_argmax(result5c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')
        else:
            result5 = tf.nn.max_pool(result5c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')

    with tf.variable_scope('unpool5'):
        with tf.variable_scope('unpool'):
            if have_gpu:
                uresult5 = unpool_with_argmax(result5, arg5)
            else:
                uresult5 = unpool(result5)

        uresult5c = make_conv2d_layer(uresult5, conv_size, 512, is_training=is_training, scope='convc', batch_norm_decay_rate=batch_norm_decay_rate)
        uresult5b = make_conv2d_layer(uresult5c, conv_size, 512, is_training=is_training, scope='convb', batch_norm_decay_rate=batch_norm_decay_rate)
        uresult5a = make_conv2d_layer(uresult5b, conv_size, 512, is_training=is_training, scope='conva', batch_norm_decay_rate=batch_norm_decay_rate)

    with tf.variable_scope('unpool4'):
        with tf.variable_scope('unpool'):
            if have_gpu:
                uresult4 = unpool_with_argmax(uresult5a, arg4)
            else:
                uresult4 = unpool(uresult5a)

        uresult4c = make_conv2d_layer(uresult4, conv_size, 512, is_training=is_training, scope='convc', batch_norm_decay_rate=batch_norm_decay_rate)
        uresult4b = make_conv2d_layer(uresult4c, conv_size, 512, is_training=is_training, scope='convb', batch_norm_decay_rate=batch_norm_decay_rate)
        uresult4a = make_conv2d_layer(uresult4b, conv_size, 256, is_training=is_training, scope='conva', batch_norm_decay_rate=batch_norm_decay_rate)

    with tf.variable_scope('unpool3'):
        with tf.variable_scope('unpool'):
            if have_gpu:
                uresult3 = unpool_with_argmax(uresult4a, arg3)
            else:
                uresult3 = unpool(uresult4a)

        uresult3c = make_conv2d_layer(uresult3, conv_size, 256, is_training=is_training, scope='convc', batch_norm_decay_rate=batch_norm_decay_rate)
        uresult3b = make_conv2d_layer(uresult3c, conv_size, 256, is_training=is_training, scope='convb', batch_norm_decay_rate=batch_norm_decay_rate)
        uresult3a = make_conv2d_layer(uresult3b, conv_size, 128, is_training=is_training, scope='conva', batch_norm_decay_rate=batch_norm_decay_rate)

    with tf.variable_scope('unpool2'):
        with tf.variable_scope('unpool'):
            if have_gpu:
                uresult2 = unpool_with_argmax(uresult3a, arg2)
            else:
                uresult2 = unpool(uresult3a)

        uresult2b = make_conv2d_layer(uresult2, conv_size, 128, is_training=is_training, scope='convb', batch_norm_decay_rate=batch_norm_decay_rate)
        uresult2a = make_conv2d_layer(uresult2b, conv_size, 64, is_training=is_training, scope='conva', batch_norm_decay_rate=batch_norm_decay_rate)

    with tf.variable_scope('unpool1'):
        with tf.variable_scope('unpool'):
            if have_gpu:
                uresult1 = unpool_with_argmax(uresult2a, arg1)
            else:
                uresult1 = unpool(uresult2a)

        uresult1b = make_conv2d_layer(uresult1, conv_size, 64, is_training=is_training, scope='convb', batch_norm_decay_rate=batch_norm_decay_rate)
        uresult1a = make_conv2d_layer(uresult1b, conv_size, 2, is_training=is_training, scope='conva', batch_norm_decay_rate=batch_norm_decay_rate)

    return uresult1a


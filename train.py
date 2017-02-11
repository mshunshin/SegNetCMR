######
# batch_norm use: http://stackoverflow.com/questions/40081697/getting-low-test-accuracy-using-tensorflow-batch-norm-function
#
#
######

import os
import sys
import time
import random

import scipy.misc

import numpy as np

import tensorflow as tf

from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.framework import ops


#####JAMES These should be True and 20 for you######
HAVE_GPU = False
SAVE_INTERVAL = 2

TRAINING_DIR = './Data/Training'
TEST_DIR = './Data/Test'

RUN_NAME = "Run3x3"
CONV_SIZE = 3

ROOT_LOG_DIR = './Output'
CHECKPOINT_FN = 'model.ckpt'

#Set this to 0.9 at the beginning, so that the test data doesn't look stupid for the first few iterations, then to 0.99 or 0.999 as you go on.
BATCH_NORM_DECAY = 0.9

MAX_STEPS = 20000
BATCH_SIZE = 6

LOG_DIR = os.path.join(ROOT_LOG_DIR, RUN_NAME)
CHECKPOINT_FL = os.path.join(LOG_DIR, CHECKPOINT_FN)

def main():
    training_data = GetData(TRAINING_DIR)
    test_data = GetData(TEST_DIR)

    g = tf.Graph()

    with g.as_default():

        images, labels, is_training = placeholder_inputs()

        logits = inference(images=images, is_training=is_training)

        make_output_image(images=images, logits=logits, labels=labels)

        loss = loss_calc(logits=logits, labels=labels)

        train_op, global_step = training(loss=loss, learning_rate=1e-04)

        accuracy = evaluation(logits=logits, labels=labels)

        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver([x for x in tf.global_variables() if 'Adam' not in x.name])

        sm = tf.train.SessionManager()

        with sm.prepare_session("", init_op=init, saver=saver, checkpoint_dir=LOG_DIR) as sess:

            sess.run(tf.variables_initializer([x for x in tf.global_variables() if 'Adam' in x.name]))

            train_writer = tf.summary.FileWriter(LOG_DIR + "/Train", sess.graph)
            test_writer = tf.summary.FileWriter(LOG_DIR + "/Test")

            global_step_value, = sess.run([global_step])

            print("Last trained iteration was: ", global_step_value)

            for step in range(global_step_value+1, global_step_value+MAX_STEPS+1):

                print("Iteration: ", step)

                images_batch, labels_batch = training_data.next_batch(BATCH_SIZE)

                train_feed_dict = {images: images_batch,
                                   labels: labels_batch,
                                   is_training: True}

                _, train_loss_value, train_accuracy_value, train_summary_str = sess.run([train_op, loss, accuracy, summary], feed_dict=train_feed_dict)

                if step % SAVE_INTERVAL == 0:

                    print("Train Loss: ", train_loss_value)
                    print("Train accuracy: ", train_accuracy_value)
                    train_writer.add_summary(train_summary_str, step)
                    train_writer.flush()

                    images_batch, labels_batch = test_data.next_batch(BATCH_SIZE)

                    test_feed_dict = {images: images_batch,
                                      labels: labels_batch,
                                      is_training: False}

                    test_loss_value, test_accuracy_value, test_summary_str = sess.run([loss, accuracy, summary], feed_dict=test_feed_dict)

                    print("Test Loss: ", test_loss_value)
                    print("Test accuracy: ", test_accuracy_value)
                    test_writer.add_summary(test_summary_str, step)
                    test_writer.flush()



                    saver.save(sess, CHECKPOINT_FL, global_step=step)
                    print("Session Saved")
                    print("================")



class GetData():
    def __init__(self, data_dir):
        images_list =[]
        labels_list = []

        self.source_list = []

        examples = 0
        print("loading images")
        label_dir = os.path.join(data_dir, "Labels")
        image_dir = os.path.join(data_dir, "Images")
        for label_root, dir, files in os.walk(label_dir):
            for file in files:
                if not file.endswith((".png", ".jpg", ".gif")):
                    continue
                try:
                    folder = os.path.relpath(label_root, label_dir)
                    image_root = os.path.join(image_dir, folder)


                    image = scipy.misc.imread(os.path.join(image_root, file))
                    label = scipy.misc.imread(os.path.join(label_root, file))

                    images_list.append(image[...,0][...,None]/255)
                    labels_list.append((label[...,0]>1).astype(np.int64))
                    examples = examples + 1
                except Exception as e:
                    print(e)
        print("finished loading images")
        self.examples = examples
        print("Number of examples found: ", examples)
        self.images = np.array(images_list)
        self.labels = np.array(labels_list)

    def next_batch(self, batch_size):

        if len(self.source_list) < batch_size:
            new_source = list(range(self.examples))
            random.shuffle(new_source)
            self.source_list.extend(new_source)

        examples_idx = self.source_list[:batch_size]
        del self.source_list[:batch_size]

        return self.images[examples_idx,...], self.labels[examples_idx,...]


def placeholder_inputs():

    images = tf.placeholder(tf.float32, [BATCH_SIZE, 256, 256, 1])
    labels = tf.placeholder(tf.int64, [BATCH_SIZE, 256, 256])
    is_training = tf.placeholder(tf.bool)

    return images, labels, is_training


def inference(images, is_training):

    tf.summary.image('input', images, max_outputs=3)

    with tf.variable_scope('pool1'):
        result1a = make_conv2d_layer(images, CONV_SIZE, 64, is_training=is_training, scope='conva')
        result1b = make_conv2d_layer(result1a, CONV_SIZE, 64, is_training=is_training, scope='convb')
        if HAVE_GPU:
            result1, arg1 = tf.nn.max_pool_with_argmax(result1b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')
        else:
            result1 = tf.nn.max_pool(result1b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')

    with tf.variable_scope('pool2'):
        result2a = make_conv2d_layer(result1, CONV_SIZE, 128, is_training=is_training, scope='conva')
        result2b = make_conv2d_layer(result2a, CONV_SIZE, 128, is_training=is_training, scope='convb')
        if HAVE_GPU:
            result2, arg2 = tf.nn.max_pool_with_argmax(result2b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')
        else:
            result2 = tf.nn.max_pool(result2b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')

    with tf.variable_scope('pool3'):
        result3a = make_conv2d_layer(result2, CONV_SIZE, 256, is_training=is_training, scope='conva')
        result3b = make_conv2d_layer(result3a, CONV_SIZE, 256, is_training=is_training, scope='convb')
        result3c = make_conv2d_layer(result3b, CONV_SIZE, 256, is_training=is_training, scope='convc')
        if HAVE_GPU:
            result3, arg3 = tf.nn.max_pool_with_argmax(result3c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')
        else:
            result3 = tf.nn.max_pool(result3c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')

    with tf.variable_scope('pool4'):
        result4a = make_conv2d_layer(result3, CONV_SIZE, 512, is_training=is_training, scope='conva')
        result4b = make_conv2d_layer(result4a, CONV_SIZE, 512, is_training=is_training, scope='convb')
        result4c = make_conv2d_layer(result4b, CONV_SIZE, 512, is_training=is_training, scope='convc')
        if HAVE_GPU:
            result4, arg4 = tf.nn.max_pool_with_argmax(result4c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')
        else:
            result4 = tf.nn.max_pool(result4c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')

    with tf.variable_scope('pool5'):
        result5a = make_conv2d_layer(result4, CONV_SIZE, 512, is_training=is_training, scope='conva')
        result5b = make_conv2d_layer(result5a, CONV_SIZE, 512, is_training=is_training, scope='convb')
        result5c = make_conv2d_layer(result5b, CONV_SIZE, 512, is_training=is_training, scope='convc')
        if HAVE_GPU:
            result5, arg5 = tf.nn.max_pool_with_argmax(result5c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')
        else:
            result5 = tf.nn.max_pool(result5c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')

    with tf.variable_scope('unpool5'):
        with tf.variable_scope('unpool'):
            if HAVE_GPU:
                uresult5 = unpool_with_argmax(result5, arg5)
            else:
                uresult5 = unpool(result5)

        uresult5c = make_conv2d_layer(uresult5, CONV_SIZE, 512, is_training=is_training, scope='convc')
        uresult5b = make_conv2d_layer(uresult5c, CONV_SIZE, 512, is_training=is_training, scope='convb')
        uresult5a = make_conv2d_layer(uresult5b, CONV_SIZE, 512, is_training=is_training, scope='conva')

    with tf.variable_scope('unpool4'):
        with tf.variable_scope('unpool'):
            if HAVE_GPU:
                uresult4 = unpool_with_argmax(uresult5a, arg4)
            else:
                uresult4 = unpool(uresult5a)

        uresult4c = make_conv2d_layer(uresult4, CONV_SIZE, 512, is_training=is_training, scope='convc')
        uresult4b = make_conv2d_layer(uresult4c, CONV_SIZE, 512, is_training=is_training, scope='convb')
        uresult4a = make_conv2d_layer(uresult4b, CONV_SIZE, 256, is_training=is_training, scope='conva')

    with tf.variable_scope('unpool3'):
        with tf.variable_scope('unpool'):
            if HAVE_GPU:
                uresult3 = unpool_with_argmax(uresult4a, arg3)
            else:
                uresult3 = unpool(uresult4a)

        uresult3c = make_conv2d_layer(uresult3, CONV_SIZE, 256, is_training=is_training, scope='convc')
        uresult3b = make_conv2d_layer(uresult3c, CONV_SIZE, 256, is_training=is_training, scope='convb')
        uresult3a = make_conv2d_layer(uresult3b, CONV_SIZE, 128, is_training=is_training, scope='conva')

    with tf.variable_scope('unpool2'):
        with tf.variable_scope('unpool'):
            if HAVE_GPU:
                uresult2 = unpool_with_argmax(uresult3a, arg2)
            else:
                uresult2 = unpool(uresult3a)

        uresult2b = make_conv2d_layer(uresult2, CONV_SIZE, 128, is_training=is_training, scope='convb')
        uresult2a = make_conv2d_layer(uresult2b, CONV_SIZE, 64, is_training=is_training, scope='conva')

    with tf.variable_scope('unpool1'):
        with tf.variable_scope('unpool'):
            if HAVE_GPU:
                uresult1 = unpool_with_argmax(uresult2a, arg1)
            else:
                uresult1 = unpool(uresult2a)

        uresult1b = make_conv2d_layer(uresult1, CONV_SIZE, 64, is_training=is_training, scope='convb')
        uresult1a = make_conv2d_layer(uresult1b, CONV_SIZE, 2, is_training=is_training, scope='conva')

    return uresult1a

def make_conv2d_layer(input, conv_size, output_layers, is_training=True, scope=None):
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
                                          decay=BATCH_NORM_DECAY,
                                          updates_collections=ops.GraphKeys.UPDATE_OPS)
        result = tf.nn.relu(bn)

    return result

def make_output_image(images, logits, labels):
    cast_labels = tf.cast(labels, tf.uint8) * 128
    cast_labels = cast_labels[...,None]
    tf.summary.image('input_labels', cast_labels, max_outputs=3)

    classification1 = tf.nn.softmax(logits = logits, dim=-1)[...,1]
    output_image_gb = images[...,0]
    output_image_r = classification1 + tf.multiply(images[...,0], (1-classification1))
    output_image = tf.stack([output_image_r, output_image_gb, output_image_gb], axis=3)
    tf.summary.image('output_mixed', output_image, max_outputs=3)

    output_image_binary = tf.argmax(logits, 3)
    output_image_binary = tf.cast(output_image_binary[...,None], tf.float32) * 128/255
    tf.summary.image('output_labels', output_image_binary, max_outputs=3)

def loss_calc(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss', loss)
    return loss

def training(loss, learning_rate):

    global_step = tf.Variable(0, name='global_step', trainable=False)

    #This motif is needed to hook up the batch_norm updates to the training
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss, global_step=global_step)

    return train_op, global_step

def evaluation(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 3), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy


################
##Functions that should be in tensorflow but arn't
################

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
            out = tf.concat(i, [out, tf.zeros_like(out)])
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out

@ops.RegisterGradient("MaxPoolWithArgmax")
def _MaxPoolGradWithArgmax(op, grad, unused_argmax_grad):
    return gen_nn_ops._max_pool_grad_with_argmax(op.inputs[0],
                                                 grad,
                                                 op.outputs[1],
                                                 op.get_attr("ksize"),
                                                 op.get_attr("strides"),
                                                 padding=op.get_attr("padding"))

if __name__ == '__main__':
    main()
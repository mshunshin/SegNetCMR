import os

import tensorflow as tf

import SegNetCMR as sn

HAVE_GPU = False
SAVE_INTERVAL = 2

TRAINING_DIR = './Data/Training'
TEST_DIR = './Data/Test'

RUN_NAME = "Run3x3"
CONV_SIZE = 3

ROOT_LOG_DIR = './Output'
CHECKPOINT_FN = 'model.ckpt'

#Start off at 0.9, then increase.
BATCH_NORM_DECAY_RATE = 0.9

MAX_STEPS = 20000
BATCH_SIZE = 6

LOG_DIR = os.path.join(ROOT_LOG_DIR, RUN_NAME)
CHECKPOINT_FL = os.path.join(LOG_DIR, CHECKPOINT_FN)

def main():
    training_data = sn.GetData(TRAINING_DIR)
    test_data = sn.GetData(TEST_DIR)

    g = tf.Graph()

    with g.as_default():

        images, labels, is_training = sn.placeholder_inputs(batch_size=BATCH_SIZE)

        logits = sn.inference(images=images, is_training=is_training, conv_size=CONV_SIZE, batch_norm_decay_rate=BATCH_NORM_DECAY_RATE)

        sn.add_output_images(images=images, logits=logits, labels=labels)

        loss = sn.loss_calc(logits=logits, labels=labels)

        train_op, global_step = sn.training(loss=loss, learning_rate=1e-04)

        accuracy = sn.evaluation(logits=logits, labels=labels)

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


if __name__ == '__main__':
    main()
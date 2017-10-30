import tensorflow as tf

def add_output_images(images, logits, labels, max_outputs=3):

    tf.summary.image('input', images, max_outputs=max_outputs)

    output_image_bw = images[..., 0]

    labels1 = tf.cast(labels[...,0], tf.float32)

    input_labels_image_r = labels1 + (output_image_bw * (1-labels1))
    input_labels_image = tf.stack([input_labels_image_r, output_image_bw, output_image_bw], axis=3)
    tf.summary.image('input_labels_mixed', input_labels_image, max_outputs=3)

    classification1 = tf.nn.softmax(logits = logits, dim=-1)[...,1]

    output_labels_image_r = classification1 + (output_image_bw * (1-classification1))
    output_labels_image = tf.stack([output_labels_image_r, output_image_bw, output_image_bw], axis=3)
    tf.summary.image('output_labels_mixed', output_labels_image, max_outputs=3)

    return


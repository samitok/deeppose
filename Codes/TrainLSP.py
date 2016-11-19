from LSPGlobals import FLAGS
import GetLSPData
import LSPModels
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
import time
from datetime import datetime
import LSPGlobals
from LSPDrawLines import draw_pose_on_image as draw

# Constants used for dealing with the files, matches convert_to_records.

train_set_file = os.path.join(FLAGS.data_dir, 'train.tfrecords')
validation_set_file = os.path.join(FLAGS.data_dir, 'validation.tfrecords')


def main():

    if not (os.path.exists(train_set_file) & os.path.exists(validation_set_file)):
        GetLSPData.main()

    if not gfile.Exists(FLAGS.train_dir):
        gfile.MakeDirs(FLAGS.train_dir)
    
    train()


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'label': tf.FixedLenFeature([LSPGlobals.TotalLabels], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })

    # now return the converted data
    image_as_vector = tf.decode_raw(features['image_raw'], tf.uint8)
    image_as_vector.set_shape([LSPGlobals.TotalImageBytes])
    image = tf.reshape(image_as_vector, [FLAGS.input_size, FLAGS.input_size, FLAGS.input_depth])
    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image_float = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return label, image_float

    
def inputs(is_train):
    """Reads input data num_epochs times."""
    filename = train_set_file if is_train else validation_set_file

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=None)

        # get single examples
        label, image = read_and_decode(filename_queue)

        # groups examples into batches randomly
        images_batch, labels_batch = tf.train.shuffle_batch(
            [image, label], batch_size=FLAGS.batch_size,
            capacity=3000,
            min_after_dequeue=1000)

        return images_batch, labels_batch
 

def train():
    with tf.Graph().as_default():
        # Global step variable for tracking processes.
        global_step = tf.Variable(0, trainable=False)

        # Prepare data batches
        train_set_batch, train_label_batch = inputs(is_train=True)
        validation_set_batch, validation_label_batch = inputs(is_train=False)

        # Placeholder to switch between train and test sets.
        image_batch = tf.placeholder(tf.float32,
                                     shape=[FLAGS.batch_size, FLAGS.input_size, FLAGS.input_size, FLAGS.input_depth])
        label_batch = tf.placeholder(tf.int32,
                                     shape=[FLAGS.batch_size, LSPGlobals.TotalLabels])
        keep_probability = tf.placeholder(tf.float32)
        
        # Build a Graph that computes the logits predictions from the inference model.
        logits = LSPModels.inference(image_batch, keep_prob=keep_probability)
        
        # Calculate loss.
        loss, mean_pixel_error = LSPModels.loss(logits, label_batch)
        
        # Build a Graph that trains the model with one batch of examples and updates the model parameters.
        train_op = LSPModels.train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())
        
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()
        
        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()
        
        with tf.Session() as sess:
            # Start populating the filename queue.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            
            sess.run(init)
            
            step_init = 0
            checkpoint = tf.train.get_checkpoint_state(FLAGS.train_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, checkpoint.model_checkpoint_path)
                step_init = sess.run(global_step)
            else:
                print("No checkpoint found...")

            summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph=sess.graph)
            
            for step in range(step_init, FLAGS.max_steps):
                
                start_time = time.time()
                images, labels = sess.run([train_set_batch, train_label_batch])
                feed_dict = {image_batch: images,
                             label_batch: labels,
                             keep_probability: 0.75}
                _, pixel_error_value = sess.run([train_op, mean_pixel_error], feed_dict=feed_dict)
                duration = time.time() - start_time

                if not step == 0:
                    # Print current results.
                    if step % 10 == 0:
                        num_examples_per_step = FLAGS.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)

                        format_str = '%s: step %d, MeanPixelError = %.1f pixels (%.1f examples/sec; %.3f sec/batch)'
                        print(format_str % (datetime.now(), step, pixel_error_value,
                                            examples_per_sec, sec_per_batch))

                    # Check results for validation set
                    if (step % 200 == 0) and (step != 0):
                        images, labels = sess.run([validation_set_batch, validation_label_batch])
                        feed_dict = {image_batch: images,
                                     label_batch: labels,
                                     keep_probability: 1}
                        produced_labels, pixel_error_value = sess.run([logits, mean_pixel_error], feed_dict=feed_dict)

                        draw(images[0, ...], produced_labels[0, ...], FLAGS.drawing_dir, step/100)
                        print('Test Set MeanPixelError: %.1f pixels' % pixel_error_value)

                    # Add summary to summary writer
                    if (step % 500 == 0) and (step != 0):
                        summary_str = sess.run(summary_op, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)

                    # Save the model checkpoint periodically.
                    if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)
                        print('Model checkpoint saved for step %d' % step)
        
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    main()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from LSPGlobals import FLAGS
import GetLSPData
import LSPModels
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import time
from datetime import datetime
import LSPGlobals
import math
import matplotlib.pyplot as plt

def main():
    trainLabelsFileName = os.path.join(FLAGS.data_dir, FLAGS.trainLabels_fn)
    testLabelsFileName = os.path.join(FLAGS.data_dir, FLAGS.testLabels_fn)
    if not (os.path.exists(trainLabelsFileName) & os.path.exists(testLabelsFileName)):
        GetLSPData.main()
    if gfile.Exists(FLAGS.eval_dir):
        gfile.DeleteRecursively(FLAGS.eval_dir)
    gfile.MakeDirs(FLAGS.eval_dir)
    
    eval(testLabelsFileName)
        
def read_my_file_format(filename_queue):
    #reader = tf.TextLineReader()
    #_, value = reader.read(filename_queue)
    
    #image_filename = tf.train.string_input_producer([value])
    image_reader = tf.WholeFileReader()
    _, image_data = image_reader.read(filename_queue)
    
    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(image_data, tf.uint8)
    
    '''
    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    record_defaults = [['']] + [[1.0] for _ in range(28)]
    columns = tf.decode_csv(value, record_defaults=record_defaults)
    '''
    # The first bytes represent the label, which we convert from uint8->int32.
    labels_ = tf.cast(tf.slice(record_bytes, [0], [LSPGlobals.TotalLabels]), tf.float32)
    
    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(tf.slice(record_bytes, [LSPGlobals.TotalLabels], [LSPGlobals.TotalImageBytes]),
                          [FLAGS.input_depth, FLAGS.input_size, FLAGS.input_size])
    # Convert from [depth, height, width] to [height, width, depth].
    processed_example = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)
    

    return processed_example, labels_


def input_pipeline(argfilenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        argfilenames, num_epochs=num_epochs, shuffle=True)
    
    image_file, label = read_my_file_format(filename_queue)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    
    example_batch, label_batch = tf.train.shuffle_batch(
        [image_file, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    '''
    reader = tf.WholeFileReader()
    key, value = reader.read(filenames_batch)

    example_batch = tf.image.decode_jpeg(value, name='decode_jpeg')
   '''
    #example_batch = GetLSPData.produce_image_batch(filenames_batch)
    
    return example_batch, label_batch
    

def eval_once(saver, logits, labels_, imgage):
    """Run Eval once.
    
    Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            #global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return
        
        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                               start=True))
            
            or_labels, predictions, imageFile = sess.run([labels_,logits,imgage])
            
            print('%s:' % datetime.now())
            print(predictions)
            print(or_labels)
            imgreordered = np.squeeze(imageFile)
            plt.imshow(imgreordered)
            plt.show()
        
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
        
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

    
    
def eval(testLabelsFileName):
    with open(testLabelsFileName) as f:
        testSet = f.read().splitlines()
        
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        
        testSet_batch, testLabel_batch = input_pipeline(testSet, 1, num_epochs=None)
        # Build a Graph that computes the logits predictions from the
        # inference model.
        
        logits = LSPModels.inference(testSet_batch, 1)
        
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(LSPGlobals.MOVING_AVERAGE_DECAY)
        variables_to_restore = {}
        for v in tf.all_variables():
          if v in tf.trainable_variables():
            restore_name = variable_averages.average_name(v)
          else:
            restore_name = v.op.name
          variables_to_restore[restore_name] = v
        saver = tf.train.Saver(variables_to_restore)
     
        #while True:
        eval_once(saver, logits, testLabel_batch, testSet_batch)
        #time.sleep(11)


if __name__ == '__main__':
    main()

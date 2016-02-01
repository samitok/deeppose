from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from LSPGlobals import FLAGS
import LSPModels
import tensorflow as tf
import time
from datetime import datetime
from LSPDrawLines import drawPoseOnImage as draw
import glob
import math
import numpy as np
import PIL.Image as Image
from os.path import basename as b

def main():    
    imagelist = sorted(glob.glob(FLAGS.input_dir + '*.' + FLAGS.input_type))
    if not len(imagelist):
        print('No input found!')
        return
    
    evalDeepPose(imagelist)
    
def resizeLabels(argLabels, argResize):
    for i in xrange(argLabels.shape[0]):
        argLabels[i,:,0] *= argResize[i,0]
        argLabels[i,:,1] *= argResize[i,1]   
         
    return argLabels

def evalDeepPose(imagelist):            
    with tf.Graph().as_default():
        # Placeholder to switch between train and test sets.
        dataShape = [FLAGS.batch_size, FLAGS.input_size, FLAGS.input_size, FLAGS.input_depth]
        input_images = tf.placeholder(tf.float32, shape=dataShape)

        # Build a Graph that computes the logits predictions from the inference model.
        logits = LSPModels.inference(input_images, FLAGS.batch_size, keepProb=1)
        
        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())
        
        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()
        
        with tf.Session() as sess:
            # Start populating the filename queue.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            
            sess.run(init)
            
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("No checkpoint found!")
                return                
            
            numSteps = int(math.floor(len(imagelist)/FLAGS.batch_size)+0.1)
            if (len(imagelist)%FLAGS.batch_size > 0):
                numSteps = numSteps +1

            resizeValues = np.empty([FLAGS.batch_size, 2], dtype=np.float32)
            input_pack = np.empty(dataShape, dtype=np.float32)
            
            for step in xrange(numSteps):
                
                for i in xrange(FLAGS.batch_size):
                    index = i + step*FLAGS.batch_size
                    if not index >= len(imagelist):
                        imgFile = Image.open(imagelist[index])
                        (imWidth, imHeight) = imgFile.size
                        imgFile = imgFile.resize((FLAGS.input_size,FLAGS.input_size), Image.ANTIALIAS)
                
                        resizeValues[i, 0] = float(imWidth)/FLAGS.input_size
                        resizeValues[i, 1] = float(imHeight)/FLAGS.input_size
                        
                        input_pack[i,...] = np.asarray(imgFile)
                
                start_time = time.time()
                labels = sess.run([logits], feed_dict={input_images: input_pack})
                duration = time.time() - start_time
                             
                labels_reshaped = np.asarray(labels).reshape([FLAGS.batch_size, FLAGS.label_count, FLAGS.label_size])
                resizedLabels = resizeLabels(labels_reshaped, resizeValues)
                    
                for i in xrange(FLAGS.batch_size):
                    index = i + step*FLAGS.batch_size
                    if not index >= len(imagelist):
                        imgFile = Image.open(imagelist[index])
                        
                        img = np.asarray(imgFile).astype(np.uint8)
                        draw(img, resizedLabels[i,...], FLAGS.output_dir, 0, fname=b(imagelist[index]), reshaped=True)   
                             
                examples_per_sec = FLAGS.batch_size / duration   
                format_str = ('%s: Done: %d, (FeedForward Time: %.1f examples/sec)')
                print (format_str % (datetime.now(), (step+1)*FLAGS.batch_size, examples_per_sec))
                        

            coord.request_stop()
            coord.join(threads)
            print('Process Finished...')

if __name__ == '__main__':
    main()

import tensorflow as tf
import LSPGlobals
from LSPGlobals import FLAGS
import re


def _activation_summary(x):
    """Helper to create summaries for activations.
    
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % LSPGlobals.TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))



def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    
    Returns:
      Variable Tensor
    """
    var = tf.Variable(tf.random_normal(shape, stddev=stddev), name=name)
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var



def inference(images, batch_size, keepProb=1):
    """    
    Args:
      images: Images returned from distorted_inputs() or inputs().
    
    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = tf.Variable(tf.random_normal([11, 11, 3, 20], stddev=1e-4), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 5, 5, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[20]), name='biases')
        
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)
    
    # norm1
    norm1 = tf.nn.lrn(conv1, name='lrn1')
    
    
    # pool1
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1],
                           padding='SAME', name='pool1')
    
    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = tf.Variable(tf.random_normal([5, 5, 20, 35], stddev=1e-4), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME') #use_cudnn_on_gpu=False,
        biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[35]), name='biases')
        
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)
    
    # norm2
    norm2 = tf.nn.lrn(conv2, name='lrn2')
    #norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    
    '''
    # conv2
    with tf.variable_scope('conv3') as scope:
        kernel = tf.Variable(tf.random_normal([3, 3, 1, 192], stddev=1e-4), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[192]), name='biases')
        
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv3)
        '''
        
    
    # conv4
    with tf.variable_scope('conv4') as scope:
        kernel = tf.Variable(tf.random_normal([3, 3, 35, 50], stddev=1e-4), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME') #use_cudnn_on_gpu=False,
        biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[50]), name='biases')
        
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv4)
        
        
    
    # conv5
    with tf.variable_scope('conv5') as scope:
        kernel = tf.Variable(tf.random_normal([3, 3, 50, 75], stddev=1e-4), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME') #use_cudnn_on_gpu=False,
        biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[75]), name='biases')
        
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv5)
        
        
    pool3 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    
    # local3
    with tf.variable_scope('local1') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        dim = 1
        for d in pool3.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(pool3, [batch_size, dim])
        
        weights = _variable_with_weight_decay('weights', shape=[dim, 1024], stddev=0.04, wd=0.004)
        biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[1024]), name='biases')
        local1 = tf.nn.relu_layer(reshape, weights, biases, name=scope.name)
        #if train:
        #    local1 = tf.nn.dropout(local1, 0.5)
        _activation_summary(local1)
    
    # local4
    with tf.variable_scope('local2') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1024, 1024], stddev=0.04, wd=0.004)        
        biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[1024]), name='biases')
        
        local2 = tf.nn.relu_layer(local1, weights, biases, name=scope.name)
        #if train:
        #    local2 = tf.nn.dropout(local2, 0.5)
        _activation_summary(local2)
    
    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(tf.random_normal([1024, LSPGlobals.TotalLabels], stddev=1/1024.0), name='weights')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[LSPGlobals.TotalLabels]), name='biases')
        softmax_linear = tf.nn.xw_plus_b(local2, weights, biases, name=scope.name)
        #dropped_softmax_linear = tf.nn.dropout(softmax_linear, keepProb)
        _activation_summary(softmax_linear)
    
    return softmax_linear




def loss(logits, labels):
    """Calculates Mean Pixel Error.
    
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]
    
    Returns:
      Loss tensor of type float.
    """
    
    labelValidity = tf.sign(labels, name='label_validity')
    
    minop = tf.sub(logits, labels, name='Diff_Op')
    
    absop = tf.abs(minop, name='Abs_Op')
    
    lossValues = tf.mul(labelValidity, absop, name='lossValues')
    
    loss_mean = tf.reduce_mean(lossValues, name='MeanPixelError')
    
    tf.add_to_collection('losses', loss_mean)
    
    return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss_mean
   


def _add_loss_summaries(total_loss):
    """Add summaries for losses in DeepPose model.
    
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    
    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name +' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))
    
    return loss_averages_op

  

def train(total_loss, global_step):
    """Train DeepPose model.
    
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    
    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = FLAGS.example_per_epoch / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
    
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(FLAGS.initial_learn_rate,
                                    global_step,
                                    decay_steps,
                                    FLAGS.learn_decay_factor,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)
    
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)
    
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    
    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
    
    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)
    
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    
    return train_op

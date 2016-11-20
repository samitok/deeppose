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


def print_size(x, name=''):
    if not name == '':
        print(name + ': ' + str(x.get_shape()))
    else:
        print(x.get_shape())


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
    '''if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)'''
    return var


def inference(images, keep_prob=1):
    """    
    Args:
      images: Images returned from distorted_inputs() or inputs().
      keep_prob: keep probability for dropout.
    
    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().

    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = tf.Variable(tf.random_normal([11, 11, 3, 96], stddev=1e-4), name='weights')
        conv1 = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME', name=scope.name)
        _activation_summary(conv1)
    print('Printing DeepPose network')
    print_size(conv1, name='conv1')

    # lrn1
    lrn1 = tf.nn.lrn(conv1, name='lrn1')
    print_size(lrn1, name='lrn1')

    # pool1
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    print_size(pool1, name='pool1')
    
    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = tf.Variable(tf.random_normal([5, 5, 96, 256], stddev=1e-4), name='weights')
        conv2 = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        _activation_summary(conv2)
    print_size(conv2, name='conv2')
    
    # lrn2
    lrn2 = tf.nn.lrn(conv2, name='lrn2')
    print_size(lrn2, name='lrn2')

    # pool2
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    print_size(pool2, name='pool2')

    # conv3
    with tf.variable_scope('conv3') as scope:
        kernel = tf.Variable(tf.random_normal([3, 3, 256, 384], stddev=1e-4), name='weights')
        conv3 = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        _activation_summary(conv3)
    print_size(conv3, name='conv3')

    # conv4
    with tf.variable_scope('conv4') as scope:
        kernel = tf.Variable(tf.random_normal([3, 3, 384, 384], stddev=1e-4), name='weights')
        conv4 = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        _activation_summary(conv4)
    print_size(conv4, name='conv4')

    # conv5
    with tf.variable_scope('conv5') as scope:
        kernel = tf.Variable(tf.random_normal([3, 3, 384, 256], stddev=1e-4), name='weights')
        conv5 = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        _activation_summary(conv5)
    print_size(conv5, name='conv5')

    pool3 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool3')
    print_size(pool3, name='pool3')

    # Move everything into depth so we can perform a single matrix multiply.
    image_size_after_pool3 = 6*6*256
    image_as_vector = tf.reshape(pool3, [FLAGS.batch_size, image_size_after_pool3])

    # full1
    with tf.variable_scope('full1') as scope:
        weights = tf.Variable(tf.random_normal([image_size_after_pool3, 4096], stddev=0.04), name='weights')
        biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[4096]), name='biases')

        full1 = tf.nn.relu(tf.matmul(image_as_vector, weights) + biases, name=scope.name)
        # full1_dropped = tf.nn.dropout(full1, keep_prob)
        _activation_summary(full1)
    print_size(full1, name='full1')

    # full2
    with tf.variable_scope('full2') as scope:
        weights = tf.Variable(tf.random_normal([4096, 4096], stddev=0.04), name='weights')
        biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[4096]), name='biases')

        full2 = tf.nn.relu(tf.matmul(full1, weights) + biases, name=scope.name)
        # full2_dropped = tf.nn.dropout(full2, keep_prob)
        _activation_summary(full2)
    print_size(full2, name='full2')

    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax') as scope:
        weights = tf.Variable(tf.random_normal([4096, LSPGlobals.TotalLabels], stddev=0.04), name='weights')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[LSPGlobals.TotalLabels]), name='biases')
        softmax = tf.nn.xw_plus_b(full2, weights, biases, name=scope.name)
        _activation_summary(softmax)
    print_size(softmax, name='softmax')
    
    return softmax


def loss(logits, labels):
    """Calculates Mean Pixel Error.
    
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]
    
    Returns:
      Loss tensor of type float.
    """
    with tf.variable_scope('total_loss') as scope:
        label_validity = tf.cast(tf.sign(labels, name='label_validity'), tf.float32)

        labels_float = tf.cast(labels, tf.float32)

        minus_op = tf.sub(logits, labels_float, name='Diff_Op')

        abs_op = tf.abs(minus_op, name='Abs_Op')

        loss_values = tf.mul(label_validity, abs_op, name='lossValues')

        loss_mean = tf.reduce_mean(loss_values, name='MeanPixelError')

        tf.add_to_collection('losses', loss_mean)

        return tf.add_n(tf.get_collection('losses'), name=scope.name), loss_mean
   

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
    
    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name + ' (raw)', l)
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

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    var_list_all = tf.trainable_variables()
    var_list1 = var_list_all[:5]
    var_list2 = var_list_all[5:]

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt1 = tf.train.GradientDescentOptimizer(FLAGS.initial_learn_rate*10)
        opt2 = tf.train.GradientDescentOptimizer(FLAGS.initial_learn_rate)
        # grads = opt.compute_gradients(total_loss)
        # grads = tf.gradients(total_loss, var_list1 + var_list2)
        grads1 = opt1.compute_gradients(total_loss, var_list1)
        grads2 = opt2.compute_gradients(total_loss, var_list2)
        # grads1 = grads[:len(var_list1)]
        # grads2 = grads[len(var_list1):]

    # Apply gradients.
    # apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    apply_gradient_op1 = opt1.apply_gradients(grads1, global_step=global_step)
    apply_gradient_op2 = opt2.apply_gradients(grads2, global_step=global_step)
    apply_gradient_op = tf.group(apply_gradient_op1, apply_gradient_op2)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
    
    # Add histograms for gradients.
    for grad, var in grads1 + grads2:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)
    
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    
    return train_op

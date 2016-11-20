import tensorflow as tf
import os.path as pt

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('download_url', 'http://www.comp.leeds.ac.uk/mat4saj/',  """LSP dataset url to download""")
tf.app.flags.DEFINE_string('comp_filename', 'lspet_dataset.zip', """Name of downloaded compressed file.""")
tf.app.flags.DEFINE_string('extracted_file', 'joints.mat', """Name of extracted joints file""")

tf.app.flags.DEFINE_string('data_dir', pt.expanduser('~/Desktop/LSP_data/'), """Folder for downloading and extracting data.""")
tf.app.flags.DEFINE_string('orimage_dir', pt.expanduser('~/Desktop/LSP_data/images/'), """Folder for saving resized train files""")
tf.app.flags.DEFINE_string('resized_dir', pt.expanduser('~/Desktop/LSP_data/resized_images/'), """Folder for saving resized train files""")
tf.app.flags.DEFINE_string('train_dir', pt.expanduser('~/Desktop/LSP_data/TrainData'), """Folder for saving train files""")
tf.app.flags.DEFINE_string('input_dir', pt.expanduser('~/Desktop/LSP_data/EvalData/'),  """Input images directory.""")
tf.app.flags.DEFINE_string('output_dir', pt.expanduser('~/Desktop/LSP_data/EvalData/Drawings'),  """Output images directory.""")
tf.app.flags.DEFINE_string('drawing_dir', pt.expanduser('~/Desktop/LSP_data/TrainData/Drawings'), """Folder for saving train files""")

tf.app.flags.DEFINE_string('input_type', 'jpg',  """Input type.""")

tf.app.flags.DEFINE_string('trainLabels_fn', 'train_joints.csv', """Train labels file.""")
tf.app.flags.DEFINE_string('testLabels_fn', 'test_joints.csv', """Test labels file.""")

tf.app.flags.DEFINE_integer('input_size', 100, """One side of CNN's input image size""")
tf.app.flags.DEFINE_integer('input_depth', 3, """Color component size CNN's input image""")
tf.app.flags.DEFINE_integer('label_count', 14, """Label count of images""")
tf.app.flags.DEFINE_integer('label_size', 2, """Label size of images""")
tf.app.flags.DEFINE_integer('batch_size', 128, """Batch size for train""")
tf.app.flags.DEFINE_integer('eval_size', 1, """Batch size for eval""")

tf.app.flags.DEFINE_integer('max_steps', 1000000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('moving_average_decay', 0.9999, """The decay to use for the moving average""")
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 50000, """Epochs after which learning rate decays""")
tf.app.flags.DEFINE_integer('learn_decay_factor', 0.9, """Learning rate decay factor""")
tf.app.flags.DEFINE_integer('initial_learn_rate', 0.002, """Initial learning rate""")
tf.app.flags.DEFINE_integer('example_per_epoch', 128, """Number of Examples per Epoch for Train""")


TotalImageBytes = FLAGS.input_size * FLAGS.input_size * FLAGS.input_depth
TotalLabels = FLAGS.label_count * FLAGS.label_size

TOWER_NAME = '-----------'

class BodyParts:
    Right_ankle = 0
    Right_knee = 1
    Right_hip = 2
    Left_hip = 3
    Left_knee = 4
    Left_ankle = 5
    Right_wrist = 6
    Right_elbow = 7
    Right_shoulder = 8
    Left_shoulder = 9
    Left_elbow = 10
    Left_wrist = 11
    Neck = 12
    Head_top = 13
    

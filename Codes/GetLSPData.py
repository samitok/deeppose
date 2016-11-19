from LSPGlobals import FLAGS
import LSPGlobals
import os
import urllib.request as url_request
import sys
import zipfile
import glob
import numpy as np
from scipy.io import loadmat
from scipy import misc
import tensorflow as tf
# from pdb import set_trace as bp
from print_progress import print_progress


def main():
    filename = maybe_download()

    if not os.path.exists(os.path.join(FLAGS.data_dir, FLAGS.extracted_file)):
        extract_file(filename)
    else:
        print('Already Extracted.')

    parse_resize_image_and_labels()


def maybe_download():
    if not os.path.exists(FLAGS.data_dir):
        os.mkdir(FLAGS.data_dir)
    file_path = os.path.join(FLAGS.data_dir, FLAGS.comp_filename)
    if not os.path.exists(file_path):
        print('Downloading ', file_path, '.')
        file_path, _ = url_request.urlretrieve(FLAGS.download_url + FLAGS.comp_filename, file_path,
                                               reporthook=download_progress)
        stat_info = os.stat(file_path)
        print('Successfully downloaded', stat_info.st_size, 'bytes.')
    else:
        print(file_path, 'already exists.')

    return file_path


def download_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r Downloaded %d%% of %d megabytes" % (percent, total_size / (1024 * 1024)))
    sys.stdout.flush()


def extract_file(filename):
    print('Extracting ', filename, '.')
    opener, mode = zipfile.ZipFile, 'r'
    cwd = os.getcwd()
    os.chdir(os.path.dirname(filename))
    try:
        zip_file = opener(filename, mode)
        try: zip_file.extractall()
        finally: zip_file.close()
    finally:
        os.chdir(cwd)
        print('Done extracting')


def parse_resize_image_and_labels():
    print('Resizing and packing images and labels to bin files.\n')
    np.random.seed(1701)  # to fix test set

    jnt_fn = FLAGS.data_dir + 'joints.mat'

    joints = loadmat(jnt_fn)
    joints = joints['joints'].swapaxes(0, 2).swapaxes(1, 2)
    invisible_joints = joints[:, :, 2] < 0.5
    joints[invisible_joints] = 0
    joints = joints[..., :2]

    image_list = np.asarray(sorted(glob.glob(FLAGS.orimage_dir + '*.jpg')))

    image_indexes = list(range(0, len(image_list)))
    np.random.shuffle(image_indexes)

    train_validation_split = int(len(image_list)*FLAGS.train_set_ratio)
    validation_test_split = int(len(image_list)*(FLAGS.train_set_ratio+FLAGS.validation_set_ratio))

    train_indexes = np.asarray(image_indexes[:train_validation_split])
    validation_indexes = np.asarray(image_indexes[train_validation_split:validation_test_split])
    test_indexes = np.asarray(image_indexes[validation_test_split:])

    write_to_tfrecords(image_list[train_indexes], joints[train_indexes], 'train')
    write_to_tfrecords(image_list[validation_indexes], joints[validation_indexes], 'validation')
    write_to_tfrecords(image_list[test_indexes], joints[test_indexes], 'test')

    print('Done.')


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature_list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def prepare_image(original_image_path):
    image = misc.imread(original_image_path)
    scaled_image = misc.imresize(image, (FLAGS.input_size, FLAGS.input_size), interp='bicubic')
    return scaled_image, image.shape[0], image.shape[1]


def scale_label(label, original_height, original_width):
    label[:, 0] *= (FLAGS.input_size / float(original_width))
    label[:, 1] *= (FLAGS.input_size / float(original_height))
    return label.reshape(LSPGlobals.TotalLabels)


def write_to_tfrecords(image_paths, labels, name):
    num_examples = image_paths.shape[0]

    filename = os.path.join(FLAGS.data_dir, name + '.tfrecords')

    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(num_examples):
        image, or_height, or_width = prepare_image(image_paths[index])  # FIXME read file
        image_raw = image.tostring()

        label = scale_label(labels[index], or_height, or_width)

        features = tf.train.Features(feature={
            # 'height': _int64_feature(FLAGS.input_size),
            # 'width': _int64_feature(FLAGS.input_size),
            # 'depth': _int64_feature(FLAGS.input_depth),
            'label': _int64_feature_list(label.astype(int).tolist()),
            'image_raw': _bytes_feature(image_raw)})

        example = tf.train.Example(features=features)

        writer.write(example.SerializeToString())

        print_progress(index, num_examples-1, prefix='Progress:', suffix='Complete', bar_length=50)

    writer.close()


if __name__ == "__main__":
    main()

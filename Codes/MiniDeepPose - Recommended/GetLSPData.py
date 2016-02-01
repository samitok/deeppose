from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from LSPGlobals import FLAGS
import os
import urllib
import sys
import zipfile
import glob
import numpy as np
from scipy.io import loadmat
from os.path import basename as b
from PIL import Image
import LSPGlobals

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
    filepath = os.path.join(FLAGS.data_dir, FLAGS.comp_filename)
    if not os.path.exists(filepath):
        print('Downloading ', filepath, '.')
        filepath, _ = urllib.urlretrieve(FLAGS.download_url + FLAGS.comp_filename, filepath, reporthook=dlProgress)
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', statinfo.st_size, 'bytes.')
    else:
        print(filepath, 'already exists.')
        
    return filepath

def dlProgress(count, blockSize, totalSize):
    percent = int(count*blockSize*100/totalSize)
    sys.stdout.write("\r Downloaded %d%% of %d megabytes" % (percent, totalSize/(1024*1024)))
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
    np.random.seed(1701) # to fix test set

    if not os.path.exists(FLAGS.resized_dir):
        os.mkdir(FLAGS.resized_dir)

    jnt_fn = FLAGS.data_dir + 'joints.mat'
    
    joints = loadmat(jnt_fn)
    joints = joints['joints'].swapaxes(0, 2).swapaxes(1, 2)
    invisible_joints = joints[:, :, 2] < 0.5
    joints[invisible_joints] = 0
    joints = joints[...,:2]

    N_test = int(len(joints) * 0.1)
    permTest = np.random.permutation(int(len(joints)))[:N_test].tolist()

    imagelist = sorted(glob.glob(FLAGS.orimage_dir + '*.jpg'))

    fp_train = open(os.path.join(FLAGS.data_dir, FLAGS.trainLabels_fn), 'w')
    fp_test = open(os.path.join(FLAGS.data_dir, FLAGS.testLabels_fn), 'w')
    for index, img_fn in enumerate(imagelist):
        imgFile = Image.open(img_fn)
        (imWidth, imHeight) = imgFile.size
        imgFile = imgFile.resize((FLAGS.input_size,FLAGS.input_size), Image.ANTIALIAS)
        newFileName = os.path.join(FLAGS.resized_dir, b(img_fn).replace( "jpg", "bin" ))

        joints[index, :, 0] *= FLAGS.input_size/float(imWidth)
        joints[index, :, 1] *= FLAGS.input_size/float(imHeight)
        
        im_label_pack = np.concatenate((  joints[index, :, :].reshape(LSPGlobals.TotalLabels),
                                          np.asarray(imgFile).reshape(LSPGlobals.TotalImageBytes)  ))
        im_label_pack.astype(np.uint8).tofile(newFileName)

        if index in permTest:
            print(newFileName, file=fp_test)        
        else:
            print(newFileName, file=fp_train)
            
        if (index % 100 == 0):
            sys.stdout.write("\r%d done" % index) #"\r" deletes previous line
            sys.stdout.flush()
        
        #"\r" deletes previous line
        sys.stdout.write("\r")
        sys.stdout.flush()
        
    print('Done.')
  
  

if __name__ == "__main__":
    main()

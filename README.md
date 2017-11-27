*Attention*: This project is in early development stage. So codes and performance are in a very primitive level.
========

DeepPose
========

NOTE: This is not official implementation. Original paper is [DeepPose: Human Pose Estimation via Deep Neural Networks](http://arxiv.org/abs/1312.4659).
SECOND NOTE: This implementation was a project for my Pattern Recognition Course at [METU](http://www.metu.edu.tr/). **Codes are in a very primitive level.** But people might find them useful.

# Requirements
- [TensorFlow](https://github.com/tensorflow/tensorflow) (Google's Neural Network Toolbox)
- Python 3.5.x
- Numpy
- SciPy.io (For loading .mat files)
- PIL or Pillow


# Important
 Edit values in 'LSPGlobals.py' as you want them. All the codes run using values in that file.

# Data preparation

```
python3 GetLSPData.py
```

This script downloads Leeds Sports Pose Dataset (http://sam.johnson.io/research/lsp.html) and performs resizing as your Neural Network input size. Resized images and their labels are saved into binary files.

Dataset:

- [LSP Dataset](http://human-pose.mpi-inf.mpg.de/#download)

# Start training

Just run:

```
python3 TrainLSP.py
```


## To Follow Progress

```
tensorboard --logdir=/path/to/log-directory   #path is '~/Desktop/LSP_data/TrainData' if LSPGlobals.py is unchanged
```


# Evaluating the trained model
```
python3 EvalDeepPose.py
```
This will get all images placed in '--input_dir' with extension '--input_type' will draw stick figures on images based on estimations from the model. Drawn images will be placed in '--output_dir'.


## Video

I recommend you to use ffmpeg to turn videos into images, feed them to network and make video from the drawn images using ffmpeg again.

[Here is my example](https://www.youtube.com/watch?v=Aqa-uWqb5fg)

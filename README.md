# SegNetCMR
A [Tensorflow](https://www.tensorflow.org/) implementation of [SegNet](https://mi.eng.cam.ac.uk/projects/segnet/) to segments CMR images

## Aims
1. A demonstration of a more complete Tensorflow program including saving state and resuming.
2. Provide an ready-to-go example of medical segmentation with sufficient training and validation data, in a usable format (PNGs).

## Requirements
You must have a GPU and install the tensorflow-gpu version as the cpu version does not have tf.nn.max_pool_with_argmax()
1. Python >=3.6: Best to use the [Conda](https://www.continuum.io/downloads) distribution
2. tensorflow-gpu >=0.11

## Todo
1. Add code to run on your own data (currently there is only the training code present)

## Running
Make sure you have conda and tensorflow installed

```commandline
conda install tensorflow-gpu
python
Python 3.6.1 | packaged by conda-forge | (default, Sep  8 2016, 14:36:38)
```
The git clone this repository
```commandline
git clone https://github.com/mshunshin/SegNetCMR.git
```

And start the training from the folder
```commandline
cd /path/to/SegNetCMR
python train.py
```

And in another terminal window start tensorboard
```commandline
tensorboard --logdir ./Output
```
Then in your webbrowser go to [http://localhost:6006](http://localhost:6006)

## Training and test data
Many thanks to the [Sunnybrook Health Sciences Centre](http://www.cardiacatlas.org/studies/sunnybrook-cardiac-data/) for providing a set of CMR data with associated contours.
Unfortunately, in the latest release the filenames have become a little mangled, and don't match up with the contours.
I have gone through the files and matched them up; exported the DICOMS as PNGs and converted the list of coordinates of the contours to PNGs as well.

The first two sets of CMRs are included as training data, the last set as test data.



## With thanks to
[andreaazzini/segnet](https://github.com/andreaazzini/segnet): A Tensorflow SegNet translation

[pydicom](https://github.com/darcymason/pydicom): A pure python dicom library

[StackOverflow Tensorflow batch_norm thread](http://stackoverflow.com/questions/40081697/getting-low-test-accuracy-using-tensorflow-batch-norm-function)

[GitHub Tensorflow unpool thread](https://github.com/tensorflow/tensorflow/issues/2169)

## Issues and annoyances
1. The original SegNet uses max_pool_with_argmax, and requires an unpool_with_argmax. Unfortunately, Tensorflow does not provide an unpool_with_argmax. Fortunately there is code in the github thread above to make your own.
2. This version of unpool_with_argmax runs on the CPU not GPU so is a little slower.
3. Tensorflow does not provide a CPU version of max_pool_with_argmax, so if you don't have a GPU you can't run this.
4. Tensorflow forgot to include a function for gradients for maxpoolwithargmax, so it is included at the bottom of train.py
5. The name mangling of the Sunnybrook CMR data - I have fixed this and the data is included in the download.
6. SegNet works better with a version of softmax that is inversely weighted by class frequency. I can't get the weights to work properly with the tensorflow provided version - my private version has some code to do it by hand - but it would be nice if we could get it to work using the official function.

## License
SegNetCMR: MIT license

SunnyBrook Cardiac Data: Public Domain

pydicom: MIT license

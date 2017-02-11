import os
import sys

import numpy as np

from scipy.misc import imsave
import scipy.ndimage

import pydicom

training_dicom_dir = "./test/a"
training_labels_dir = "./test/b"

training_png_dir = "./Data/Training/Images/Sunnybrook_Part2"
training_png_labels_dir = "./Data/Training/Labels/Sunnybrook_Part2"

for root, dirs, files in os.walk(training_labels_dir):
    for file in files:
        if file.endswith("-icontour-manual.txt"):
            try:
                prefix, _ = os.path.split(root)
                prefix, _ = os.path.split(prefix)
                _, patient = os.path.split(prefix)

                file_fn = file.strip("-icontour-manual.txt") + ".dcm"
                print(file_fn)
                print(patient)
                dcm = pydicom.read_file(os.path.join(training_dicom_dir, patient, file_fn))
                print(dcm.pixel_array.shape)
                img = np.concatenate((dcm.pixel_array[...,None], dcm.pixel_array[...,None], dcm.pixel_array[...,None]), axis=2)
                labels = np.zeros_like(dcm.pixel_array)

                print(img.shape)
                print(labels.shape)

                with open(os.path.join(root, file)) as labels_f:
                    for line in labels_f:
                        x, y = line.split(" ")
                        labels[int(float(y)), int(float(x))] = 128

                labels = scipy.ndimage.binary_fill_holes(labels)

                img_labels = np.concatenate((labels[..., None], labels[..., None], labels[..., None]), axis=2)

                imsave(os.path.join(training_png_dir, patient + "-" + file_fn + ".png"), img)
                imsave(os.path.join(training_png_labels_dir, patient + "-" + file_fn + ".png"), img_labels)
            except Exception as e:
                print(e)


#!/usr/bin/env python3
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

from jetson_inference import imageNet
from jetson_utils import loadImage

import argparse


# parse the command line
parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
args = parser.parse_args()

# load an image (into shared CPU/GPU memory)
img = loadImage(args.filename)

# load the recognition network
net = imageNet(model="resnet18.onnx", labels="labels.txt", input_blob="input_0", output_blob="output_0")

# classify the image
class_idx, confidence = net.Classify(img)

# find the object description
class_desc = net.GetClassDesc(class_idx)

# print out the result
print("Image is recognized as "+ str(class_desc) +" (class #"+ str(class_idx) +") with " + str(confidence*100)+"% confidence")

Dict = {'Agaricus': 'Agaricus is the genus of mushroom that includes button, crimini, and portobello', 'Amanita': 'Amanita mushrooms usually have caps with intricate patterns', 'Boletus': 'Boletus mushrooms are unique in that they have a pore surface instead of gills', 'Cortinarius': 'Cortinarius is the largest genus of mushroom, and they can always be found near trees', 'Entoloma': 'Many Entoloma mushrooms can be toxic', 'Hygrocybe': 'Hygrocibe mushrooms are colorful grassland mushrooms', 'Lactarius': 'Lactarius mushrooms, also known as milk caps, secrete a milky fluid when cut or damaged', 'Russula': 'Russula mushrooms are often fairly large and brightly colored', 'Suillus': 'Some Suillus mushrooms are prized for their buttery flavor and texture'}

print(Dict[class_desc])
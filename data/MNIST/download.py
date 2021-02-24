# source: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py

import urllib.request
import gzip
import os
import struct
import sys

from array import array
from os import path

import png


def download_and_unpack(fname, path):
    tmpfile = os.path.join(path, 'tmp.gz')
    savefile = os.path.join(path, fname)
    # download file
    url = "http://yann.lecun.com/exdb/mnist/" + fname + ".gz"
    urllib.request.urlretrieve(url, tmpfile)
    # unzip
    with gzip.open(tmpfile, 'rb') as f:
        file_content = f.read()
    with open(savefile, 'wb') as f:
        f.write(file_content)

    # delete download
    os.remove(tmpfile)


def read(dataset="train", path="."):
    if dataset is "train":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "test":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'test' or 'train'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array("B", fimg.read())
    fimg.close()

    return lbl, img, size, rows, cols


def write_dataset(labels, data, size, rows, cols, output_dir, mask=None):
    # create output directories for labels 0-9
    output_dirs = [
        path.join(output_dir, str(i))
        for i in range(10)
    ]
    # make directories
    for dir in output_dirs:
        if not path.exists(dir):
            os.makedirs(dir)

    # write data
    for (i, label) in enumerate(labels):
        if mask and not mask[i]:
            continue
        output_filename = path.join(output_dirs[label], str(i) + ".png")
        # i: img index
        # label: number in the image
        print("writing " + output_filename)
        with open(output_filename, "wb") as h:
            w = png.Writer(cols, rows, greyscale=True)
            data_i = [
                data[(i*rows*cols + j*cols): (i*rows*cols + (j+1)*cols)]
                for j in range(rows)
            ]
            w.write(h, data_i)


def main():
    input_path = './data/MNIST/'
    output_path = './data/MNIST/'

    # download
    for fname in ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']:
        download_and_unpack(fname, input_path)

    # save as images
    labels, data, size, rows, cols = read("test", input_path)
    write_dataset(labels, data, size, rows, cols,
                  path.join(output_path, "test"))
    labels, data, size, rows, cols = read("train", input_path)
    write_dataset(labels, data, size, rows, cols,
                  path.join(output_path, "train"), mask=[True]*50000 + [False]*10000)
    write_dataset(labels, data, size, rows, cols,
                  path.join(output_path, "val"), mask=[False]*50000 + [True]*10000)


if __name__ == '__main__':
    main()

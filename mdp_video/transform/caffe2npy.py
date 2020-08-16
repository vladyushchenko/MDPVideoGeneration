import argparse
from typing import Tuple

import numpy as np
import caffe


def convert_numpy_binaryproto(input_path: str, output_path: str) -> None:
    """
    Convert binaryproto to nupe array.

    :param input_path: input
    :param output_path: output
    """
    print("Reading {}".format(input_path))
    avg_img = np.load(input_path)

    # avg_img is your numpy array with the average data
    blob = caffe.proto.caffe_pb2.BlobProto()
    blob.num, blob.height, blob.width, blob.channels = avg_img.shape
    blob.data.extend(avg_img.astype(float).flat)

    with open(output_path, "wb") as binaryproto_file:
        binaryproto_file.write(blob.SerializeToString())


def blob_read(file_path: str) -> np.ndarray:
    """
    Read binaryproto and return flat array.

    :param file_path: file path to read
    :return: np array
    """
    blob = caffe.proto.caffe_pb2.BlobProto()

    with open(file_path, "rb") as f:
        bin_mean = f.read()
        blob.ParseFromString(bin_mean)
        assert blob.diff is not None
        print(blob.num)
        print(blob.channels)
        print(blob.height)
        print(blob.width)
        # print(blob.diff)
        return np.asarray(blob.diff)


def blob_read_shape(file_path: str, shape: Tuple) -> np.ndarray:
    """
    Read binaryproto and return flat array.
    """
    blob = caffe.proto.caffe_pb2.BlobProto()

    with open(file_path, "rb") as f:
        bin_mean = f.read()
        blob.ParseFromString(bin_mean)
        assert blob.diff is not None
        return np.asarray(blob.diff).reshape(shape)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser("Converter binary caffe / numpy array")
    PARSER.add_argument("binary_path")
    PARSER.add_argument("numpy_path")
    ARGS = PARSER.parse_args()
    NPY_MEAN = blob_read(ARGS.binary_path)
    print(NPY_MEAN)
    print("Out shape: {}".format(NPY_MEAN.shape))
    np.save(ARGS.numpy_path, NPY_MEAN)

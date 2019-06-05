import sys
sys.path.insert(0, ".")
import argparse
import os, lmdb, zipfile, tqdm
from io import BytesIO
import caffe
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="", help="path to data dir")
parser.add_argument("--output_path", default="", help="path to output lmdb")
parser.add_argument("--label_npy", default="", help="path to label npy file")
parser.add_argument("--imgsize", default=64, type=int, help="resize image")
args = parser.parse_args()

use_zip = (".zip" in args.data_path)

# read from zip
data_file = zipfile.ZipFile(args.data_path)
files = data_file.namelist()
files.sort()
files = files[1:]

label = np.load(args.label_npy)
class_num = label.shape[-1]

def datum_from_image(img, label):
    # for uint8 image data
    datum = caffe.proto.caffe_pb2.Datum()
    datum.channels = img.shape[2]
    datum.height = img.shape[0]
    datum.width = img.shape[1]
    datum.data = img.tobytes()
    #datum.float_data.extend(Xi.numpy().astype(float).flat)
    datum.label = int(label)
    return datum

# normal celeba
# open lmdb
env = lmdb.open(args.output_path, map_size=len(files) * args.imgsize * args.imgsize * 3 * 8 * 3)
with env.begin(write=True) as txn:
    for idx in tqdm.tqdm(len(files)):
        if use_zip:
            img = np.asarray(Image.open(BytesIO(data_file.read(files[idx]))))
        else:
            img_path = os.path.join(args.data_path, files[idx])
            img = np.asarray(Image.open(open(img_path, "rb")))

        # celeba processing
        img = img[50:50+128, 25:25+128] #uint8 format
        img = np.asarray(Image.fromarray(img).resize((args.imgsize, args.imgsize))) # resize
        datum = datum_from_image(img, label[idx][0])
        str_id = '{:08}'.format(idx)
        txn.put(str_id.encode('ascii'), datum.SerializeToString())


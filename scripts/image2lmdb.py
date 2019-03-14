## import 
import os, lmdb, zipfile
import caffe
import numpy as np
from PIL import Image

data_path = sys.argv[1]
npy_path = sys.argv[2]
lmdb_output = sys.argv[3]
use_zip = True

# read from zip
data_file = zipfile.ZipFile(data_path)
files = data_file.namelist()
files.sort()
files = files[1:]

label = np.load(npy_path)
class_num = label.shape[-1]

# open lmdb
env = lmdb.open(lmdb_output, map_size=map_size)

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
with env.begin(write=True) as txn:
    for idx in range(len(files)):
        if use_zip:
            img = np.asarray(Image.open(BytesIO(data_file.read(files[idx]))))
        else:
            img_path = os.path.join(data_path, files[idx])
            img = np.asarray(Image.open(open(img_path, "rb")))

        # celeba processing
        img = img[50:50+128, 25:25+128] #uint8 format
        datum = datum_from_image(img, label[idx][0])
        str_id = '{:08}'.format(idx)
        txn.put(str_id.encode('ascii'), datum.SerializeToString())


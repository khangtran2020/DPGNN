import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from tqdm import tqdm

SIZE = 224

masked_image_path = '../Datasets/MIR_FLICKR/masked/'
resize_image_path = '../Datasets/MIR_FLICKR/resize/'
image_df = pd.read_csv('../Datasets/MIR_FLICKR/mir.csv')
image_df['image_path'] = image_df['image_path'].map(lambda x: x.split('/')[-1])
img_path = []
for img in image_df['image_path']:
    img_path.append(img)
print(image_df.head())
print(img_path[:5])

base_model = tf.keras.applications.resnet50.ResNet50(
    input_shape=(SIZE, SIZE, 3),
    weights='imagenet',
    include_top=False,
    pooling='avg',
)

none_mask = None
mask = None
indx = 0
for img_file in tqdm(img_path):
    none_path = resize_image_path + img_file
    mask_path = masked_image_path + img_file
    img_none = cv2.imread(none_path, cv2.IMREAD_COLOR)
    img_none = np.expand_dims(img_none, axis=0)
    img_mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    img_mask = np.expand_dims(img_mask, axis=0)
    # print(img_none.shape, img_mask.shape)
    none_feat = base_model.predict(img_none)
    mask_feat = base_model.predict(img_mask)
    if (indx == 0):
        none_mask = none_feat
        mask = mask_feat
    else:
        none_mask = np.concatenate((none_mask, none_feat), axis=0)
        mask = np.concatenate((mask, mask_feat), axis=0)
    indx += 1
np.save("../Datasets/MIR_FLICKR/resnet50_plain_feat.npy", none_mask)
np.save("../Datasets/MIR_FLICKR/resnet50_mask_feat.npy", mask)

print("Done")
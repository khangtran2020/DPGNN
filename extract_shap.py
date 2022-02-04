import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import numpy as np
import shap
import tensorflow as tf
import tensorflow.keras.backend as K
from tqdm import tqdm
import json

SIZE = 224

plain_feat = np.load("../Datasets/MIR_FLICKR/resnet50_plain_feat.npy")

base_model = tf.keras.applications.resnet50.ResNet50(
    input_shape=(SIZE, SIZE, 3),
    weights='imagenet',
    include_top=True,
    pooling='avg',
)
weight = base_model.layers[-1].get_weights()
inputs = tf.keras.Input(shape = (2048,))
outputs = tf.keras.layers.Dense(1000, activation="softmax")(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.set_weights(weight)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
)

def map2layer(x, layer):
    feed_dict = dict(zip([model.layers[0].input], [x.copy()]))
    return K.get_session().run(model.layers[layer].input, feed_dict)

# getting ImageNet 1000 class names
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
with open(shap.datasets.cache(url)) as file:
    class_names = [v[1] for v in json.load(file).values()]
print("Number of ImageNet classes:", len(class_names))

e = shap.GradientExplainer((model.layers[-1].input, model.layers[-1].output),
                           map2layer(plain_feat.copy(), -1))

indx = 0
shap_score = None
for feat in tqdm(plain_feat):
    temp = np.expand_dims(feat, axis=0)
    shap_values, indexes = e.shap_values(map2layer(temp, -1), ranked_outputs=1)
    if indx == 0:
        shap_score = shap_values[0]
    else:
        shap_score = np.concatenate((shap_score, shap_values[0]), axis=0)
    indx += 1
np.save("../Datasets/MIR_FLICKR/resnet_50_shap_score.npy", np.array(shap_score))
print("Process Done")
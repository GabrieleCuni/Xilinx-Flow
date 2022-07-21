import os 
import argparse
import tensorflow as tf

"""
> saved_model_cli show --dir vai_quant_models/mobilenet_v1_0.25_128
> The given SavedModel contains the following tag-sets:
"""

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="mobilenet_v1_0.25_128")
args = parser.parse_args()

# model = tf.saved_model.load(os.path.join("vai_quant_models", args.model), tags="saved_model")

# tf_models/mobilenet_v1_0.25_128
# model = tf.saved_model.load(os.path.join("tf_models", args.model), tags="saved_model")

model = tf.compat.v1.saved_model.load_v2(os.path.join("vai_quant_models", args.model))

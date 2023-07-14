import numpy as np
import tensorflow as tf

import rsqtoa.fapprox.rsqtoa_fapprox_config as config

def _build_dataset(batch_size, *dataset_components):
  """builds train and test-dataset, which should be better to digest for tensorflow! Note that
  * dataset_components = (xs, ys, ys_dx) or (xs, ys). All tuple-elements are np.arrays
  * on the validation-dataset, the model does not internally compute gradients. Thus, test-batches are much bigger!
  """
  tf_data_set = tf.data.Dataset.from_tensor_slices(tuple(dataset_components))
  tf_data_set = tf_data_set.shuffle(buffer_size=config.shuffle_buffer_size)
  tf_data_set = tf_data_set.batch(batch_size)

  return tf_data_set

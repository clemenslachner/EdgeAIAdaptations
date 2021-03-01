import tensorflow as tf
from tensorflow.python.client import device_lib

print("\nDEVICE List ------------------\n %s ------------------\n" % device_lib.list_local_devices())

if(tf.test.is_gpu_available()):
    print("GPU found and ready to use for TF-gpu")
else:
    print("GPU not available for TF-gpu usage. Check device list if gpu is found there!")



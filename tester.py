from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class Tester:
    def __init__(self,model):
        self.model=model

device_name = tf.test.gpu_device_name()
if device_name == '/device:GPU:0':
    print('Found GPU at:{}!'.format(device_name))
else:
    print('Tensorflow have not found GPU,run on CPU mode!')

train_set = [".\\tfrecord\\train.tfrecord"]
val_set = [".\\tfrecord\\val.tfrecord"]

from dataprovider import dataprovider

# create dateprovide for model
dataset_train = dataprovider(train_set,61,1037,[224,224,3],32739,40)
dataset_val = dataprovider(val_set,61,100,[224,224,3],4982,40)



import vgg16
import vgg19

tester = Tester(vgg16.VGG16(dataset_train,dataset_val,0.01,400,os.path.join('.','checkpoints','vgg_16.ckpt')))
#tester = Tester(vgg19.VGG19(dataset,0.01,400,".\\checkpoints\\vgg_19.ckpt",))

tester.model.train_all_epochs()






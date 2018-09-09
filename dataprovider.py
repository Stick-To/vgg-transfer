from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

# dataprovide class that provide data iterator for model

def parse_function(record):
    features= { 'label': tf.FixedLenFeature([], tf.string),
                 'shape': tf.FixedLenFeature([], tf.string),
                'image_raw': tf.FixedLenFeature([], tf.string),
                        }
    example = tf.parse_single_example(record,features);
    label = tf.decode_raw(example['label'], tf.int64)
    shape = tf.decode_raw(example['shape'], tf.int32)
    image = tf.image.decode_jpeg(example["image_raw"],channels=3)
    return label,shape,image

def get_datset_iterator(filename, buffer_size, batch_size=32):
    data_set = tf.data.TFRecordDataset(filenames=filename)
    data_set = data_set.map(parse_function).shuffle(buffer_size = buffer_size).batch(batch_size).repeat()
    iterator = tf.data.Iterator.from_structure(data_set.output_types,data_set.output_shapes)
    init_op = iterator.make_initializer(data_set)
    print("get",filename,"iterator done!")
    return iterator,init_op

class dataprovider:
    def __init__(self, train_set,val_set,num_classes,num_train,num_val,image_shape,buffer_size_train,buffer_size_val,batch_size=32):
        self.train_set = train_set
        self.val_set = val_set
        self.train_iterator,self.train_init_op = get_datset_iterator(train_set, batch_size=batch_size,buffer_size=buffer_size_train)
        self.val_iterator,self.val_init_op = get_datset_iterator(val_set, batch_size=batch_size,buffer_size=buffer_size_val)
        self.batch_size = batch_size
        self.n_classes = num_classes
        self.num_train = num_train
        self.num_val = num_val
        self.data_shape = image_shape


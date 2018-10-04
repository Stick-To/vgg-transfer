from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow as pytf
import numpy as np
import time
from datetime import timedelta
import os

VGG_MEAN = np.array([103.939, 116.779, 123.68]).reshape((1,1,1,3))

class VGG16:
    def __init__(self,dataprovider_train,dataprovider_val,init_learning_rate,epoch,checkpoint,reduce_epoch_time1=0.3,reduce_epoch_time2=0.6):
        """
             checkpoint:ckpt file to be read
             dataset:the folder name to be create in checkpoints/ that save checkpoints for The parameters of this time
             reduce_epoch_time1:divide learning_rate by 10
             reduce_epoch_time2:divide learning_rate by 10
        """
        self.dataprovider_train = dataprovider_train
        self.data_shape = dataprovider_train.data_shape
        self.num_classes = dataprovider_train.num_classes
        self.train_initop = dataprovider_train.initop
        self.train_iterator = dataprovider_train.iterator
        self.num_train = dataprovider_train.num_samples
        self.train_batch_size = dataprovider_train.batch_size
        self.train_labels, _, self.train_images = self.train_iterator.get_next()
        self.train_images = tf.cast(self.train_images, tf.float32) - VGG_MEAN

        self.is_test = True if dataprovider_val is not None else False
        if self.is_test:
            self.dataprovider_val = dataprovider_val
            self.val_initop = dataprovider_val.initop
            self.val_iterator = dataprovider_val.iterator
            self.num_val = dataprovider_val.num_samples
            self.val_batch_size = dataprovider_val.batch_size
            self.val_labels, _, self.val_images = self.val_iterator.get_next()
            self.val_images = tf.cast(self.val_images, tf.float32) - VGG_MEAN

        self.learning_late = init_learning_rate
        self.checkpoint = checkpoint

        self.global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)

        self.epoch = int(epoch)
        self.epoch_reduce_lr1 = int(epoch*reduce_epoch_time1)
        self.epoch_reduce_lr2 = int(epoch*reduce_epoch_time2)
        self._build_input()
        self._build_graph()
        self._create_optimizer()
        #self._create_summary()
        self._create_checkpoints()
        self._init_session()
    def conv_layer(self,bottom,filter,bias,name):
        conv = tf.nn.conv2d(bottom,filter=filter,strides=[1,1,1,1],name="kernel"+name,padding="SAME")
        conv_bias = tf.nn.bias_add(conv,bias=bias ,name="bias"+name)
        return tf.nn.relu(conv_bias)
    def max_pool(self,bottom,name):
        return tf.nn.max_pool(bottom,[1,2,2,1],[1,2,2,1],padding='SAME',name=name)
    def fc_layer(self, bottom, name, units,activation=None):
        return tf.layers.dense(bottom, units,activation=activation,
                    name=name)
    def _build_input(self):
        shape = [None]
        shape.extend(self.data_shape)
        self.images = tf.placeholder(tf.float32, shape, name='images')
        self.labels = tf.placeholder(tf.int32, [None,self.num_classes], name='labels')
        #self.is_training = tf.placeholder(tf.bool, [])

    #read parameters from ckpt
    def _build_graph(self):
        reader = pytf.NewCheckpointReader(self.checkpoint)
        with tf.variable_scope("conv"):

            self.conv1_1=self.conv_layer(self.images,
                                         reader.get_tensor("vgg_16/conv1/conv1_1/weights"),
                                         reader.get_tensor("vgg_16/conv1/conv1_1/biases"),
                                         name = "conv1_1")
            self.conv1_2=self.conv_layer(self.conv1_1,
                                         reader.get_tensor("vgg_16/conv1/conv1_2/weights"),
                                         reader.get_tensor("vgg_16/conv1/conv1_2/biases"),
                                         name = "conv1_2")
            self.pool1=self.max_pool(self.conv1_2,name="pool1")

            self.conv2_1=self.conv_layer(self.pool1,
                                         tf.get_variable(name='kenrel_conv2_1',
                                                         initializer=reader.get_tensor("vgg_16/conv2/conv2_1/weights"),
                                                         trainable=True),
                                         tf.get_variable(name='bias_conv2_1',
                                                         initializer=reader.get_tensor("vgg_16/conv2/conv2_1/biases"),
                                                         trainable=True),
                                         name = "conv2_1")
            self.conv2_2=self.conv_layer(self.conv2_1,
                                         tf.get_variable(name='kernel_conv2_2',
                                                         initializer=reader.get_tensor("vgg_16/conv2/conv2_2/weights"),
                                                         trainable=True),
                                         tf.get_variable(name='bias_conv2_2',
                                                         initializer=reader.get_tensor("vgg_16/conv2/conv2_2/biases"),
                                                         trainable=True),
                                         name="conv2_2")
            self.pool2=self.max_pool(self.conv2_2,name="pool2")

            self.conv3_1=self.conv_layer(self.pool2,
                                         tf.get_variable(name='kernel_conv3_1',
                                                         initializer=reader.get_tensor("vgg_16/conv3/conv3_1/weights"),
                                                         trainable=True),
                                         tf.get_variable(name='bias_conv_3_1',
                                                         initializer=reader.get_tensor("vgg_16/conv3/conv3_1/biases"),
                                                         trainable=True),
                                         name = "conv3_1")
            self.conv3_2=self.conv_layer(self.conv3_1,
                                         tf.get_variable(name='kernel_conv3_2',
                                                         initializer=reader.get_tensor("vgg_16/conv3/conv3_2/weights"),
                                                         trainable=True),
                                         tf.get_variable(name='bias_conv3_2',
                                                         initializer=reader.get_tensor("vgg_16/conv3/conv3_2/biases"),
                                                         trainable=True),
                                         name = "conv3_2")
            self.conv3_3=self.conv_layer(self.conv3_2,
                                         tf.get_variable(name='kernel_conv3_3',
                                                         initializer=reader.get_tensor("vgg_16/conv3/conv3_3/weights"),
                                                         trainable=True),
                                         tf.get_variable(name='bias_conv3_3',
                                                         initializer=reader.get_tensor("vgg_16/conv3/conv3_3/biases"),
                                                         trainable=True),
                                         name = "conv3_3")
            self.pool3=self.max_pool(self.conv3_3,name="pool3")

            self.conv4_1=self.conv_layer(self.pool3,
                                         tf.get_variable(name='kernel_conv4_1',
                                                         initializer=reader.get_tensor("vgg_16/conv4/conv4_1/weights"),
                                                         trainable=True),
                                         tf.get_variable(name='bias_conv4_1',
                                                         initializer=reader.get_tensor("vgg_16/conv4/conv4_1/biases"),
                                                         trainable=True),
                                         name = "conv4_1")
            self.conv4_2=self.conv_layer(self.conv4_1,
                                         tf.get_variable(name='kernel_conv4_2',
                                                         initializer=reader.get_tensor("vgg_16/conv4/conv4_2/weights"),
                                                         trainable=True),
                                         tf.get_variable(name='bias_conv4_2',
                                                         initializer=reader.get_tensor("vgg_16/conv4/conv4_2/biases"),
                                                         trainable=True),
                                         name = "conv4_2")
            self.conv4_3=self.conv_layer(self.conv4_2,
                                         tf.get_variable(name='kernel_conv4_3',
                                                         initializer=reader.get_tensor("vgg_16/conv4/conv4_3/weights"),
                                                         trainable=True),
                                         tf.get_variable(name='bias_conv4_3',
                                                         initializer=reader.get_tensor("vgg_16/conv4/conv4_3/biases"),
                                                         trainable=True),
                                         name = "conv4_3")
            self.pool4=self.max_pool(self.conv4_3,name="pool4")

            self.conv5_1=self.conv_layer(self.pool4,
                                         tf.get_variable(name='kernel_conv5_1',
                                                         initializer=reader.get_tensor("vgg_16/conv5/conv5_1/weights"),
                                                         trainable=True),
                                         tf.get_variable(name='bias_conv5_1',
                                                         initializer=reader.get_tensor("vgg_16/conv5/conv5_1/biases"),
                                                         trainable=True),
                                         name = "conv5_1")
            self.conv5_2=self.conv_layer(self.conv5_1,
                                         tf.get_variable(name='kernel_conv5_2',
                                                         initializer=reader.get_tensor("vgg_16/conv5/conv5_2/weights"),
                                                         trainable=True),
                                         tf.get_variable(name='bias_conv5_2',
                                                         initializer=reader.get_tensor("vgg_16/conv5/conv5_2/biases"),
                                                         trainable=True),
                                         name = "conv5_2")
            self.conv5_3=self.conv_layer(self.conv5_2,
                                         tf.get_variable(name='kernel_conv5_3',
                                                         initializer=reader.get_tensor("vgg_16/conv5/conv5_3/weights"),
                                                         trainable=True),
                                         tf.get_variable(name='bias_conv5_3',
                                                         initializer=reader.get_tensor("vgg_16/conv5/conv5_3/biases"),
                                                         trainable=True),
                                         name = "conv5_3")
            self.pool5=self.max_pool(self.conv5_3,name="pool5")
        with tf.variable_scope("fc"):
            self.flatten = tf.layers.flatten(self.pool5, name="flatten")
            self.fc6 = self.fc_layer(self.flatten,"fc6",4096, activation=tf.nn.relu)
            self.fc7 = self.fc_layer(self.flatten,"fc7",4096, activation=tf.nn.relu)
            self.fc8 = self.fc_layer(self.flatten,"fc8",self.num_classes)
        with tf.variable_scope("loss"):
            self.loss = tf.losses.softmax_cross_entropy(self.labels,self.fc8, reduction=tf.losses.Reduction.MEAN)
            self.pred = tf.argmax(self.fc8,axis=1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred,tf.argmax(self.labels,axis=1)),tf.float32))

    def _create_optimizer(self):
        with tf.variable_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_late).minimize(
                self.loss,global_step=self.global_step)

    def _create_summary(self):
        try:
            os.makedirs(os.path.join('.','graph',str(self.learning_late)))
        except:
            pass
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            self.summary_op = tf.summary.merge_all()


    def _create_checkpoints(self):
        save_path = os.path.join('.', 'checkpoints')
        try:
            os.makedirs(save_path)
        except:
            pass
        self.save_path  = os.path.join(save_path, 'check')
    def _init_session(self):
        self.sess=tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
    def load_model(self):
        path = self.save_path
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(path))
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Restore model at step ",self.global_step.eval(session=self.sess))
        else:
            print("train the model from the scratch")


    def save_model(self):
            self.saver.save(self.sess,self.save_path,global_step=self.global_step)

    def train_one_epoch(self):
        total_loss = []
        total_accuracy = []
        self.sess.run(self.train_initop)
        total_iter = self.num_train//self.train_batch_size
        for i in range(total_iter):
            images,labels = self.sess.run([self.train_images,self.train_labels])
            loss, acc, _ = self.sess.run([self.loss,self.accuracy, self.optimizer],feed_dict={
                self.images:images,
                self.labels:labels
            })
            total_loss.append(loss)
            total_accuracy.append(acc)
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        return mean_loss, mean_accuracy
    def validate(self):

        total_loss = []
        total_accuracy = []
        self.sess.run(self.val_initop)
        total_iter = self.num_val//self.val_batch_size
        for i in range(total_iter):
            images,labels = self.sess.run([self.val_images,self.val_labels])
            loss, acc= self.sess.run([self.loss,self.accuracy],feed_dict={
                self.images:images,
                self.labels:labels
            })
            total_loss.append(loss)
            total_accuracy.append(acc)
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        return mean_loss, mean_accuracy

    def test(self, image):
        img = image - VGG_MEAN
        pred= self.sess.run([self.pred],feed_dict={
            self.images:img,
            })
        return pred
    def train_all_epochs(self):

        self.load_model()
        total_start_time = time.time()
        for epo in range(1, self.epoch+1):
            print("\n","-"*25," Train epoch %d"%epo,"-"*25)
            start_time = time.time()
            if epo == self.epoch_reduce_lr1 or epo == self.epoch_reduce_lr2:
                self.learning_late /= 10
                print("devide the learning rate by 10 at epoch %d" %epo)
            loss, acc = self.train_one_epoch()
            print("train epoch %d" %epo,"mean loss %f" %loss,"mean accuracy %f" %acc)
            loss, acc = self.validate()
            self.save_model()
            print("val epoch %d" %epo,"mean loss %f" %loss,"mean accuracy %f" %acc)
            time_per_epoch = time.time() - start_time
            time_total_used = time.time() - total_start_time

            print("epoch %d"%epo,"used time %s "%str(timedelta(seconds=time_per_epoch)))
            print("total %d epoch now "%epo,"used time %s "%str(timedelta(seconds=time_total_used)))







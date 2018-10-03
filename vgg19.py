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
class VGG19:
    def __init__(self, data_provider, init_learning_rate, epoch, checkpoint, reduce_lr_time_1=0.3, reduce_lr_time_2=0.8,train_acc=0.0,val_acc=0.0):
        self.data_provider = data_provider

        self.data_shape = data_provider.data_shape
        self.num_classes = data_provider.n_classes
        self.train_initop = data_provider.train_init_op
        self.val_initop = data_provider.val_init_op
        self.train_iterator = data_provider.train_iterator
        self.val_iterator = data_provider.val_iterator
        self.num_train = data_provider.num_train
        self.num_val = data_provider.num_val
        self.batch_size = data_provider.batch_size
        self.train_labels,_ ,self.train_images = self.train_iterator.get_next()
        self.train_images =tf.cast(self.train_images,tf.float32) - VGG_MEAN
        self.val_labels,_,self.val_images = self.val_iterator.get_next()
        self.val_images =tf.cast(self.val_images,tf.float32) - VGG_MEAN

        self.learning_rate = init_learning_rate
        self.epoch = epoch
        self.reader = pytf.NewCheckpointReader(checkpoint)
        self.reduce_lr_time_1 = int(self.epoch * reduce_lr_time_1)
        self.reduce_lr_time_2 = int(self.epoch * reduce_lr_time_2)
        self.global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)
        self.train_acc = train_acc
        self.val_acc = val_acc
        self._define_input()
        self._build_graph()
        self._create_optimizer()
        self._create_summary()
        self._create_checkpoints()
        self._init_session()
    def conv_layer(self, bottom, filter, bias, name):
            conv = tf.nn.conv2d(bottom, filter=filter,strides=[1,2,2,1],padding='SAME', name='kernel'+name)
            conv_bias = tf.nn.bias_add(conv,bias=bias,name='bias'+name)
            return conv_bias
    def max_pool(self,bottom,name):
        return tf.nn.max_pool(bottom,[1,2,2,1],[1,2,2,1], padding='SAME', name=name)
    def fc_layer(self,bottom, units, name, activation=None):
        return tf.layers.dense(bottom, units, activation=activation,
                               name = name)
    def _define_input(self):
        shape=[None]
        shape.extend(self.data_shape)
        self.images = tf.placeholder(name='images', shape=shape, dtype=tf.float32)
        self.labels = tf.placeholder(name='labels', shape=[None, self.num_classes], dtype=tf.int32)
    def _build_graph(self):
        with tf.variable_scope('conv'):
            self.conv1_1 = self.conv_layer(self.images,
                        self.reader.get_tensor('vgg_19/conv1/conv1_1/weights'),
                        self.reader.get_tensor('vgg_19/conv1/conv1_1/biases'),
                        name = 'conv1_1')
            self.conv1_2 = self.conv_layer(self.conv1_1,
                        self.reader.get_tensor('vgg_19/conv1/conv1_2/weights'),
                        self.reader.get_tensor('vgg_19/conv1/conv1_2/biases'),
                        name = 'conv1_2')
            self.pool1 = self.max_pool(self.conv1_2,'pool1')

            self.conv2_1 = self.conv_layer(self.pool1,
                        self.reader.get_tensor('vgg_19/conv2/conv2_1/weights'),
                        self.reader.get_tensor('vgg_19/conv2/conv2_1/biases'),
                        name = 'conv2_1')
            self.conv2_2 = self.conv_layer(self.conv2_1,
                        self.reader.get_tensor('vgg_19/conv2/conv2_2/weights'),
                        self.reader.get_tensor('vgg_19/conv2/conv2_2/biases'),
                        name = 'conv2_2')
            self.pool2 = self.max_pool(self.conv2_2,'pool2')

            self.conv3_1 = self.conv_layer(self.pool2,
                        self.reader.get_tensor('vgg_19/conv3/conv3_1/weights'),
                        self.reader.get_tensor('vgg_19/conv3/conv3_1/biases'),
                        name = 'conv3_1')
            self.conv3_2 = self.conv_layer(self.conv3_1,
                        self.reader.get_tensor('vgg_19/conv3/conv3_2/weights'),
                        self.reader.get_tensor('vgg_19/conv3/conv3_2/biases'),
                        name = 'conv3_2')
            self.conv3_3 = self.conv_layer(self.conv3_2,
                        self.reader.get_tensor('vgg_19/conv3/conv3_3/weights'),
                        self.reader.get_tensor('vgg_19/conv3/conv3_3/biases'),
                        name = 'conv3_3')
            self.conv3_4 = self.conv_layer(self.conv3_3,
                        self.reader.get_tensor('vgg_19/conv3/conv3_4/weights'),
                        self.reader.get_tensor('vgg_19/conv3/conv3_4/biases'),
                        name = 'conv3_4')
            self.pool3 = self.max_pool(self.conv3_4,'pool3')

            self.conv4_1 = self.conv_layer(self.pool3,
                        self.reader.get_tensor('vgg_19/conv4/conv4_1/weights'),
                        self.reader.get_tensor('vgg_19/conv4/conv4_1/biases'),
                        name = 'conv4_1')
            self.conv4_2 = self.conv_layer(self.conv4_1,
                        self.reader.get_tensor('vgg_19/conv4/conv4_2/weights'),
                        self.reader.get_tensor('vgg_19/conv4/conv4_2/biases'),
                        name = 'conv4_2')
            self.conv4_3 = self.conv_layer(self.conv4_2,
                        self.reader.get_tensor('vgg_19/conv4/conv4_3/weights'),
                        self.reader.get_tensor('vgg_19/conv4/conv4_3/biases'),
                        name = 'conv4_3')
            self.conv4_4 = self.conv_layer(self.conv4_3,
                        self.reader.get_tensor('vgg_19/conv4/conv4_4/weights'),
                        self.reader.get_tensor('vgg_19/conv4/conv4_4/biases'),
                        name = 'conv4_4')
            self.pool4 = self.max_pool(self.conv4_4,'pool4')

            self.conv5_1 = self.conv_layer(self.pool4,
                        self.reader.get_tensor('vgg_19/conv5/conv5_1/weights'),
                        self.reader.get_tensor('vgg_19/conv5/conv5_1/biases'),
                        name = 'conv5_1')
            self.conv5_2 = self.conv_layer(self.conv5_1,
                        self.reader.get_tensor('vgg_19/conv5/conv5_2/weights'),
                        self.reader.get_tensor('vgg_19/conv5/conv5_2/biases'),
                        name = 'conv5_2')
            self.conv5_3 = self.conv_layer(self.conv5_2,
                        self.reader.get_tensor('vgg_19/conv5/conv5_3/weights'),
                        self.reader.get_tensor('vgg_19/conv5/conv5_3/biases'),
                        name = 'conv5_3')
            self.conv5_4 = self.conv_layer(self.conv5_3,
                        self.reader.get_tensor('vgg_19/conv5/conv5_4/weights'),
                        self.reader.get_tensor('vgg_19/conv5/conv5_4/biases'),
                        name = 'conv5_4')
            self.pool5 = self.max_pool(self.conv5_4,'pool5')
        with tf.variable_scope('fc_layer'):
            self.flatten = tf.layers.flatten(self.pool5)
            self.fc6 = self.fc_layer(self.flatten, 4096,'fc6', tf.nn.relu)
            self.fc7 = self.fc_layer(self.fc6, 1024, 'fc7', tf.nn.relu)
            self.fc8 = self.fc_layer(self.fc7,self.num_classes, 'fc8')
        with tf.variable_scope('loss'):
            self.loss = tf.losses.softmax_cross_entropy(self.labels, self.fc8, reduction=tf.losses.Reduction.MEAN)
            self.pred = tf.argmax(self.fc8, axis=1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred,tf.argmax(self.labels,axis=1)), dtype=tf.float32))
    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                        global_step=self.global_step)
    def _create_summary(self):
        try:
            os.makedirs(os.path.join('.','graph'))
        except:
            pass
        with tf.variable_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            self.summary_op = tf.summary.merge_all()
    def _create_checkpoints(self):
        self.train_save_path = os.path.join('.','checkpoints','train','train')
        self.val_save_path = os.path.join('.','checkpoints','val','val')
        try:
            os.makedirs(self.train_save_path)
        except:
            pass
        try:
            os.makedirs(self.val_save_path)
        except:
            pass
    def _init_session(self):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(os.path.join('.','graph'), self.sess.graph)
        self.val_saver = tf.train.Saver()
        self.train_saver = tf.train.Saver()
    def load_model(self,mode='val'):
        if(mode=='val'):
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.val_save_path))
            if ckpt and ckpt.model_checkpoint_path:
                self.val_saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("Successfually restore val model from iter ",self.global_step.eval())
            else:
                print("Train model from scratch")
        else:
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.train_save_path))
            if ckpt and ckpt.model_checkpoint_path:
                self.train_saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("Successfually restore train model from iter ",self.global_step.eval())
            else:
                print("Train model from scratch")
    def save_model(self,mode):
        if(mode =='val'):
            self.val_saver.save(self.sess, self.val_save_path, global_step=self.global_step)
        else:
            self.train_saver.save(self.sess,self.train_save_path,global_step=self.global_step)
    def train_one_epoch(self):
        total_loss = []
        total_acc = []
        self.sess.run(self.train_initop)
        for i in range(self.num_train//self.batch_size):
            images, labels = self.sess.run([self.train_images, self.train_labels])
            loss, acc, _, summary = self.sess.run([self.loss,self.accuracy,self.optimizer,self.summary_op],
                                feed_dict={
                                    self.images:images,
                                    self.labels:labels
                                })
            self.writer.add_summary(summary,global_step=self.global_step.eval())
            total_loss.append(loss)
            total_acc.append(acc)
        mean_loss = np.mean(total_loss)
        mean_acc = np.mean(total_acc)
        return mean_loss, mean_acc
    def test(self):
        total_loss = []
        total_acc = []
        self.sess.run(self.val_initop)
        for i in range(self.num_val//self.batch_size):
            images, labels = self.sess.run([self.val_images, self.val_labels])
            loss, acc,pred = self.sess.run([self.loss,self.accuracy,self.pred],
                                feed_dict={
                                    self.images:images,
                                    self.labels:labels
                                })
            total_loss.append(loss)
            total_acc.append(acc)
        print(pred)
        mean_loss = np.mean(total_loss)
        mean_acc = np.mean(total_acc)
        return mean_loss, mean_acc
    def train_all_epochs(self):
        self.load_model()
        total_start_time = time.time()
        for i in range(1, self.epoch+1):
            if(i==self.reduce_lr_time_1 or i ==self.reduce_lr_time_1):
                self.learning_rate/=10
            print('-'*25,'epoch %d'%i,'-'*25)
            print("Training...")
            start_time = time.time()
            loss, acc = self.train_one_epoch()
            print('train epoch %d'%i,'mean loss:',loss,' mean accuracy:',acc)
            self.train_acc = acc
            self.save_model('train')
            print('Val...')
            loss, acc = self.test()
            if(acc >= self.val_acc):
                self.val_acc = acc
                self.save_model('val')
                f = open("vallog.txt",'a')
                f.write("epoch: "+str(i)+" loss:"+str(loss)+" acc: "+str(acc)+"\n")
                f.close()
            print('val epoch %d'%i,'mean loss:',loss,' mean accuracy:',acc)
            total_per_epoch = time.time() - start_time
            total_used_time = time.time() - total_start_time
            print("epoch %d"%i,"used time %s "%str(timedelta(seconds=total_per_epoch)))
            print("total %d epoch now "%i,"used time %s "%str(timedelta(seconds=total_used_time)))
        self.writer.close()


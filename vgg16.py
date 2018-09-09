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
    def __init__(self,data_provider,init_learning_rate,epoch,checkpoint,dataset,reduce_epoch_time1=0.3,reduce_epoch_time2=0.6):
        """
             checkpoint:ckpt file to be read
             dataset:the folder name to be create in checkpoints/ that save checkpoints for The parameters of this time
             reduce_epoch_time1:divide learning_rate by 10
             reduce_epoch_time2:divide learning_rate by 10
        """
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

        self.learning_late = init_learning_rate
        self.reader = pytf.NewCheckpointReader(checkpoint)

        self.dataset = dataset
        self.global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)

        self.epoch = int(epoch)
        self.epoch_reduce_lr1 = int(epoch*reduce_epoch_time1)
        self.epoch_reduce_lr2 = int(epoch*reduce_epoch_time2)
        self._build_input()
        self._build_graph()
        self._create_optimizer()
        self._create_summary()
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
        self.images = tf.placeholder(tf.float32, [None, self.data_shape[0], self.data_shape[1] , self.data_shape[2]])
        self.labels = tf.placeholder(tf.int32, [None,self.num_classes])
        #self.is_training = tf.placeholder(tf.bool, [])

    #read parameters from ckpt
    def _build_graph(self):
	    # the layers defined by vgg_16.ckpt
        with tf.variable_scope("conv"):

            self.conv1_1=self.conv_layer(self.images,self.reader.get_tensor("vgg_16/conv1/conv1_1/weights"),
                        self.reader.get_tensor("vgg_16/conv1/conv1_1/biases"),
                        name = "conv1_1")
            self.conv1_2=self.conv_layer(self.conv1_1,self.reader.get_tensor("vgg_16/conv1/conv1_2/weights"),
                        self.reader.get_tensor("vgg_16/conv1/conv1_2/biases"),
                        name = "conv1_2")
            self.pool1=self.max_pool(self.conv1_2,name="pool1")

            self.conv2_1=self.conv_layer(self.pool1,self.reader.get_tensor("vgg_16/conv2/conv2_1/weights"),
                        self.reader.get_tensor("vgg_16/conv2/conv2_1/biases"),
                        name = "conv2_1")
            self.conv2_2=self.conv_layer(self.conv2_1,self.reader.get_tensor("vgg_16/conv2/conv2_2/weights"),
                        self.reader.get_tensor("vgg_16/conv2/conv2_2/biases"),
                        name="conv2_2")
            self.pool2=self.max_pool(self.conv2_2,name="pool2")

            self.conv3_1=self.conv_layer(self.pool2,self.reader.get_tensor("vgg_16/conv3/conv3_1/weights"),
                        self.reader.get_tensor("vgg_16/conv3/conv3_1/biases"),
                        name = "conv3_1")
            self.conv3_2=self.conv_layer(self.conv3_1,self.reader.get_tensor("vgg_16/conv3/conv3_2/weights"),
                        self.reader.get_tensor("vgg_16/conv3/conv3_2/biases"),
                        name = "conv3_2")
            self.conv3_3=self.conv_layer(self.conv3_2,self.reader.get_tensor("vgg_16/conv3/conv3_3/weights"),
                        self.reader.get_tensor("vgg_16/conv3/conv3_3/biases"),
                        name = "conv3_3")
            self.pool3=self.max_pool(self.conv3_3,name="pool3")

            self.conv4_1=self.conv_layer(self.pool3,self.reader.get_tensor("vgg_16/conv4/conv4_1/weights"),
                        self.reader.get_tensor("vgg_16/conv4/conv4_1/biases"),
                        name = "conv4_1")
            self.conv4_2=self.conv_layer(self.conv4_1,self.reader.get_tensor("vgg_16/conv4/conv4_2/weights"),
                        self.reader.get_tensor("vgg_16/conv4/conv4_2/biases"),
                        name = "conv4_2")
            self.conv4_3=self.conv_layer(self.conv4_2,self.reader.get_tensor("vgg_16/conv4/conv4_3/weights"),
                        self.reader.get_tensor("vgg_16/conv4/conv4_3/biases"),
                        name = "conv4_3")
            self.pool4=self.max_pool(self.conv4_3,name="pool4")

            self.conv5_1=self.conv_layer(self.pool4,self.reader.get_tensor("vgg_16/conv5/conv5_1/weights"),
                        self.reader.get_tensor("vgg_16/conv5/conv5_1/biases"),
                        name = "conv5_1")
            self.conv5_2=self.conv_layer(self.conv5_1,self.reader.get_tensor("vgg_16/conv5/conv5_2/weights"),
                        self.reader.get_tensor("vgg_16/conv5/conv5_2/biases"),
                        name = "conv5_2")
            self.conv5_3=self.conv_layer(self.conv5_2,self.reader.get_tensor("vgg_16/conv5/conv5_3/weights"),
                        self.reader.get_tensor("vgg_16/conv5/conv5_3/biases"),
                        name = "conv5_3")
            self.pool5=self.max_pool(self.conv5_3,name="pool5")
# the layers defined by myself
        with tf.variable_scope("fc"):
            self.flatten = tf.layers.flatten(self.pool5, name="flatten")
            self.fc6 = self.fc_layer(self.flatten,"fc6",4096, activation=tf.nn.relu)
            self.fc7 = self.fc_layer(self.flatten,"fc7",4096, activation=tf.nn.relu)
            self.fc8 = self.fc_layer(self.flatten,"fc8",self.num_classes)
# the loss defined by myself
        with tf.variable_scope("loss"):
            self.loss = tf.losses.softmax_cross_entropy(self.labels,self.fc8, reduction=tf.losses.Reduction.MEAN)
            self.pred = tf.argmax(self.fc8,axis=0)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred,tf.argmax(self.labels,axis=0)),tf.float32))

    def _create_optimizer(self):
        with tf.variable_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_late).minimize(
                self.loss,global_step=self.global_step)

    def _create_summary(self):
        try:
            os.makedirs("graphs\\"+self.dataset+"\\lr"+str(self.learning_late)+"\\")
        except:
            pass
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram accuracy', self.accuracy)
            self.summary_op = tf.summary.merge_all()


    def _create_checkpoints(self):
        try:
            os.makedirs("checkpoints\\"+self.dataset+"\\train\\")
        except:
            pass
        try:
            os.makedirs("checkpoints\\"+self.dataset+"\\val\\")
        except:
            pass
        self.save_train_path = "checkpoints\\"+self.dataset+"\\train\\train"
        self.save_val_path = "checkpoints\\"+self.dataset+"\\val\\val"
    def _init_session(self):
        self.sess=tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter("graphs\\"+self.dataset+"\\lr"+str(self.learning_late)+"\\", self.sess.graph)
        self.train_saver = tf.train.Saver()
        self.val_saver = tf.train.Saver()
    def load_model(self, mode="val"):
        if(mode == "val"):
            path = self.save_val_path
        else:
            path = self.save_train_path
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(path))
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Restore model at step ",self.global_step.eval(session=self.sess))
        else:
            print("train the model from the scratch")


    def save_model(self,mode="val"):
        if(mode=="val"):
            self.saver.save(self.sess,self.save_val_path,global_step=self.global_step)
        else:
            self.saver.save(self.sess,self.save_train_path,global_step=self.global_step)

    def train_one_epoch(self):
        total_loss = []
        total_accuracy = []
        self.sess.run(self.train_initop)
        total_iter = self.num_train//self.batch_size
        for i in range(total_iter):
            images,labels = self.sess.run([self.train_images,self.train_labels])
            loss, acc, _, summary = self.sess.run([self.loss,self.accuracy, self.optimizer,self.summary_op],feed_dict={
                self.images:images,
                self.labels:labels
            })
            self.writer.add_summary(summary, global_step=self.global_step.eval(session=self.sess))
            total_loss.append(loss)
            total_accuracy.append(acc)
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        return mean_loss, mean_accuracy
    def test(self):
        total_loss = []
        total_accuracy = []
        self.sess.run(self.val_initop)
        total_iter = self.num_val//self.batch_size
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
            self.save_model("train")
            print("train epoch %d" %epo,"mean loss %f" %loss,"mean accuracy %f" %acc)
            loss, acc = self.test(epo)
            self.save_model("val")
            print("val epoch %d" %epo,"mean loss %f" %loss,"mean accuracy %f" %acc)
            time_per_epoch = time.time() - start_time
            time_total_used = time.time() - total_start_time

            print("epoch %d"%epo,"used time %s "%str(timedelta(seconds=time_per_epoch)))
            print("total %d epoch now "%epo,"used time %s "%str(timedelta(seconds=time_total_used)))
        self.writer.close()







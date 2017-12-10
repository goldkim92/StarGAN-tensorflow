import os
import tensorflow as tf
from collections import namedtuple

from module import generator, discriminator, sce_loss, recon_loss

class stargan(object):
    def __init__(self,sess,args):
        __log_dir = os.path.join('.','assets','log')
        __ckpt_dir = os.path.join('.','assets','checkpoint')
        __batch_size = 1
        __image_size = 128
        __image_channel = 3
        __n_labels = 6
        __lambda_cls = 1
        __lambda_rec = 10
        __lr = 0.0001
        __beta1 = 0.5
        __continue_train = False
        
        self.sess = sess
        self.log_dir = __log_dir
        self.ckpt_dir = __ckpt_dir
        self.batch_size = __batch_size
        self.image_size = __image_size
        self.image_channel = __image_channel
        self.n_labels = __n_labels
        self.lambda_cls = __lambda_cls
        self.lambda_rec = __lambda_rec
        self.lr = __lr
        self.beta1 = __beta1
        self.continue_train = __continue_train
        
        # hyper-parameter for building the module
        OPTIONS = namedtuple('OPTIONS', ['batch_size', 'image_size'])
        self.options = OPTIONS(self.batch_size, self.image_size)
        
        self.build_model()
        self.saver = tf.train.Saver()
        
    def build_model(self):
        # placeholder
        # input_image: A, target_image: B
        self.real_A = tf.placeholder(tf.float32, 
                                           [None, self.image_size, self.image_size, self.image_channel + self.n_labels],
                                           name = 'input_images')
        self.real_B = tf.placeholder(tf.float32, 
                                           [None, self.image_size, self.image_size, self.image_channel + self.n_labels],
                                           name = 'target_images')
        
        # generate image
        self.fake_B = generator(self.real_A, self.options, False, name='gen')
        self.fake_A = generator(self.fake_B, self.options, True, name='gen')
        
        # discriminate image
        # src: real or fake, cls: domain classification 
        self.src_real_B, self.cls_real_B = discriminator(self.real_B, self.options, False, name='disc')
        self.src_fake_B, self.cls_fake_B = discriminator(self.fake_B, self.options, True, name='disc')
        
        # loss
        ## discriminator loss ##
        ### adversarial loss
        self.d_real_adv_loss = sce_loss(self.src_real_B, tf.ones_like(self.src_real_B))
        self.d_fake_adv_loss = sce_loss(self.src_fake_B, tf.zeros_like(self.src_fake_B))
        ### domain classification loss
        self.d_real_cls_loss = sce_loss(self.cls_real_B, self.real_B[:,:,:,self.image_channel:])
        ### disc loss function
        self.d_loss = self.d_real_adv_loss + self.d_fake_adv_loss + self.lambda_cls * self.d_real_cls_loss
        
        ## generator loss ##
        ### adv loss
        self.g_fake_adv_loss = sce_loss(self.src_fake_B, tf.ones_like(self.src_fake_B))
        ### domain classificatioin loss
        self.g_fake_cls_loss = sce_loss(self.cls_fake_B, self.real_B[:,:,:,self.image_channel:])
        ### reconstruction loss
        self.g_recon_loss = recon_loss(self.real_A, self.fake_A)
        ### gen loss function
        self.g_loss = self.g_fake_adv_loss + self.lambda_cls * self.g_fake_cls_loss + self.lambda_rec * self.g_recon_loss
        
        # trainable variables
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'disc' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]
#        for var in t_vars: print(var.name)
        
        # optimizer
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(self.g_loss, var_list=self.g_vars)
        
    
    def train(self):
        # summary setting
        self.summary()
        # load train data
        self.load_data()
        # variable initialize
        self.sess.run(tf.global_variables_initializer())
        # load or not checkpoint
        if self.continue_train and self.checkpoint_load():
            print(" [*] before training, Load SUCCESS ")
        else:
            print(" [!] before training, no need to Load ")
        
        #train
        
        
        
    
    
    def summary(self):
        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        
        # session
        tf.summary.scalar('d_loss', self.d_loss)
        tf.summary.scalar('g_loss', self.g_loss)
        self.summary = tf.summary.merge_all()
       
        
    def load_data(self):
        return;
    
    def checkpoint_load(self):
        print(" [*] Reading checkpoint...")
        
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)        
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.ckpt_dir, ckpt_name))
            return True
        else:
            return False    
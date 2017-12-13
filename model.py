import os
import numpy as np
import tensorflow as tf
from collections import namedtuple
from tqdm import tqdm
from glob import glob
import random

from module import generator, discriminator, cls_loss, recon_loss, gan_loss, wgan_gp_loss
from util import load_data_list, attr_extract, preprocess_attr, preprocess_image, preprocess_input, save_images

class stargan(object):
    def __init__(self,sess,args):
        #
        self.sess = sess
        self.data_dir = args.data_dir # ./data/celebA
        self.log_dir = args.log_dir # ./assets/log
        self.ckpt_dir = args.ckpt_dir # ./assets/checkpoint
        self.sample_dir = args.sample_dir # ./assets/sample
        self.epoch = args.epoch # 100
        self.batch_size = args.batch_size # 16
        self.image_size = args.image_size # 64
        self.image_channel = args.image_channel # 3
        self.nf = args.nf # 64
        self.n_label = args.n_label # 10
        self.lambda_gp = args.lambda_gp
        self.lambda_cls = args.lambda_cls # 1
        self.lambda_rec = args.lambda_rec # 10
        self.lr = args.lr # 0.0001
        self.beta1 = args.beta1 # 0.5
        self.continue_train = args.continue_train # False
        self.snapshot = args.snapshot # 100
        self.adv_type = args.adv_type
        
        # hyper-parameter for building the module
        OPTIONS = namedtuple('OPTIONS', ['batch_size', 'image_size', 'nf', 'n_label', 'lambda_gp'])
        self.options = OPTIONS(self.batch_size, self.image_size, self.nf, self.n_label, self.lambda_gp)
        
        # build model & make checkpoint saver 
        self.build_model()
        self.saver = tf.train.Saver()
        
    def build_model(self):
        # placeholder
        # input_image: A, target_image: B
        self.real_A = tf.placeholder(tf.float32,
                                     [self.batch_size, self.image_size, self.image_size, self.image_channel + self.n_label],
                                     name = 'input_images')
        self.real_B = tf.placeholder(tf.float32,
                                     [self.batch_size, self.image_size, self.image_size, self.image_channel + self.n_label],
                                     name = 'target_images')
        self.attr_B = tf.placeholder(tf.float32, [self.batch_size, self.n_label], name='target_attr')
        
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [self.batch_size, self.image_size, self.image_size, self.image_channel],
                                            name = 'fake_images_sample') # use when updating discriminator
        
        # generate image
        self.fake_B = generator(self.real_A, self.options, False, name='gen')
        self.fake_A = generator(tf.concat([self.fake_B,self.real_A[:,:,:,self.image_channel:]], axis=3), self.options, True, name='gen')
        
        # discriminate image
        # src: real or fake, cls: domain classification 
        self.src_real_B, self.cls_real_B = discriminator(self.real_B[:,:,:,:self.image_channel], 
                                                         self.options, False, name='disc')
        self.g_src_fake_B, self.g_cls_fake_B = discriminator(self.fake_B, self.options, True, name='disc') # use when updating generator
        self.d_src_fake_B, self.d_cls_fake_B = discriminator(self.fake_B_sample, self.options, True, name='disc') # use when updating discriminator
        
        # loss
        ## discriminator loss ##
        ### adversarial loss
        if self.adv_type == 'WGAN':
            gp_loss = wgan_gp_loss(self.real_B[:,:,:,:self.image_channel], self.fake_B_sample, self.options)
            self.d_adv_loss = tf.reduce_mean(self.d_src_fake_B) - tf.reduce_mean(self.src_real_B) + gp_loss
        else: # 'GAN'
            d_real_adv_loss = gan_loss(self.src_real_B, tf.ones_like(self.src_real_B))
            d_fake_adv_loss = gan_loss(self.d_src_fake_B, tf.zeros_like(self.d_src_fake_B))
            self.d_adv_loss = d_real_adv_loss + d_fake_adv_loss
        ### domain classification loss
        self.d_real_cls_loss = cls_loss(self.cls_real_B, self.attr_B)
        ### disc loss function
        self.d_loss = self.d_adv_loss + self.lambda_cls * self.d_real_cls_loss
        
        ## generator loss ##
        ### adv loss
        if self.adv_type == 'WGAN':
            self.g_adv_loss = -tf.reduce_mean(self.fake_B)
        else: # 'GAN'
            self.g_adv_loss = gan_loss(self.g_src_fake_B, tf.ones_like(self.g_src_fake_B))
        ### domain classificatioin loss
        self.g_fake_cls_loss = cls_loss(self.g_cls_fake_B, self.attr_B)
        ### reconstruction loss
        self.g_recon_loss = recon_loss(self.real_A[:,:,:,:self.image_channel], self.fake_A)
        ### gen loss function
        self.g_loss = self.g_adv_loss + self.lambda_cls * self.g_fake_cls_loss + self.lambda_rec * self.g_recon_loss
        
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
        
        # load train data list & load attribute data
        dataA_files = load_data_list(self.data_dir)
        dataB_files = np.copy(dataA_files)
        self.attr_names, self.attr_list = attr_extract(self.data_dir)
        
        # variable initialize
        self.sess.run(tf.global_variables_initializer())
        
        # load or not checkpoint
        if self.continue_train and self.checkpoint_load():
            print(" [*] before training, Load SUCCESS ")
        else:
            print(" [!] before training, no need to Load ")
        
        batch_idxs = len(dataA_files) // self.batch_size # 182599
        count = 0
        #train
        for epoch in range(self.epoch):
            np.random.shuffle(dataA_files)
            np.random.shuffle(dataB_files)
            
            for idx in tqdm(range(batch_idxs)):
                count += 1
                # 
                dataA_list = dataA_files[idx * self.batch_size : (idx+1) * self.batch_size]
                dataB_list = dataB_files[idx * self.batch_size : (idx+1) * self.batch_size]
                attrA_list = [self.attr_list[os.path.basename(val)] for val in dataA_list]
                attrB_list = [self.attr_list[os.path.basename(val)] for val in dataB_list]
                
                # get batch images and labels
                attrA, attrB = preprocess_attr(self.attr_names, attrA_list, attrB_list)
                imgA, imgB = preprocess_image(dataA_list, dataB_list, self.image_size)
                dataA, dataB = preprocess_input(imgA, imgB, attrA, attrB, self.image_size, self.n_label)
                
                # updatae G network
                feed = { self.real_A: dataA, self.real_B: dataB, self.attr_B: np.array(attrB) }
                fake_B, _, g_loss, g_summary = self.sess.run([self.fake_B, self.g_optim, self.g_loss, self.g_sum],
                                                             feed_dict = feed)
                
                # update D network
                feed = { self.fake_B_sample: fake_B, self.real_B: dataB, self.attr_B: np.array(attrB) }
                _, d_loss, d_summary = self.sess.run([self.d_optim, self.d_loss, self.d_sum], feed_dict = feed)
                
                # summary
                self.writer.add_summary(g_summary, count)
                self.writer.add_summary(d_summary, count)
                
                # save checkpoint and samples
                if count % self.snapshot == 0:
                    print("Iter: %06d, g_loss: %4.4f, d_loss: %4.4f" % (count, g_loss, d_loss))
                    
                    # checkpoint
                    self.checkpoint_save(count)
                    
                    # save samples (from test dataset)
                    self.sample_save(count)
            
   
    def summary(self):
        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        
        # session : discriminator
        sum_d_1 = tf.summary.scalar('disc/adv_loss', self.d_adv_loss)
        sum_d_2 = tf.summary.scalar('disc/real_cls_loss', self.d_real_cls_loss)
        sum_d_3 = tf.summary.scalar('disc/d_loss', self.d_loss)
        self.d_sum = tf.summary.merge([sum_d_1, sum_d_2, sum_d_3])
        
        # session : generator
        sum_g_1 = tf.summary.scalar('gen/adv_loss', self.g_adv_loss)
        sum_g_2 = tf.summary.scalar('gen/fake_cls_loss', self.g_fake_cls_loss)
        sum_g_3 = tf.summary.scalar('gen/recon_loss', self.g_recon_loss)
        sum_g_4 = tf.summary.scalar('gen/g_loss', self.g_loss)
        self.g_sum = tf.summary.merge([sum_g_1, sum_g_2, sum_g_3, sum_g_4])
       
    
    def checkpoint_load(self):
        print(" [*] Reading checkpoint...")
        
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)        
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.ckpt_dir, ckpt_name))
            return True
        else:
            return False    
        
        
    def checkpoint_save(self, step):
        model_name = "stargan.model"
        self.saver.save(self.sess,
                        os.path.join(self.ckpt_dir, model_name),
                        global_step=step)
        
        
    def sample_save(self, step):
        test_files = glob(os.path.join(self.data_dir, 'test', '*'))
        
        # [5,6] with the seequnce of (realA, realB, fakeB), totally 10 set save
        testA_list = random.sample(test_files, 10)
        testB_list = random.sample(test_files, 10)
        attrA_list = [self.attr_list[os.path.basename(val)] for val in testA_list]
        attrB_list = [self.attr_list[os.path.basename(val)] for val in testB_list]
        
        # get batch images and labels
        attrA, attrB = preprocess_attr(self.attr_names, attrA_list, attrB_list)
        imgA, imgB = preprocess_image(testA_list, testB_list, self.image_size)
        dataA, dataB = preprocess_input(imgA, imgB, attrA, attrB, self.image_size, self.n_label)
                        
        # generate fakeB
        feed = { self.real_A: dataA, self.real_B: dataB }
        fake_B = self.sess.run(self.fake_B, feed_dict = feed)
        
        # save samples
        sample_file = os.path.join(self.sample_dir, '%06d.jpg'%(step))
        save_images(imgA, imgB, fake_B, self.image_size, sample_file, num=10)
                
        
    
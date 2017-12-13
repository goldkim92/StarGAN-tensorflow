import tensorflow as tf
from ops import conv2d, deconv2d, relu, lrelu, instance_norm, tanh

def generator(images, options, reuse=False, name='gen'):
    # reuse or not
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
            
        # down sampling
        x = relu(instance_norm(conv2d(images, options.nf, ks=7, s=1, name='gen_ds_conv1'), 'in1_1'))
        x = relu(instance_norm(conv2d(x, 2*options.nf, ks=4, s=2, name='gen_ds_conv2'), 'in1_2'))
        x = relu(instance_norm(conv2d(x, 4*options.nf, ks=4, s=2, name='gen_ds_conv3'), 'in1_3'))
        
        # bottleneck
        x = relu(instance_norm(conv2d(x, 4*options.nf, ks=3, s=1, name='gen_bn_conv1'), 'in2_1'))
        x = relu(instance_norm(conv2d(x, 4*options.nf, ks=3, s=1, name='gen_bn_conv2'), 'in2_2'))
        x = relu(instance_norm(conv2d(x, 4*options.nf, ks=3, s=1, name='gen_bn_conv3'), 'in2_3'))
        x = relu(instance_norm(conv2d(x, 4*options.nf, ks=3, s=1, name='gen_bn_conv4'), 'in2_4'))
        x = relu(instance_norm(conv2d(x, 4*options.nf, ks=3, s=1, name='gen_bn_conv5'), 'in2_5'))
        x = relu(instance_norm(conv2d(x, 4*options.nf, ks=3, s=1, name='gen_bn_conv6'), 'in2_6'))
        
        # up sampling
        x = relu(instance_norm(deconv2d(x, 2*options.nf, ks=4, s=2, name='gen_us_deconv1'), 'in3_1'))
        x = relu(instance_norm(deconv2d(x, options.nf, ks=4, s=2, name='gen_us_deconv2'), 'in3_2'))
        x = tanh(deconv2d(x, 3, ks=7, s=1, name='gen_us_dwconv3'))
        
        return x

def discriminator(images, options, reuse=False, name='disc'):
    # reuse or not
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
            
        # input & hidden layer
        x = lrelu(conv2d(images, options.nf, ks=4, s=2, name='disc_conv1'))
        x = lrelu(conv2d(x, 2*options.nf, ks=4, s=2, name='disc_conv2'))
        x = lrelu(conv2d(x, 4*options.nf, ks=4, s=2, name='disc_conv3'))
        x = lrelu(conv2d(x, 8*options.nf, ks=4, s=2, name='disc_conv4'))
        x = lrelu(conv2d(x, 16*options.nf, ks=4, s=2, name='disc_conv5'))
        x = lrelu(conv2d(x, 32*options.nf, ks=4, s=2, name='disc_conv6'))
        # (batch, h/64, w/64, 2048)
        
        # output layer
        x = conv2d(x, 1+options.n_label, ks=1, s=1, name='disc_conv7') # (batch, h/64, w/64, 1+n)
        x = tf.reshape(x,[-1,1+options.n_label])  # (batch, 1+n)
        src = x[:,0]
        cls = x[:,1:]
        return src, cls

def wgan_gp_loss(real_img, fake_img, options, epsilon):
    hat_img = epsilon * real_img + (1-epsilon) * fake_img
    gradients = tf.gradients(discriminator(hat_img, options, reuse=True, name='disc')[0], xs=[hat_img])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
    gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.))
    
    return options.lambda_gp * gradient_penalty
        
def gan_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels))

def cls_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels))

def recon_loss(image1, image2):
    return tf.reduce_mean(tf.abs(image1 - image2))
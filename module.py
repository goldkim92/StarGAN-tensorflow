import tensorflow as tf

def generator(images, options, reuse=False, name='gen'):
    return;

def discriminator(images, options, reuse=False, name='disc'):
    return;

def sce_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels))

def abs_criterion(image1, image2):
    return tf.reduce_mean(tf.abs(image1 - image2))
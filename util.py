import os
from glob import glob
from collections import namedtuple
import numpy as np
import scipy.misc as scm
import matplotlib.pyplot as plt


def load_data_list(data_dir):
    path = os.path.join(data_dir, 'train', '*')
    file_list = glob(path)
    return file_list


def attr_extract(data_dir):
    attr_list = {}
    
    path = os.path.join(data_dir, 'list_attr_celeba.txt')
    file = open(path,'r')
    
    n = file.readline()
    n = int(n.split('\n')[0]) #  # of celebA img: 202599
    
    attr_line = file.readline()
    attr_names = attr_line.split('\n')[0].split() # attribute name
    
    for line in file:
        row = line.split('\n')[0].split()
        img_name = row.pop(0)
        row = [int(val) for val in row]
#    img = img[..., ::-1] # bgr to rgb
        attr_list[img_name] = row   
    
    file.close()
    return attr_names, attr_list


def preprocess_attr(attr_names, attrA_list, attrB_list, attr_keys):
#    attr_keys = ['Black_Hair','Blond_Hair','Brown_Hair', 'Male', 'Young','Mustache','Pale_Skin']
    attrA = []
    attrB = []    

    for i in range(len(attrA_list)):
        tmpA = [attrA_list[i][attr_names.index(val)] for val in attr_keys]
        tmpA = [1. if val == 1 else 0. for val in tmpA]
        attrA.append(tmpA)
        tmpB = [attrB_list[i][attr_names.index(val)] for val in attr_keys]
        tmpB = [1. if val == 1 else 0. for val in tmpB]
        attrB.append(tmpB)

    return attrA, attrB


def preprocess_image(dataA_list, dataB_list, image_size, phase='train'):
    imgA = [get_image(img_path, image_size, phase=phase) for img_path in dataA_list]
    imgA = np.array(imgA)
    
    imgB = [get_image(img_path, image_size, phase=phase) for img_path in dataB_list]
    imgB = np.array(imgB)
    return imgA, imgB
        
    
def preprocess_input(imgA, imgB, attrA, attrB, image_size, n_label):
    # dataA = imgA + attrB , dataB = imgB + attrA
    attrA = np.tile(np.reshape(attrA, [-1,1,1,n_label]),[1,image_size,image_size,1])
    attrB = np.tile(np.reshape(attrB, [-1,1,1,n_label]),[1,image_size,image_size,1])
    dataA = np.concatenate((imgA, attrB), axis=3)
    dataB = np.concatenate((imgB, attrA), axis=3)
    return dataA, dataB


def get_image(img_path, data_size, phase='train'):
    img = scm.imread(img_path)
    img_crop = img[15:203,9:169,:]
    img_resize = scm.imresize(img_crop,[data_size,data_size,3])
    img_resize = img_resize/127.5 - 1.
    
    if phase == 'train' and np.random.random() >= 0.5:
        img_resize = np.flip(img_resize,1)
    
    return img_resize


def inverse_image(img):
    img = (img + 1.) * 127.5
    img[img > 255] = 255.
    img[img < 0] = 0.
    return img.astype(np.uint8)


def save_images(realA, realB, fake_B, image_size, sample_file, num=10):
    # [5,6] with the seequnce of (realA, realB, fakeB), total 10 set save
    if np.array_equal(realA, realB): # for test
        img = np.concatenate((realA[:5,:,:,:],fake_B[:5,:,:,:],
                          realA[5:,:,:,:],fake_B[5:,:,:,:]), axis=0)
        img = make3d(img, image_size, row=5, col=4)
    else: # for sample while training
        img = np.concatenate((realA[:5,:,:,:],realB[:5,:,:,:],fake_B[:5,:,:,:],
                          realA[5:,:,:,:],realB[5:,:,:,:],fake_B[5:,:,:,:]), axis=0)
        img = make3d(img, image_size, row=5, col=6)
    img = inverse_image(img)
    scm.imsave(sample_file, img)


def make3d(img, image_size, row, col):
    # img.shape = [row*col, h, w, c]
    # final: [row*h, col*w, c]
    img = np.reshape(img, [col,row,image_size,image_size,3]) # [col, row, h, w, c]
    img = unstack(img, axis=0) # col * [row, h, w, c]
    img = np.concatenate(img, axis=2) # [row, h, col*w, c]
    img = unstack(img, axis=0) # row * [h, col*w, c]
    img = np.concatenate(img, axis=0) # [row*h, col*w, c]
    return img


def unstack(img, axis):
    d =img.shape[axis]
    arr = [np.squeeze(a,axis=axis) for a in np.split(img, d, axis=axis)]
    return arr
#%% import
import os
from collections import namedtuple

#%% file and directory (csv txt file)
def attr_extract():
    attr = {}
    
    path = os.path.join('.','data', 'list_attr_celeba.txt')
    file = open(path,'r')
    
    n = file.readline()
    n = int(n.split('\n')[0]) #  # of celebA img: 202599
    
    attr_line = file.readline()
    attrs = attr_line.split('\n')[0].split() # attribute name
    attrs[0] = 'Clock_Shadow' # originally it was 5_o_ Clock_Shadow, but in order to put it in namedtuple, we must delete the number
    
    ATTRS = namedtuple('ATTRS', [attr for attr in attrs])
    
    for line in file:
        row = line.split('\n')[0].split()
        img_name = row.pop(0)
        row = ATTRS(*row)
        attr[img_name] = row
    
    file.close()
    
    return attr


Black Hair
Blond Hair
Brown Hair
Gray Hair
Male
Young
Mustache
No_Beard
Eyeglasses

# -*- coding: utf-8 -*-
"""
Created on 2019-04-28

@author: Vasili
"""
import  os
import  tensorflow as tf
p=os.path.join('.','data_path_save','1521112368','checkpoints/')
print(os.listdir(p))
ckpt_file = tf.train.latest_checkpoint(p)
if __name__ == '__main__':
    pass
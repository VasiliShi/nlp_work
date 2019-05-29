# -*- coding: utf-8 -*-
"""
Created on 2019-04-23

@author: Vasili
"""
import  argparse
import  logging

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger

def get_entity(tag_seq, char_seq):
    per = get_PER_entity(tag_seq, char_seq)
    loc = get_LOC_entity(tag_seq, char_seq)
    org = get_ORG_entity(tag_seq, char_seq)
    return per, loc, org

def get_PER_entity(tag_seq, char_seq):

    tags = []
    name = ''
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-PER':
            if name != '':
                tags.append(name)
            name = char
        elif tag == 'I-PER':
            name += char
        else:
            if name != '':
                tags.append(name)
            name = ''
        if i == len(char_seq) - 1 and name != '':
            tags.append(name)
    return  tags


def get_LOC_entity(tag_seq, char_seq):
    tags = []
    name = ''
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-LOC':
            if name != '':
                tags.append(name)
            name = char
        elif tag == 'I-LOC':
            name += char
        else:
            if name != '':
                tags.append(name)
            name = ''
        if i == len(char_seq) - 1 and name != '':
            tags.append(name)
    return  tags

def get_ORG_entity(tag_seq, char_seq):
    tags = []
    name = ''
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-ORG':
            if name != '':
                tags.append(name)
            name = char
        elif tag == 'I-ORG':
            name += char
        else:
            if name != '':
                tags.append(name)
            name = ''
        if i == len(char_seq) - 1 and name != '':
            tags.append(name)
    return  tags



if __name__ == '__main__':
    demo_tag = ['B-PER', 'I-PER', 'I-PER', 0, 0, 0, 'B-LOC', 'I-LOC', 'I-LOC', 'B-LOC', 'I-LOC', 'I-LOC', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    demo_sent = ['毛', '泽', '东', '出', '生', '在', '湖', '南', '省', '益', '阳', '市', '，', '他', '小', '时', '候', '经', '常', '出', '去', '玩']

    a = get_LOC_entity(demo_tag, demo_sent)
    print(a)


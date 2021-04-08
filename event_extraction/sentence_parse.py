#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:48:54 2021

@author: vasili
"""

from ltp import LTP
class LtpParser:
    def __init__(self):

        
        
        self.ltp = LTP()


    '''语义角色标注'''
    def format_labelrole(self, srl):
        
#        arcs = self.parser.parse(words, postags)
#        roles = self.labeller.label(words, postags, arcs)
        roles_dict = {}
        for index, roles in enumerate(srl):
            if len(roles) == 0:continue
            roles_dict[index] = {role[0]:[role[0],role[1], role[2]] for role in roles}
        return roles_dict

    '''句法分析---为句子中的每个词语维护一个保存句法依存儿子节点的字典'''
    def build_parse_child_dict(self, words, postags, arcs):
        child_dict_list = []
        format_parse_list = []
        for index in range(len(words)):
            child_dict = dict()
            for arc_index in range(len(arcs)):
                if arcs[arc_index][0] == index+1:   #arcs的索引从1开始
                    if arcs[arc_index][2] in child_dict:
                        child_dict[arcs[arc_index][2]].append(arc_index)
                    else:
                        child_dict[arcs[arc_index][2]] = []
                        child_dict[arcs[arc_index][2]].append(arc_index)
            child_dict_list.append(child_dict)
        rely_id = [arc[0] for arc in arcs]  # 提取依存父节点id
        relation = [arc[2] for arc in arcs]  # 提取依存关系
        heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]  # 匹配依存父节点词语
        for i in range(len(words)):
            # ['ATT', '李克强', 0, 'nh', '总理', 1, 'n']
            a = [relation[i], words[i], i, postags[i], heads[i], rely_id[i]-1, postags[rely_id[i]-1]]
            format_parse_list.append(a)

        return child_dict_list, format_parse_list

    '''parser主函数'''
    def parser_main(self, sentence):
        seg, hidden = self.ltp.seg([sentence])
        pos = self.ltp.pos(hidden)
        dep = self.ltp.dep(hidden)
        srl = self.ltp.srl(hidden)
        
        # words = list(self.segmentor.segment(sentence))
        # postags = list(self.postagger.postag(words))
        # arcs = self.parser.parse(words, postags)
        
        child_dict_list, format_parse_list = self.build_parse_child_dict(seg[0], pos[0], dep[0])
        roles_dict = self.format_labelrole(srl[0])
        return seg[0], pos[0], child_dict_list, roles_dict, format_parse_list


if __name__ == '__main__':
    parse = LtpParser()
    sentence = '李克强总理今天来我家了,我感到非常荣幸'
    words, postags, child_dict_list, roles_dict, format_parse_list = parse.parser_main(sentence)
    print(words, len(words))
    print(postags, len(postags))
    print(child_dict_list, len(child_dict_list))
    print(roles_dict)
    print(format_parse_list, len(format_parse_list))
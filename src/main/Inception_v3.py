#coding=utf-8
'''
Created on 2017年6月28日
@author: wingdi
导入inception-v3 .pb文件，测试图片
'''
import tensorflow as tf
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt

#dir param
label_dir = '/Users/zhengying/Documents/EclipseWorkspace/Inception-v3/src/data/imagenet_2012_challenge_label_map_proto.pbtxt'
uid_dir = '/Users/zhengying/Documents/EclipseWorkspace/Inception-v3/src/data/imagenet_synset_to_human_label_map.txt'
pd_dir = '/Users/zhengying/Documents/4_mechine_learning/dataset/inception/inception_model/classify_image_graph_def.pb'
test_img_dir = '/Users/zhengying/Documents/EclipseWorkspace/Inception-v3/src/data/images/'

class NodeLookup(object):
    def __init__(self):
        #label and img path 
        label_lookup_path = label_dir  
        uid_lookup_path = uid_dir
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        # 加载分类字符串n********对应分类名称的文件
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        #匹配0或多个n或数字，匹配0或多个空格，非空白字符，逗号
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:#处理txt建立uid和分类名称的对应关系
            parsed_items = p.findall(line)
            #获取编号字符串n********
            uid = parsed_items[0]
            #获取分类名称
            human_string = parsed_items[2]
            #保存编号字符串n********与分类名称映射关系
            uid_to_human[uid] = human_string

        # 加载分类字符串n********对应分类编号1-1000的文件
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        node_id_to_uid = {}
        for line in proto_as_ascii:#处理pdtxt 建立uid和类别号的对应关系
            if line.startswith('  target_class:'):
                #获取分类编号1-1000
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                #获取编号字符串n********
                target_class_string = line.split(': ')[1]
                #保存分类编号1-1000与编号字符串n********映射关系
                node_id_to_uid[target_class] = target_class_string[1:-2]

        #建立分类编号1-1000对应分类名称的映射关系
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            #获取分类名称
            name = uid_to_human[val]
            #建立分类编号1-1000到分类名称的映射关系
            node_id_to_name[key] = name
        return node_id_to_name#返回1-1000和分类的对应关系

    #传入分类编号1-1000返回分类名称 根据编号返回分类的名称
    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]

#创建一个图来存放google训练好的模型
with tf.gfile.FastGFile(pd_dir, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    #遍历目录
    for root,dirs,files in os.walk(test_img_dir):
        for file in files:
            if ".jpg" != os.path.splitext(file)[1]:
                continue
            #载入图片
            image_data = tf.gfile.FastGFile(os.path.join(root,file), 'rb').read()
            predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})#图片格式是jpg格式
            print("predictions_raw:",predictions)
            predictions = np.squeeze(predictions)#把结果转为1维数据 是一连串的概率
            print("predictions:",np.shape(predictions))

            #打印图片路径及名称
            image_path = os.path.join(root,file)
            print(image_path)
            #显示图片
#             img=Image.open(image_path)
#             plt.imshow(img)
#             plt.axis('off')
#             plt.show()

            #排序
            top_k = predictions.argsort()[-5:][::-1]# top-5 进行倒序排列 取最后5个
            node_lookup = NodeLookup()
            for node_id in top_k:    
                #获取分类名称
                human_string = node_lookup.id_to_string(node_id)
                #获取该分类的置信度 计算概率值
                score = predictions[node_id]
                print('%s (score = %.5f)' % (human_string, score))#打印分类的名称和概率
            print()

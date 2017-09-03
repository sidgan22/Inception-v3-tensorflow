#coding=utf-8
'''
Created on 2017年6月28日
@author: wingdi
保存inception模型 用tensorboard查看
'''
import tensorflow as tf
import os
import tarfile
import requests

#模型保存路径
inception_pretrain_model_dir = "/Users/zhengying/Documents/4_mechine_learning/dataset/inception/inception_model"
#模型结构
log_dir = '/Users/zhengying/Documents/4_mechine_learning/dataset/inception/inception_log'

def init_data():
    #inception模型下载地址 模型的参数已经确定下来
    inception_pretrain_model_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

    #创建模型存放路径
    if not os.path.exists(inception_pretrain_model_dir):
        os.makedirs(inception_pretrain_model_dir)

    #声明文件名和文件保存路径
    filename = inception_pretrain_model_url.split('/')[-1]
    filepath = os.path.join(inception_pretrain_model_dir, filename)

    #如果模型不存在 下载模型
    if not os.path.exists(filepath):
        print("download: ", filename)
        r = requests.get(inception_pretrain_model_url, stream=True)
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    print("finish: ", filename)
    #解压文件
    tarfile.open(filepath, 'r:gz').extractall(inception_pretrain_model_dir)
    #创建模型结构存放文件

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

def train():
    #classify_image_graph_def.pb为google训练好的模型 
    inception_graph_def_file = os.path.join(inception_pretrain_model_dir, 'classify_image_graph_def.pb')
    with tf.Session() as sess:
        #创建一个图来存放google训练好的模型
        with tf.gfile.FastGFile(inception_graph_def_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')#把模型导入到我们的图里面
        #保存图的结构
        writer = tf.summary.FileWriter(log_dir, sess.graph)#把图的结构写到log里面 方便我们查看
        writer.close()
    
if __name__ == "__main__":
    init_data()
    train()    
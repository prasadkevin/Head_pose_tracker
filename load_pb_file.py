# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:09:24 2020

@author: prasa
"""

import tensorflow as tf
from tensorflow.python.platform import gfile

GRAPH_PB_PATH = './assets/pose_model/saved_model.pb' #path to your .pb file

with tf.Session() as sess:
  print("load graph")
  with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    graph_nodes=[n for n in graph_def.node]


for node in graph_nodes[-10:]:
    
#    if node.name == "output_locations/Reshape":
        print(node)
        
        
loaded = tf.saved_model.load("./assets/pose_model/saved_model.pb")


print(list(loaded.signatures.keys())) # ["saved_model"]
        
 
#import tensorflow as tf
#from tensorflow.python.platform import gfile
#GRAPH_PB_PATH = './assets/pose_model/saved_model.pb'
#with tf.Session() as sess:
#   print("load graph")
#   with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
#       graph_def = tf.GraphDef()
#   graph_def.ParseFromString(f.read())
#   sess.graph.as_default()
#   tf.import_graph_def(graph_def, name='')
#   graph_nodes=[n for n in graph_def.node]
#   names = []
#   for t in graph_nodes:
#      names.append(t.name)
#   print(names)
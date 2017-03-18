from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import os.path
import re
import sys
import tarfile

from scipy import linalg, mat, dot
import scipy.spatial as sp
import numpy as np
from six.moves import urllib
from scipy import linalg, mat, dot #ADDED
import tensorflow as tf


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def compareFeatures(image):

  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:

    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    #Get pool3 layer - get the image features and print them
    feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
    feature_set = sess.run(feature_tensor,
                            {'DecodeJpeg/contents:0': image_data})
    feature_set = np.squeeze(feature_set)
    #print(feature_set) 


    directory = '/Users/L299490/Documents/Win7share/256_ObjectCategories/127.laptop-101'
    print(directory)
    for filename in os.listdir(directory):
     if filename.endswith(".jpg"):
      imagec = os.path.join(directory, filename)	
      if not tf.gfile.Exists(imagec):
       print("File " + filename + " does not exist")
       continue
      image_data = tf.gfile.FastGFile(imagec, 'rb').read()
      fs = sess.run(feature_tensor, {'DecodeJpeg/contents:0': image_data})
      fs = np.squeeze(fs)      
      # Calculate cosine distance between feature set and transposed set
      #c = dot(feature_set,fs.T)/linalg.norm(feature_set)/linalg.norm(fs)
      c = 1 - sp.distance.cosine(feature_set, fs) # same as above
      print(c)
      #print(image + " - " + filename + " - disatance: ", c)      	
      #print(os.path.join(directory, filename))
      continue
     else:
      continue  


def main(_):
  #maybe_download_and_extract()
  image = (FLAGS.image_file if FLAGS.image_file else
           os.path.join(FLAGS.model_dir, 'cropped_panda.jpg'))
  #run_inference_on_image(image)
  compareFeatures(image)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # classify_image_graph_def.pb:
  #   Binary representation of the GraphDef protocol buffer.
  # imagenet_synset_to_human_label_map.txt:
  #   Map from synset ID to a human readable string.
  # imagenet_2012_challenge_label_map_proto.pbtxt:
  #   Text representation of a protocol buffer mapping a label to synset ID.
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--image_file',
      type=str,
      default='',
      help='Absolute path to image file.'
  )
  parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=5,
      help='Display this many predictions.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
#a = mat([-0.711,0.730])
#b = mat([-1.099,0.124])
#c = dot(a,b.T)/linalg.norm(a)/linalg.norm(b)
#print(c)
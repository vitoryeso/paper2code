import tensorflow as tf
import numpy as np


class yoloV1Loss:
  def __init__(self, lambda_coord, lambda_class, S, B, C, width=224, height=224):
    self.lambda_coord = lambda_coord
    self.lambda_class = lambda_class
    self.B = B
    self.C = C

    self.width = width
    self.height = height

    self.S = range(1, self.width/S + 1)
    map(lambda x: x*self.width, self.S)


  def transform_label(pseudo_label):
     # y: array of objects
     #      like (n_objects, 4coords + class)
     # output:
     #      (S, S, B*(5 + C)) 
     #      x and y relative to the cell
     #      w and h relative to the entire img
     #      confidence is a boolean here because we need the predictions to compute the IOU
     #      if confidence is true, we compute iou. if not, we dont.
    label = tf.zeros((self.S, self.S, B*(5+self.C)))
    for obj in pseudo_label:
      x, y = obj[0], obj[1]
      s1, s2 = 0, 0
      for s in self.S:
        if x <= s:
            s1 = s
        if y <= s:
            s2 = s
      label_x = x - (s1 - self.S[0])
      label_y = y - (s2 - self.S[0])
      
      w,  h = obj[2], obj[3]
      class_id = obj[4]
      label[self.S.index(s1), self.S.index(s2), class_id*5:class_id*5 + 5] = tf.Tensor([label_x, label_y, w, h, 1], dtype="float16")



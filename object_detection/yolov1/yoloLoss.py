import tensorflow as tf
import numpy as np
from math import floor


class yoloV1Loss:
  def __init__(self, lambda_coord=0.5, lambda_class=0.5, S=8, B=2, C=3, width=224, height=224):
    self.lambda_coord = lambda_coord
    self.lambda_class = lambda_class
    self.B = B
    self.C = C

    self.width = width
    self.height = height

    self.Sx = S
    self.Sy = S
    #self.S = range(1, self.width/S + 1)
    #map(lambda x: x*self.width, self.S)


  def transform_label(self, pseudo_label):
     # pseudo_label: array of objects
     #      like (n_objects, 4coords + class)
     # output:
     #      (S, S, B*(5 + C)) 
     #      x and y relative to the cell
     #      w and h relative to the entire img
     #      confidence is a boolean here because we need the predictions to compute the IOU
     #      if confidence is true, we compute iou. if not, we dont.
    label = np.zeros((self.Sx, self.Sy, self.B*(5+self.C)), dtype=np.float16)
    for x, y, w, h, class_id in pseudo_label:
        Sx = int(x*self.Sx/self.width)
        Sy = int(y*self.Sy/self.height)
        start = 0
        end = 5 + self.C

        # while already have an object with the same class in the same cell, we go to the next B
        while label[Sx, Sy, start] != 0:
            start += end + 1
            end = start + 5 + self.C
        if end < self.B*(5 + self.C):
            # new coordinates relatives to the begin of the respective cell
            new_x = (x*self.Sx - Sx*self.width)/self.width
            new_y = (y*self.Sy - Sy*self.height)/self.height
            # [x, y, w, h] + [class_0_prob, ... class_C_prob] + [confidence_score]
            # remember, confidence is P(obj) * IOU(true, predicted)
            # but we only compute this IOU at the training, therefore we will multiply later label[5]*IOU
            classes = np.zeros(self.C + 1, dtype=np.float16)

            classes[class_id] = 1.0 # class score
            classes[-1] = 1.0 # confidence 

            coords = np.array([new_x, new_y, w/self.width, h/self.height], dtype=np.float16)

            label[Sx, Sy, start:end] = np.hstack([coords, classes])
            del classes, coords
            
    return label

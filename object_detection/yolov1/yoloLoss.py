import tensorflow as tf
import numpy as np
from math import floor

class yoloV1Loss:
  def __init__(self, lambda_coord=0.5, lambda_noobj=0.5, S=8, B=2, C=3, width=224, height=224):
    self.lambda_coord = lambda_coord
    self.lambda_class = lambda_noobj
    self.B = B
    self.C = C

    self.width = width
    self.height = height

    self.Sx = S
    self.Sy = S

  def transform_label(self, pseudo_label):
     # pseudo_label: array of objects
     #      like (n_objects, 4coords + class)
     # output:
     #      (S, S, B*len([confidence, x, y, w, h]) + C) 
     #      confidence is a boolean here because we need the predictions to compute the IOU
     #      if confidence is true, we compute iou. if not, we dont.
     #      x and y relative to the cell
     #      w and h relative to the entire img
    label = np.zeros((self.Sx, self.Sy, self.B*5 + self.C), dtype=np.float16)
    for x, y, w, h, class_id in pseudo_label:
        Sx = int(x*self.Sx/self.width)
        Sy = int(y*self.Sy/self.height)

        # class probabilities
        assert class_id < self.C
        if label[Sx, Sy, 5*self.B + class_id] != 1.0:
            label[Sx, Sy, 5*self.B + class_id] = 1.0

        start = 0
        # while already have an object in the same cell, we go to the next Bounding box
        while label[Sx, Sy, start] != 0 and start <= 5*(self.B-1):
            start += 5

        if start <= 5*(self.B-1):
            # confidence. this means that we'll compute the iou for this Bounding box
            label[Sx, Sy, start] = 1.0
            
            # new coordinates relatives to the begin of the respective cell
            new_x = (x*self.Sx - Sx*self.width)/self.width
            new_y = (y*self.Sy - Sy*self.height)/self.height

            coords = np.array([new_x, new_y, w/self.width, h/self.height], dtype=np.float16)

            print(coords)
            label[Sx, Sy, start+1:start+5] = coords
            del coords
            
    return label
"""

    def compute(self, label, predictions):
        loss = 0.0
        label = self.transform_label(label)

        # todo: loss for each coordinate and confidence, if that bounding box is responsible for the object
        # loss for classes predictions. if the object class falls into the cell 
        # loss for each cell which failed predicting an object center
        # loss for each class, at each grid cell
        
        # (S, S, (x, y, w, h, c0, ..., cC, confidence))
        for sx in range(self.Sx):
            for sy in range(self.Sy):
                for b in range(1, self.B):

                    # if exist an object in this cell (checking if confidence is != 0)
                    if label[sx, sy, b*(5+self.C)] != 0:
                        if predictions[sx, sy, b*(5+self.C)] == 0:
                            # compute obj confidence loss
                            label_confidence = label[sx, sy, b*(5+self.C)]
                            predictions_confidence = predictions[sx, sy, b*(5+self.C)]
 
                            loss += (label_confidence - predictions_confidence)**2

                    # adding 
                    else:

""" 

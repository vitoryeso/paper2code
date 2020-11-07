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

            label[Sx, Sy, start+1:start+5] = coords
            del coords
            
    return label

    def compute(self, label, predictions):
        total_loss = 0.0
        label = self.transform_label(label)

        # first we need to get the bounding boxes which are responsible for the prediction
        # if the cell_label have an or more objects, we choose the bounding box with highest IOU to be the responsible

        for sx in range(self.Sx):
            for sy in range(self.Sy):
                for b in range(self.B):
                    # checking if exist more any object inside this cell
                    if label[sx, sy, b*5] == 0.0:
                        # compute noobj loss
                        # label confidence is zero. lambda_noobj * (label_confidence - predicted_confidence)**2
                        loss += self.lambda_noobj * predictions[sx, sy, b*5]**2
                    else:

                    

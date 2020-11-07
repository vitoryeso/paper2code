import tensorflow as tf
import numpy as np

from utils import IOU
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

    def responsible_box(idx, label, predictions):
        max_iou = 0.0
        responsible = 0
        for b in self.B:
            iou = IOU(label[5*idx+1:5*idx+4], predictions[5*b+1:5*b+4])
            if max_iou < iou:
                max_iou = iou
                responsible = b
        return b, max_iou

    def compute(self, label, predictions):
        total_loss = 0.0
        label = self.transform_label(label)

        # first we need to get the bounding boxes which are responsible for the prediction
        # if the cell_label have an or more objects, we choose the bounding box with highest IOU to be the responsible

        for sx in range(self.Sx):
            for sy in range(self.Sy):
                # classes probabilities loss
                for c in self.C:
                    label_prob = label[sx, sy, self.B*5 + c]
                    predict_prob = label[sx, sy, self.B*5 + c]
                    loss += (label_prob - predict_prob)**2

                bndbox_ids = []
                n_objects = 0
                for b in range(self.B):
                    # checking if exist more any object inside this cell
                    if label[sx, sy, b*5] == 0.0:
                        # compute noobj loss
                        # label confidence is zero. lambda_noobj * (label_confidence - predicted_confidence)**2
                        loss += self.lambda_noobj * predictions[sx, sy, b*5]**2
                    else:
                        true_objects += 1
                        bndbox_ids.append(b)
                if true_objects > 1:
                    print("oh noooo")
                elif true_objects == 1:
                    responsible, iou = self.responsible_box(bndbox_ids[0], label[sx, sy, :], predictions[sx, sy, :])

                    # obj confidence loss
                    loss += (iou - predictions[sx, sy, responsible*5])**2
                        
                    # x, y loss
                    x_loss = (label[sx, sy, 5*bndbox_ids[0] + 1] - predictions[sx, sy, responsible*5 + 1])**2
                    y_loss = (label[sx, sy, 5*bndbox_ids[0] + 2] - predictions[sx, sy, responsible*5 + 2])**2
                    loss += self.lambda_coord * (x_loss + y_loss)

                    # w, h loss
                    w_loss = (label[sx, sy, 5*bndbox_ids[0] + 3] - predictions[sx, sy, responsible*5 + 3])**2
                    h_loss = (label[sx, sy, 5*bndbox_ids[0] + 4] - predictions[sx, sy, responsible*5 + 4])**2
                    loss += self.lambda_coord * (w_loss + h_loss)

        return loss

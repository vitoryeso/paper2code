import numpy as np
import time

from yoloLoss import yoloV1Loss

loss = yoloV1Loss()

label = [ [[40, 44, 100, 100, 1]], [[120, 30, 160, 60, 0], [10, 200, 80, 90, 2]], [[40, 44, 100, 100, 1]], [[40, 44, 100, 100, 1]], [[40, 44, 100, 100, 1]] ]

predictions = np.zeros((5,8,8,13))
start = time.time()
print("loss: ", loss.compute_batch(label, predictions))
end = time.time()
print("time: ", end - start)

import numpy as np
from yoloLoss import yoloV1Loss

loss = yoloV1Loss()

label = [120, 30, 160, 60, 0]
transformed = loss.transform_label([label])

print(label)
print(transformed.shape)

predictions = np.zeros_like(transformed)
print("loss: ", loss.compute(transformed, predictions))

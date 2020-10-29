from yoloLoss import yoloV1Loss

loss = yoloV1Loss()

label = [120, 30, 60, 60, 2]
transformed = loss.transform_label([label])

print(label)
print(transformed.shape)
for i in range(transformed.shape[-1]):
    print(transformed[:, :, i])

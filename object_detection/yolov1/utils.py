def overlap(x1, w1, x2, w2):
    left = max(x1 - w1/2, x2 - w2/2)
    right = min(x1 + w1/2, x2 + w2/2)
    return right - left

def intersection(boxA, boxB):
    # box is like [x, y, w, h]
    x1, y1, w1, h1 = boxA
    x2, y2, w2, h2 = boxB
    intersection_w = overlap(x1, w1, x2, w2)
    intersection_h = overlap(y1, h1, y2, h2)
    if intersection_w < 0 or intersection_h < 0:
        return 0
    return intersection_w * intersection_h

def IOU(boxA, boxB):
    i = intersection(boxA, boxB)

    # the union is simply Area(A) + Area(B) - intersection(A, B)
    u = boxA[2]*boxA[3] + boxB[2]*boxB[3] - i
    if i == 0 or u == 0:
        return 0
    return i/u

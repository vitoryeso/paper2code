def IOU(boxA, boxB):
    xAmin, xAmax = boxA[0] - boxA[2]/2, boxA[0] + boxA[2]/2
    yAmin, yAmax = boxA[1] - boxA[3]/2, boxA[1] + boxA[3]/2
    
    xBmin, xBmax = boxB[0] - boxB[2]/2, boxB[0] + boxB[2]/2
    yBmin, yBmax = boxB[1] - boxB[3]/2, boxB[1] + boxB[3]/2

    # intersection rectangle
    xA = max(xAmin, xBmin)
    yA = max(yAmin, yBmin)
    xB = min

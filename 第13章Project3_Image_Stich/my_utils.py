import cv2

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
        print('width is None', dim)

    else:
        r = width / float(w)
        dim = (width, int(h * r))
        print('height is None', dim)
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
import cv2


def resize(image_path):
    # image = cv2.imread('./images/%s' % image_path)
    image = cv2.imread("./%s" % image_path)
    image_resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    # cv2.imwrite("./images/%s" % image_path, image_resized)
    print("aqui")
    cv2.imwrite("./%s" % image_path, image_resized)

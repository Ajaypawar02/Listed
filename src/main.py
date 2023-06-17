import sys
import cv2
from PIL import Image
sys.path.append('src')

from ImageCaption import ImageCaption
from args import args

if __name__ == '__main__':
  ImgCap = ImageCaption()
  photo = cv2.imread("/home/ajay/Listed/data/Image3.png")
  num_captions = 3
  print(ImgCap.photo_upload(photo, num_captions))
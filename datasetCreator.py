import cv2
vidcap = cv2.VideoCapture('4.mp4')
success,image = vidcap.read()
count = 88
while success:
  cv2.imwrite("Dataset/Intrested/frame%d.jpg" % count, image)
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
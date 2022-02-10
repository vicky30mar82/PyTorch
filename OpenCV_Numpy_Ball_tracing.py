import numpy as np
import cv2

def read_rgb_image(image_name, show):
  rgb_image = cv2.imread(image_name)
  if show:
    cv2.imshow("RGB Image", rgb_image)
  return rgb_image

def filter_color(rgb_image, lower_bound_color, upper_bound_color):
  hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
  cv2.imshow("hsv_image", hsv_image)
  yellowLower = (30, 150, 100)
  yellowUpper = (50, 255, 255)

  mask = cv2.inRange(hsv_image, lower_bound_color, upper_bound_color)
  return mask

def getContours(bin_image):
  _, contours, hierarchy = cv2.findContours(bin_image.copy,
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
  return contours

  def draw_ball_contours(bin_image, rgb_image, contours):
    black_image = np.zeros([bin_image.shape[0], bin_image.shape[1], 3], 'uint8')

    for c in contours:
      area = cv2.contourArea(c)
      perimeter = cv2.arcLength(c, True)

      ((x, y), radius) = cv2.minEnclosingCircle(c)
      if(area > 100):
        cv2.drawContours(rgb_image, [c], -1, (150, 250, 150), 1)
        cv2.drawContours(black_image, [c], -1, (150, 250, 150), 1)
        cx, cy = get_contour_center(c)
        cv2.circle(rgb_image, (cx, cy), (int)(radius), (0,0,255), 1)
        cv2.circle(black_image, (cx, cy), (int)(radius), (0,0,255), 1)
        cv2.circle(black_image, (cx, cy), 5, (150, 150, 255), -1)
        print("Area:{}, Perimeter: {}".format(area, perimeter))
    print("Num of Contours: {}".format(len(contours)))
    cv2.imshow("RGB Image Contours", rgb_image)
    cv2.imshow("RGB Image Contours", black_image)

  def get_contour_center(contour):
    M = cv2.moments(contour)
    cx = -1
    cy = -1
    if(M['m00'] != 0):
      cx = int(M['m10']/ M['m00'])
      cy = int(M['m01']/ M['m00'])
    return cx, cy

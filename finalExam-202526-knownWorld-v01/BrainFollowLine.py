from pyrobot.brain import Brain

import cv2
from pyrobot.tools.followLineTools import findLineDeviation

class BrainFollowLine(Brain):
 
  NO_FORWARD = 0
  SLOW_FORWARD = 0.1
  MED_FORWARD = 0.5
  FULL_FORWARD = 1.0

  NO_TURN = 0
  MED_LEFT = 0.5
  HARD_LEFT = 1.0
  MED_RIGHT = -0.5
  HARD_RIGHT = -1.0

  NO_ERROR = 0

  def setup(self):
    pass

  def destroy(self):
    cv2.destroyAllWindows()

  def step(self):
    cv_image = self.robot.getImage()

    # display the robot's camera's image using opencv
    cv2.imshow("Stage Camera Image", cv_image)
    cv2.waitKey(1)

    # write the image to a file, for debugging etc.
    # cv2.imwrite("debug-capture.png", cv_image)

    # convert the image into grayscale
    imageGray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # determine the robot's deviation from the line.
    foundLine,error = findLineDeviation(imageGray)
    # print("findLineDeviation returned ",foundLine,error)

#     # display a debug image using opencv
#     middleRowIndex = cv_image.shape[1]//2
#     centerColumnIndex = cv_image.shape[0]//2
#     if (foundLine):
#       cv2.rectangle(cv_image,
#                     (int(error*middleRowIndex)+middleRowIndex-5,
#                      centerColumnIndex-5),
#                     (int(error*middleRowIndex)+middleRowIndex+5,
#                      centerColumnIndex+5),
#                     (0,255,0),
#                     3)
#     cv2.imshow("Debug findLineDeviation", cv_image)
#     cv2.waitKey(1)

    # A trivial on-off controller
    if (foundLine):
      if (error > self.NO_ERROR):
        # print("Turning right.")
        self.move(self.FULL_FORWARD,self.HARD_RIGHT)
      elif (error < self.NO_ERROR):
        # print("Turning left.")
        self.move(self.FULL_FORWARD,self.HARD_LEFT)
      else:
        # print("Straight ahead.")
        self.move(self.FULL_FORWARD,self.NO_TURN)
    else:
      # if we can't see the line we just stop, this isn't very smart
      # print("Stop.")
      self.move(self.NO_FORWARD,self.NO_TURN)

def INIT(engine):
  assert (engine.robot.requires("range-sensor") and
	  engine.robot.requires("continuous-movement"))

  return BrainFollowLine('BrainFollowLine', engine)

from pyrobot.brain import Brain

import cv2
from pyrobot.tools.followLineTools import findLineDeviation

class BrainFollowLine(Brain):
 
  NO_FORWARD = 0
  VERY_SLOW_FORWARD = 0.05
  SLOW_FORWARD = 0.1
  MED_FORWARD = 0.5
  FULL_FORWARD = 1.0

  NO_TURN = 0
  MED_LEFT = 0.5
  HARD_LEFT = 1.0
  MED_RIGHT = -0.5
  HARD_RIGHT = -1.0

  NO_ERROR = 0

  # Obstacle avoidance distance
  OBSTACLE_STOP = 0.55
  OBSTACLE_WARN = 0.80

  # PD gains
  LINE_KP = 0.9
  LINE_KD = 0.5

  # Lost-line search behavior
  SEARCH_SLOW_AFTER = 15
  SEARCH_REVERSE_AFTER = 35

  def setup(self):
    self.last_error = 0.0
    self._lost_line_steps = 0
    self._search_dir = self.HARD_RIGHT

  def destroy(self):
    cv2.destroyAllWindows()

  def _min_range(self, group_name, default=3.0):
    try:
      sensors = self.robot.range[group_name]
    except Exception:
      return default

    if not sensors:
      return default

    distances = [sensor.distance() for sensor in sensors if sensor is not None]
    if not distances:
      return default
    return min(distances)

  def step(self):
    cv_image = self.robot.getImage()

    # display the robot's camera's image using opencv
    cv2.imshow("Stage Camera Image", cv_image)
    cv2.waitKey(1)

    # write the image to a file, for debugging etc.
    cv2.imwrite("debug-capture.png", cv_image)

    # convert the image into grayscale
    imageGray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # determine the robot's deviation from the line.
    foundLine,error = findLineDeviation(imageGray)
    print("findLineDeviation returned ",foundLine,error)

    # display a debug image using opencv
    middleRowIndex = cv_image.shape[1]//2
    centerColumnIndex = cv_image.shape[0]//2
    if (foundLine):
      cv2.rectangle(cv_image,
                    (int(error*middleRowIndex)+middleRowIndex-5,
                     centerColumnIndex-5),
                    (int(error*middleRowIndex)+middleRowIndex+5,
                     centerColumnIndex+5),
                    (0,255,0),
                    3)
    cv2.imshow("Debug findLineDeviation", cv_image)
    cv2.waitKey(1)

    # Read front sensor groups to detect and avoid obstacles.
    front = self._min_range("front")
    front_left = self._min_range("front-left")
    front_right = self._min_range("front-right")

    # Highest priority: avoid collisions with objects.
    if (front < self.OBSTACLE_STOP):
      if (front_left < front_right):
        self.move(self.SLOW_FORWARD, self.MED_RIGHT)
      else:
        self.move(self.SLOW_FORWARD, self.MED_LEFT)
      return

    if (front_left < self.OBSTACLE_WARN):
      self.move(self.MED_FORWARD, self.MED_RIGHT)
      return

    if (front_right < self.OBSTACLE_WARN):
      self.move(self.MED_FORWARD, self.MED_LEFT)
      return

    # Line tracking with a PD controller
    if (foundLine):
      # Reset lost line counter when the line is found
      self._lost_line_steps = 0

      # PD Control
      d_error = error - self.last_error
      turn = -(self.LINE_KP * error + self.LINE_KD * d_error)

      # Limit turn to max range
      turn = max(self.HARD_RIGHT, min(self.HARD_LEFT, turn))

      # Saving last error for the next step
      self.last_error = error

      # Forward speed
      forward = max(self.VERY_SLOW_FORWARD, self.FULL_FORWARD - abs(turn * 1.5))
      self.move(forward, turn)
    else:
      self._lost_line_steps += 1
      if self._lost_line_steps == 1:
        # Keep the direction based on error across multiple steps.
        if self.last_error > 0:
          self._search_dir = self.MED_RIGHT
        elif self.last_error < 0:
          self._search_dir = self.MED_LEFT

      # If the line is lost for a few steps, starts searching it
      if self._lost_line_steps < self.SEARCH_SLOW_AFTER:
        self.move(self.SLOW_FORWARD, self._search_dir)

      # Slows down the turn if the line is lost for more steps.
      elif self._lost_line_steps < self.SEARCH_REVERSE_AFTER:
        self.move(self.VERY_SLOW_FORWARD, self._search_dir)

      # If the line is not found after many steps, switch search direction and reset counter
      else:
        self._search_dir *= -1
        self._lost_line_steps = 0
        self.move(self.VERY_SLOW_FORWARD, self._search_dir)

def INIT(engine):
  assert (engine.robot.requires("range-sensor") and
	  engine.robot.requires("continuous-movement"))

  return BrainFollowLine('BrainFollowLine', engine)

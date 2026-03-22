from pyrobot.brain import Brain

import cv2
from pyrobot.tools.followLineTools import findLineDeviation

class BrainFollowLine(Brain):
 
  NO_FORWARD = 0
  CRAWL_FORWARD = 0.05
  SLOW_FORWARD = 0.1
  MED_FORWARD = 0.5
  FULL_FORWARD = 1.0

  NO_TURN = 0
  MED_LEFT = 0.5
  HARD_LEFT = 1.0
  MED_RIGHT = -0.5
  HARD_RIGHT = -1.0

  NO_ERROR = 0

  # Distance thresholds (meters) for obstacle avoidance.
  OBSTACLE_STOP = 0.55
  OBSTACLE_WARN = 0.80

  # Proportional gain for line tracking.
  LINE_KP = 0.9
  LINE_KD = 0.5

  SHOW_CAMERA = True

  # Steps before slowing the search pivot (avoids overshooting).
  PIVOT_SLOW_AFTER  = 15
  # Steps before reversing search direction (pendulum, for >90° turns).
  PIVOT_REVERSE_AFTER = 35

  def setup(self):
    self._window_ready = False
    self._last_error = 0.0
    self._lost_line_steps = 0
    self._search_dir = self.HARD_RIGHT

    if self.SHOW_CAMERA:
      try:
        cv2.startWindowThread()
      except Exception:
        pass

  def destroy(self):
    if self.SHOW_CAMERA:
      try:
        cv2.destroyAllWindows()
      except Exception:
        pass

  def _show_camera(self, window_name, image):
    if not self.SHOW_CAMERA:
      return

    try:
      if not self._window_ready:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        self._window_ready = True
      cv2.imshow(window_name, image)
      cv2.waitKey(1)
    except Exception:
      # Keep navigation running even if GUI backend is unavailable.
      pass

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

    self._show_camera("Stage Camera Image", cv_image)

    # convert the image into grayscale
    imageGray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # determine the robot's deviation from the line.
    foundLine,error = findLineDeviation(imageGray)
    print("findLineDeviation returned ",foundLine,error)

    # Display a debug image with a green rectangle showing the detected line position.
    middleRowIndex = cv_image.shape[1]//2
    centerColumnIndex = cv_image.shape[0]//2
    debug_image = cv_image.copy()
    if (foundLine):
      cv2.rectangle(debug_image,
                    (int(error*middleRowIndex)+middleRowIndex-5,
                     centerColumnIndex-5),
                    (int(error*middleRowIndex)+middleRowIndex+5,
                     centerColumnIndex+5),
                    (0,255,0),
                    3)
    self._show_camera("Debug findLineDeviation", debug_image)

    # Read front sonar groups to detect and avoid obstacles.
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

    # If no obstacle is near, follow the line.
    if (foundLine):
      self._lost_line_steps = 0
      # Reset search direction based on current error so next loss is handled correctly.
      self._search_dir = self.HARD_RIGHT if error >= 0 else self.HARD_LEFT

      d_error = error - self._last_error
      turn = -(self.LINE_KP * error + self.LINE_KD * d_error)

      # If error changes rapidly (overshooting), dampen the turn heavily.
      if abs(d_error) > 0.15:
        turn *= 0.4  # reduce by 60% to prevent overshoot

      turn = max(self.HARD_RIGHT, min(self.HARD_LEFT, turn))
      self._last_error = error

      abs_error = abs(error)
      if abs_error > 0.55:
        forward = self.NO_FORWARD       # pure pivot on sharp turns
      elif abs_error > 0.40:
        forward = self.SLOW_FORWARD
      elif abs_error > 0.20:
        forward = 0.7  # faster cruising
      else:
        forward = 1.0  # full speed on straight

      print(f"FOLLOW | error={error:.4f} d_error={d_error:.4f} turn={turn:.4f} forward={forward:.2f}")
      self.move(forward, turn)
    else:
      # Pendulum search for sharp bends including >90°.
      self._lost_line_steps += 1

      if (self._lost_line_steps == 1):
        # First loss: commit to direction based on last known error.
        self._search_dir = self.HARD_RIGHT if self._last_error >= 0 else self.HARD_LEFT

      elif (self._lost_line_steps == self.PIVOT_REVERSE_AFTER + 1):
        # Didn't find line in time: reverse direction (pendulum).
        self._search_dir = self.HARD_LEFT if self._search_dir == self.HARD_RIGHT else self.HARD_RIGHT

      # Slow down pivot after a while to avoid overshooting the line.
      if (self._lost_line_steps > self.PIVOT_SLOW_AFTER):
        pivot_speed = abs(self._search_dir) * 0.5  # half rate
        turn = pivot_speed if self._search_dir > 0 else -pivot_speed
      else:
        turn = self._search_dir

      print(f"SEARCH | step={self._lost_line_steps} last_error={self._last_error:.4f} turn={turn:.4f} dir={self._search_dir:.2f}")
      self.move(self.NO_FORWARD, turn)

def INIT(engine):
  assert (engine.robot.requires("range-sensor") and
	  engine.robot.requires("continuous-movement"))

  return BrainFollowLine('BrainFollowLine', engine)

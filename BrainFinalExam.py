import cv2
from pyrobot.brain import Brain

class BrainFinalExam(Brain):
    # ... class variables, etc.
    def setup(self):
    #   Put code here which should only be executed ONCE at
    # the start. For example, connecting to the camera,
    # setting up segmentation etc. Keep references in
    # instance variables.
        self.capture = cv2.VideoCapture(0)
    # set capture resolution ...
    def destroy(self):
    # Put code here to clean up everything when shutting down.
        cv2.destroyAllWindows()
    #def step(self):
        # Put code here which should be executed during every step.

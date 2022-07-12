from quadruped_spring.env.tasks import robot_tasks as rt
from quadruped_spring.utils.base_collection import CollectionBase

# Tasks to be learned with reinforcement learning

#     - "JUMPING_ON_PLACE_HEIGHT"
#         Sparse reward, maximizing max relative height respect the initial one
#         + malus on crashing + bonus on minimizing forward jumping distance
#         + bonus on minimizing the orientation changes respect the default one

#     - "JUMPING_FORWARD"
#         Sparse reward, maximizing max flight time
#         + malus on crashing + bonus on maximizing forward jumping distance
#         + bonus on minimizing the orientation changes respect the default one


class TaskCollection(CollectionBase):
    """Utility to collect all the implemented robot tasks."""

    def __init__(self):
        super().__init__()
        self.JUMPING_CURRICULUM = rt.JumpingCurriculum
        self.JUMPING_IN_PLACE = rt.JumpingInPlace
        self.JUMPING_FORWARD = rt.JumpingForward
        self.JUMPING_IN_PLACE_DENSE = rt.JumpingInPlaceDense
        self.MULTIPLE_JUMPING_IN_PLACE = rt.MultipleJumpingInPlace
        self.JUMPING_FORWARD_HEIGHT = rt.JumpingForwardHeight
        self.NO_TASK = rt.NoTask
        self._element_type = "task"
        self._dict = {
            "JUMPING_CURRICULUM": self.JUMPING_CURRICULUM,
            "JUMPING_FORWARD": self.JUMPING_FORWARD,
            "JUMPING_IN_PLACE_DENSE": self.JUMPING_IN_PLACE_DENSE,
            "JUMPING_IN_PLACE": self.JUMPING_IN_PLACE,
            "MULTIPLE_JUMPING_IN_PLACE": self.MULTIPLE_JUMPING_IN_PLACE,
            "JUMPING_FORWARD_HEIGHT": self.JUMPING_FORWARD_HEIGHT,
            "NO_TASK": self.NO_TASK,
        }

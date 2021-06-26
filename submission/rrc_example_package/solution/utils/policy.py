class RandomPolicy(object):

    """Dummy policy which uses random actions"""

    def __init__(self, action_space):
        """TODO: to be defined. """
        self.action_space = action_space

    def predict(self, observation):
        return self.action_space.sample()

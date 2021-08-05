# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
    alt_policies.states
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    Basic implemetnation of alternate grasping and control strategies for the
    trifinger robot.

    :copyright: (c) 2021 by Aditya Kamireddypalli.
    :license: MIT, see MIT LICENSE for more details.
    
    Implementation of alternate grasping and control strategies for the
    trifinger robot.
    Copyright © 2021 Aditya Kamireddypalli
    
    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
    OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    
"""


class State(object):
    def __init__(self, env):
        self.env = env

    def connect(self, *states):
        raise NotImplementedError

    def __call__(self, obs, info=None):
        raise NotImplementedError

    def reset(self):
        """clear any internal variables this state may keep"""
        raise NotImplementedError

    def get_action(self, position=None, torque=None, frameskip=1):
        """Wraps the robot actions in a dictionary

        @param position: list of joint positions
        @type  position: list
        @param torque: list of joint torques
        @type  torque: list
        @param frameskip: number of time steps to skip
        @type  frameskip: int

        @return:  returns a dictionary containing {position, torque, frameskip}
        @rtype :  dict

        """
        return {'position': position, 'torque': torque, 'frameskip': frameskip}


class StateMachine(object):
    def __init__(self, env):
        self.env = env
        self.init_state = self.build()

    def build(self):
        """Instantiate states and connect them.

        make sure to make all of the states instance variables so that
        they are reset when StateMachine.reset is called!

        Returns: base_states.State (initial state)


        Ex:

        self.state1 = State1(env, args)
        self.state2 = State2(env, args)

        self.state1.connect(...)
        self.state2.connect(...)

        return self.state1
        """
        raise NotImplementedError

    def reset(self):
        self.state = self.init_state
        self.info = {}
        print("==========================================")
        print("Resetting State Machine...")
        print(f"Entering State: {self.state.__class__.__name__}")
        print("==========================================")
        for attr in vars(self).values():
            if isinstance(attr, State):
                attr.reset()

    def __call__(self, obs):
        prev_state = self.state
        action, self.state, self.info = self.state(obs, self.info)
        if prev_state != self.state:
            print("==========================================")
            print(f"Entering State: {self.state.__class__.__name__}")
            print("==========================================")
        if action['frameskip'] == 0:
            return self.__call__(obs)
        else:
            return action

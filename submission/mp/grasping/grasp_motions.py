from .ik import IKUtils
from mp.action_sequences import ScriptedActions
from mp.const import INIT_JOINT_CONF
import numpy as np


def get_grasp_approach_actions(env, obs, grasp, move_finger=False):
    """get_grasp_approach_actions.
    retrieves grasp approach actions. Given the grasp (cartesian points on the
    object where the tips need to reach), this funtion returns an action
    generator that executes the grasp.

    Args:
        env: robot envrionment
        obs: observations from the robot
        grasp: selected grasp
    """

    # generates actions to carry out grasp
    action_sequence = ScriptedActions(env, obs['robot_tip_positions'], grasp)

    # estimates pre-grasp tip positions and joint configuration
    pregrasp_joint_conf, pregrasp_tip_pos = get_safe_pregrasp(
        env, obs, grasp
    )
    if pregrasp_joint_conf is None:
        raise RuntimeError('Feasible heuristic grasp approach is not found.')

    # actual generation of actions to carry out grasp
    action_sequence.add_raise_tips()
    action_sequence.add_heuristic_pregrasp(pregrasp_tip_pos)
    if not move_finger:
        action_sequence.add_grasp(coef=0.6)
    else:
        action_sequence.add_grasp(coef=1.0)

    act_seq = action_sequence.get_action_sequence(
        action_repeat=4 if env.simulation else 12 * 4,
        action_repeat_end=40 if env.simulation else 400
    )

    return act_seq


def get_safe_pregrasp(env, obs, grasp, candidate_margins=[1.1, 1.3, 1.5]):
    pregrasp_tip_pos = []
    pregrasp_jconfs = []
    ik_utils = IKUtils(env)
    init_tip_pos = env.platform.forward_kinematics(INIT_JOINT_CONF)
    mask = np.eye(3)[grasp.valid_tips, :].sum(0).reshape(3, -1)

    for margin in candidate_margins:
        tip_pos = grasp.T_cube_to_base(grasp.cube_tip_pos * margin)
        tip_pos = tip_pos * mask + (1 - mask) * init_tip_pos
        qs = ik_utils.sample_no_collision_ik(tip_pos)
        if len(qs) > 0:
            pregrasp_tip_pos.append(tip_pos)
            pregrasp_jconfs.append(qs[0])
            print('candidate margin coef {}: safe'.format(margin))
        else:
            print('candidate margin coef {}: no ik solution found'.format(margin))

    if len(pregrasp_tip_pos) == 0:
        print('warning: no safe pregrasp pose with a margin')
        tip_pos = grasp.T_cube_to_base(grasp.cube_tip_pos * candidate_margins[0])
        tip_pos = tip_pos * mask + (1 - mask) * init_tip_pos
        qs = ik_utils.sample_ik(tip_pos)
        if len(qs) == 0:
            return None, None
        else:
            pregrasp_tip_pos.append(tip_pos)
            pregrasp_jconfs.append(qs[0])
    return pregrasp_jconfs[-1], pregrasp_tip_pos[-1]

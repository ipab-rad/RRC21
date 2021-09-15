
from mp.const import MU, VIRTUAL_CUBOID_HALF_SIZE, INIT_JOINT_CONF, DICE_SIZE, DICE_HALF_SIZE
from mp.utils import Transform, keep_state
from mp.grasping.ik import IKUtils
from mp.grasping.force_closure import CuboidForceClosureTest, CoulombFriction
import itertools
import numpy as np

def get_finger_configuration(env, pos, quat):
    """get_finger_configuration.
    Retrieves finger to touch the closest centre of the cube. It returns a
    list of `Grasps` that carry out this action,

    Args:
        env: Simulation/Real environment
        pos: position of the cube
        quat: orientation of the cube
    """

    primitive = FingerPrimitives(env, pos, quat)
    fing_mov = primitive.get_heuristic_grasps()
    return fing_mov

def get_finger_configuration_dice(env, pos, quat):
    """get_finger_configuration.
    Retrieves finger to touch the closest centre of the cube. It returns a
    list of `Grasps` that carry out this action,

    Args:
        env: Simulation/Real environment
        pos: position of the cube
        quat: orientation of the cube
    """

    primitive = FingerPrimitives(env, pos, quat, halfsize=DICE_HALF_SIZE+0.01,
                                ignore_collision=False, allow_partial_sol=True)
    fing_mov = primitive.get_heuristic_grasps()
    return fing_mov

def get_side_face_centers(half_size, object_ori):
    """get_side_face_centers.
    Retrieves face centers of the sides of the cube that are perpedicular to
    the ground

    Args:
        half_size: half size if the cube, as specified by a constant value
        within the environment
        object_ori: current object environment
    """
    R_base_to_cube = Transform(np.zeros(3), object_ori).inverse()
    z_cube = R_base_to_cube(np.array([0, 0, 1]))
    axis = np.argmax(np.abs(z_cube))
    points = []
    for ax in range(3):
        if ax != axis:
            points.append(sample(ax, 1, half_size, np.zeros(3)))
            points.append(sample(ax, -1, half_size, np.zeros(3)))
    return np.array(points)

def sample(ax, sign, half_size=VIRTUAL_CUBOID_HALF_SIZE,
           shrink_region=[0.6, 0.6, 0.6]):
    point = np.empty(3)
    for i in range(3):
        if i == ax:
            point[ax] = sign * half_size[ax]
        else:
            point[i] = np.random.uniform(-half_size[i] * shrink_region[i],
                                         half_size[i] * shrink_region[i])
    return point

def get_three_sided_heuristic_grasps(half_size, object_ori):
    """get_three_sided_heuristic_grasps.

    Retrieves a set of 3S3F grasps.

    Args:
        half_size: half size of the cube. Defined by a constant within the
        environment
        object_ori: orientation of the object
    """
    points = get_side_face_centers(half_size, object_ori)
    grasps = []
    for ind in range(4):
        grasps.append(points[np.array([x for x in range(4) if x != ind])])
    return grasps

def get_two_sided_heurictic_grasps(half_size, object_ori):
    """get_two_sided_heurictic_grasps.
    Retrieves a set of 2S3F grasps.

    Args:
        half_size: half size of the cube. Defined by a constant within the
        environment
        object_ori: orientation of the object
    """
    side_centers = get_side_face_centers(half_size, object_ori)
    ax1 = side_centers[1] - side_centers[0]
    ax2 = side_centers[3] - side_centers[2]
    g1 = np.array([
        side_centers[0],
        side_centers[1] + 0.15 * ax2,
        side_centers[1] - 0.15 * ax2,
    ])
    g2 = np.array([
        side_centers[1],
        side_centers[0] + 0.15 * ax2,
        side_centers[0] - 0.15 * ax2,
    ])
    g3 = np.array([
        side_centers[2],
        side_centers[3] + 0.15 * ax1,
        side_centers[3] - 0.15 * ax1,
    ])
    g4 = np.array([
        side_centers[3],
        side_centers[2] + 0.15 * ax1,
        side_centers[2] - 0.15 * ax1,
    ])
    return [g1, g2, g3, g4]

def get_all_heuristic_grasps(half_size, object_ori):
    """get_all_heuristic_grasps.

    This provides you with a concatenated list of two sided and three sided
    centred grasps.

    Args:
        half_size: half size of the cube, defined as a constant in the
        environment
        object_ori: orientation of the cube.
    """
    return (
        get_three_sided_heuristic_grasps(half_size, object_ori)
        + get_two_sided_heurictic_grasps(half_size, object_ori)
    )
    # return (
    #     get_two_finger_heurictic_grasps(half_size, object_ori) +
    #     get_two_sided_heurictic_grasps(half_size, object_ori) +
    #     get_three_sided_heuristic_grasps(half_size, object_ori)
    # )

class FingerPrimitives:
    """FingerPrimitives.
    This class implements primitive finger motion. The intended usage for this
    class is the contruction of motion primitives to push, drag and grasp cubes
    and dice relevant to the `Trifinger` robot tasks.

    The purpose is to have a general movement primitive for the individual
    fingers of the robot. Grapsing primitives are implemented elsewhere.
    """

    def __init__(self, env, pos, quat, slacky_collision=True,
                 halfsize=VIRTUAL_CUBOID_HALF_SIZE,
                 ignore_collision=False, avoid_edge_faces=True, yawing_grasp=False, allow_partial_sol=False):
        self.object_pos = pos
        self.object_ori = quat
        self.ik = env.pinocchio_utils.inverse_kinematics
        self.id = env.platform.simfinger.finger_id
        self.tip_ids = env.platform.simfinger.pybullet_tip_link_indices
        self.link_ids = env.platform.simfinger.pybullet_link_indices
        self.T_cube_to_base = Transform(pos, quat)
        self.T_base_to_cube = self.T_cube_to_base.inverse()
        self.env = env
        self.ik_utils = IKUtils(env, yawing_grasp=yawing_grasp)
        self.slacky_collision = slacky_collision
        self._org_tips_init = np.array(
            self.env.platform.forward_kinematics(INIT_JOINT_CONF)
        )
        self.halfsize = halfsize
        self.tip_solver = CuboidForceClosureTest(halfsize, CoulombFriction(MU))
        self.ignore_collision = ignore_collision
        self.avoid_edge_faces = avoid_edge_faces
        self.yawing_grasp = yawing_grasp
        self.allow_partial_sol = allow_partial_sol

    def _reject(self, points_base):
        """_reject.
        This function applies a few checks on the estimated feasible ordered
        tip positions (positions on the cube that have been associated with
        robot fingers).

        To the best of my knowledge these checks are the following:
            force closure - check if fingers do not apply residual force
            collision check - fingers do not collide with the cube when
            trajectory is being executed

        Args:
            points_base:
        """
        # if not self.tip_solver.force_closure_test(self.T_cube_to_base,
        #                                           points_base):
        #     return True, None
        if self.ignore_collision:
            q = self.ik_utils._sample_ik(points_base)
        elif self.allow_partial_sol:
            q = self.ik_utils._sample_ik(points_base, allow_partial_sol=True)
        else:
            q = self.ik_utils._sample_no_collision_ik(
                points_base, slacky_collision=self.slacky_collision, diagnosis=False
            )
        if q is None:
            return True, None
        return False, q

    def assign_positions_to_fingers(self, tips):
        """assign_positions_to_fingers.
        Assigns fingers from the Trifinger robot to each of the computed tip
        positions on the cube.

        Args:
            tips: positions on the cube where the tips of the `Trifinger` robot
            could go.
        """

        # a dict to maintain the cost associated with different permutatuins of
        # `Trifinger` robot finger tips with computed finger tips.
        cost_to_inds = {}

        # iterate through different permutations of `Trifinger` robot tips and
        # maintain their respective costs
        for v in itertools.permutations([0, 1, 2]):
            sorted_tips = tips[v, :]
            cost = np.linalg.norm(sorted_tips - self._org_tips_init)
            cost_to_inds[cost] = v

        inds_sorted_by_cost = [
            val for key, val in sorted(cost_to_inds.items(), key=lambda x: x[0])
        ]
        opt_inds = inds_sorted_by_cost[0]
        opt_tips = tips[opt_inds, :]

        # verbose output
        return opt_tips, opt_inds, inds_sorted_by_cost

    def assign_position_to_finger(self, tips):
        """assign_position_to_finger.
        Assigns finger to a computed tip position on the cube. This function
        piggy backs on the `GraspSampler` class logic. It should have its on
        place.

        The purpose of this function is to try and build one finger pushing
        primitives on the cube

        Args:
            tips: point on the cube to be touched by the `Trifinger` robot
            finger tip
        """
        cost_to_inds = {}
        for v in itertools.combinations([0, 1, 2], 1):
            sorted_tips = tips[v, :]
            cost = np.linalg.norm(sorted_tips - self._org_tips_init)
            cost_to_inds[cost] = v

        inds_sorted_by_cost = [
            val for key, val in sorted(cost_to_inds.items(), key=lambda x: x[0])
        ]
        opt_inds = inds_sorted_by_cost[0]
        opt_tips = tips[opt_inds, :]

        # verbose output
        return opt_tips, opt_inds, inds_sorted_by_cost

    def get_feasible_grasps_from_tips(self, tips, finger):
        """get_feasible_grasps_from_tips.

        This function works similar to its counterpart in the `GraspSampler`
        class, however we use the mask of the `Grasp` class to instruct only
        one out of three fingers to move. Hence, it works very similar to the
        `GraspSampler` class but moves only one finger is made to move.

        The return value is an instance of the class `Grasp`.

        Args:
            tips: Estimated tip positions on the cube
            finger: finger to be moved, can take a value between 0-2
        """
        _, _, permutations_by_cost = self.assign_positions_to_fingers(tips)
        for perm in permutations_by_cost:
            ordered_tips = tips[perm, :]
            should_reject, q = self._reject(ordered_tips)
            if not should_reject:
                # use INIT_JOINT_CONF for tip positions that were not solvable
                valid_tips = [0, 1, 2]
                if self.allow_partial_sol:
                    for i in range(3):
                        if q[i * 3] is None:
                            valid_tips.remove(i)
                            q[i * 3:(i + 1) * 3] = INIT_JOINT_CONF[i * 3:(i + 1) * 3]

                # move only designated finger
                # for i in range(3):
                #     if (i != finger):
                #         valid_tips.remove(i)
                #         q[i * 3:(i + 1) * 3] = INIT_JOINT_CONF[i * 3:(i + 1) * 3]


                yield Grasp(self.T_base_to_cube(ordered_tips),
                            ordered_tips, q, self.object_pos,
                            self.object_ori, self.T_cube_to_base,
                            self.T_base_to_cube, valid_tips)

    def __call__(self, shrink_region=[0.0, 0.6, 0.0], max_retries=40):
        retry = 0
        print("sampling a random grasp...")
        with keep_state(self.env):
            while retry < max_retries:
                print('[GraspSampler] retry count:', retry)
                points = sample_side_face(3, self.halfsize, self.object_ori,
                                          shrink_region=shrink_region)
                tips = self.T_cube_to_base(points)
                for grasp in self.get_feasible_grasps_from_tips(tips):
                    return grasp
                retry += 1

        raise RuntimeError('No feasible grasp is found.')

    def get_heuristic_grasps(self):
        """get_heuristic_grasps.

        Retreives a set of heuristic grasp configurations, and returns a set of
        `Grasp` object corresponding to the heuristic grasps
        """
        grasps = get_all_heuristic_grasps(
            self.halfsize, self.object_ori,
        )

        # __import__('pudb').set_trace()
        # grasps = get_onefinger_heuristic(
        #     self.halfsize, self.object_ori,
        # )

        ret = []
        with keep_state(self.env):
            for points in grasps:

                # transform the grasp points estimated on the cube to the
                # Trifinger base coordinate system.
                __import__('pudb').set_trace()
                tips = self.T_cube_to_base(points)
                # NOTE: we sacrifice a bit of speed by not using "yield", however,
                # context manager doesn't work as we want if we use "yield".
                # performance drop shouldn't be significant
                # (get_feasible_grasps_from_tips only iterates 6 grasps!).
                # for grasp in self.get_feasible_grasps_from_tips(tips):
                #     yield grasp

                # retrieve feasible grasps for finger 0
                for finger in [0, 1, 2]:
                    ret += [grasp for grasp in
                            self.get_feasible_grasps_from_tips(tips, finger)]
            return ret

    def get_custom_grasp(self, base_tip_pos):
        q = self.ik_utils._sample_ik(base_tip_pos)
        return Grasp(self.T_base_to_cube(base_tip_pos),
                     base_tip_pos, q, self.object_pos,
                     self.object_ori, self.T_cube_to_base,
                     self.T_base_to_cube, [0, 1, 2])


class Grasp(object):
    def __init__(self, cube_tip_pos, base_tip_pos, q, cube_pos, cube_quat,
                 T_cube_to_base, T_base_to_cube, valid_tips):
        self.cube_tip_pos = cube_tip_pos
        self.base_tip_pos = base_tip_pos
        self.q = q
        self.pos = cube_pos
        self.quat = cube_quat
        self.T_cube_to_base = T_cube_to_base
        self.T_base_to_cube = T_base_to_cube
        self.valid_tips = valid_tips

    def update(self, cube_pos, cube_quat):
        self.pos = cube_pos
        self.quat = cube_quat
        self.T_cube_to_base = Transform(self.pos, self.quat)
        self.T_base_to_cube = self.T_cube_to_base.inverse()
        self.base_tip_pos = self.T_cube_to_base(self.cube_tip_pos)

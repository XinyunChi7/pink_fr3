import time

import numpy as np
import pink
import pinocchio as pin
import qpsolvers
import meshcat_shapes # web visualization
from pink.visualization import start_meshcat_visualizer
from robot_descriptions.loaders.pinocchio import load_robot_description
from tqdm import tqdm

# Define robot model
robot = load_robot_description("panda_description")
model = robot.model
data = robot.data
viz = start_meshcat_visualizer(robot)

# Define IK solver task
task = pink.tasks.FrameTask(
    "panda_link8",  # target frame pose (eef)
    position_cost=1.0,
    orientation_cost=0.75,
)

dt = 1e-3
damping = 1e-12
tolerance = 1e-3

q_solution = np.array(
    [-0.15945564,
      -1.08779923, 
      0.65036522, 
      -2.96385285, 
      -0.81375394,  
      3.30938772,
      1.17261477,  
      0.02542847,  
      0.02869188]
)
q_init = np.array(
    [
        0.32026851,
        -0.08284433,
        0.76474584,
        -1.96374742,
        0.07952368,
        3.63553733,
        2.46978477,
        0.02542847,
        0.02869188,
    ]
)

pin.forwardKinematics(model, data, q_solution)
pin.updateFramePlacements(model, data)
goal = data.oMf[model.getFrameId("panda_link8")]
task.set_target(goal)

configuration = pink.configuration.Configuration(model, data, q_init)
viz.display(configuration.q)

viewer = viz.viewer
meshcat_shapes.frame(viewer["end_effector_target"], opacity=0.5)
meshcat_shapes.frame(viewer["end_effector"], opacity=1.0)
viewer["end_effector_target"].set_transform(goal.np)

# for t in np.arange(0.0, 30.0, dt):
for step in range(1000):
    t = step * dt
    # decompose call to solve_ik to check inequality slackness:
    problem = pink.build_ik(configuration, (task,), dt, damping=damping)
    Delta_q = qpsolvers.solve_problem(problem, solver="quadprog").x

    G, h = problem.G, problem.h
    s = G.dot(Delta_q) - h
    nq = model.nq
    # print(f"{s.shape=}")
    # for lim_index, lim_type in {0: "configuration", 1: "velocity"}.items():
    #     for side_index, side in {0: "upper", 1: "lower"}.items():
    #         for i in range(model.nq):
    #             j = lim_index * (2 * nq) + side_index * nq + i
    #             print(
    #                 f"Joint {i}'s {lim_type} {side} bound ({j=}) has slackness {s[j]}"
    #             )
    velocity = Delta_q / dt

    configuration.integrate_inplace(velocity, dt)

    tcp_pose = configuration.get_transform_frame_to_world("panda_link8")
    viewer["end_effector"].set_transform(tcp_pose.np)
    print(f"Current configuration: {configuration.q}")
    print(f"TCP pose: ", tcp_pose)

    error = pin.log(tcp_pose.actInv(goal)).vector
    if (
        np.linalg.norm(error[:3]) < tolerance
        and np.linalg.norm(error[3:]) < tolerance
    ):
        print("Solved ik successfully")
    viz.display(configuration.q)
    time.sleep(dt)

else:
    print("Finished / Failed to solve ik")
import time
import numpy as np
import pink
import pinocchio as pin
import qpsolvers
from pink.visualization import start_meshcat_visualizer
from robot_descriptions.loaders.pinocchio import load_robot_description
import meshcat_shapes

# Load robot and visualizer
robot = load_robot_description("panda_description")
model = robot.model
data = robot.data
viz = start_meshcat_visualizer(robot)
viewer = viz.viewer

# IK task
task = pink.tasks.FrameTask(
    "panda_link8",
    position_cost=1.0,
    orientation_cost=0.75,
)

# IK parameters
dt = 1e-3
damping = 1e-12

# Initial joint configuration
q_init = np.array([
    0.32026851, -0.08284433, 0.76474584, -1.96374742,
    0.07952368, 3.63553733, 2.46978477, 0.02542847, 0.02869188
])
configuration = pink.configuration.Configuration(model, data, q_init)

# Compute original pose
pin.forwardKinematics(model, data, configuration.q)
pin.updateFramePlacements(model, data)
frame_id = model.getFrameId("panda_link8")
original_pose = data.oMf[frame_id]
original_rpy = pin.rpy.matrixToRpy(original_pose.rotation)

print("🔹 Original EEF Pose (Before IK):")
print(f"Position (m): x={original_pose.translation[0]:.4f}, y={original_pose.translation[1]:.4f}, z={original_pose.translation[2]:.4f}")
print("Orientation (degrees):")
print(f"  Roll  = {np.degrees(original_rpy[0]):.2f}°")
print(f"  Pitch = {np.degrees(original_rpy[1]):.2f}°")
print(f"  Yaw   = {np.degrees(original_rpy[2]):.2f}°")

# Define delta pose
delta_twist = np.array([
    0.02, 0.01, 0.01,                 # Δx, Δy, Δz (m)
    np.deg2rad(20.0), 0.0, 0.0        # Δroll = 10°, pitch/yaw = 0
])
delta_transform = pin.exp6(delta_twist)
target_pose = original_pose.act(delta_transform)
target_rpy = pin.rpy.matrixToRpy(target_pose.rotation)

print("\n🔸 Target EEF Pose (After Delta Applied):")
print(f"Position (m): x={target_pose.translation[0]:.4f}, y={target_pose.translation[1]:.4f}, z={target_pose.translation[2]:.4f}")
print("Orientation (degrees):")
print(f"  Roll  = {np.degrees(target_rpy[0]):.2f}°")
print(f"  Pitch = {np.degrees(target_rpy[1]):.2f}°")
print(f"  Yaw   = {np.degrees(target_rpy[2]):.2f}°")

# Set IK target
task.set_target(target_pose)

# Visuals
viz.display(configuration.q)
meshcat_shapes.frame(viewer["end_effector_target"], opacity=0.5)
meshcat_shapes.frame(viewer["end_effector"], opacity=1.0)
viewer["end_effector_target"].set_transform(target_pose.np)

# Solve IK
problem = pink.build_ik(configuration, (task,), dt, damping=damping)
solution = qpsolvers.solve_problem(problem, solver="quadprog")

if solution is not None:
    Delta_q = solution.x
    velocity = Delta_q / dt
    configuration.integrate_inplace(velocity, dt)

    actual_pose = configuration.get_transform_frame_to_world("panda_link8")
    actual_rpy = pin.rpy.matrixToRpy(actual_pose.rotation)

    viewer["end_effector"].set_transform(actual_pose.np)
    viz.display(configuration.q)

    print("\n✅ Actual EEF Pose (After IK Applied):")
    print(f"Position (m): x={actual_pose.translation[0]:.4f}, y={actual_pose.translation[1]:.4f}, z={actual_pose.translation[2]:.4f}")
    print("Orientation (degrees):")
    print(f"  Roll  = {np.degrees(actual_rpy[0]):.2f}°")
    print(f"  Pitch = {np.degrees(actual_rpy[1]):.2f}°")
    print(f"  Yaw   = {np.degrees(actual_rpy[2]):.2f}°")
else:
    print("❌ IK solution failed.")

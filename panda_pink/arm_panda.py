#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Franka panda arm tracking a moving target."""

import numpy as np
import qpsolvers

import meshcat_shapes
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from pink.utils import custom_configuration_vector
from pink.visualization import start_meshcat_visualizer
import pinocchio as pin

try:
    from loop_rate_limiters import RateLimiter
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples use loop rate limiters, "
        "try `[conda|pip] install loop-rate-limiters`"
    ) from exc

try:
    from robot_descriptions.loaders.pinocchio import load_robot_description
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples need robot_descriptions, "
        "try `[conda|pip] install robot_descriptions`"
    ) from exc


if __name__ == "__main__":
    robot = load_robot_description("panda_description", root_joint=None)

    viz = start_meshcat_visualizer(robot)
    viewer = viz.viewer
    meshcat_shapes.frame(viewer["end_effector_target"], opacity=0.5)
    meshcat_shapes.frame(viewer["end_effector"], opacity=1.0)

    end_effector_task = FrameTask(
        "panda_link8",
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=0.75,  # [cost] / [rad]
      
    )

    posture_task = PostureTask(
        cost=1e-3,  # [cost] / [rad]
    )

    tasks = [end_effector_task, posture_task]
    q_ref = np.array([
        0.32026851, -0.08284433, 0.76474584, -1.96374742,
        0.07952368, 3.63553733, 2.46978477, 0.02542847, 0.02869188
    ])
    configuration = pink.Configuration(robot.model, robot.data, q_ref)
    for task in tasks:
        task.set_target_from_configuration(configuration)
    viz.display(configuration.q)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]
    while True:
        # Update task targets
        end_effector_target = end_effector_task.transform_target_to_world
        print(f"t = {t:.3f} s, end-effector target: {end_effector_target.np}")
        # end_effector_target.translation[0] = 0.2 
        end_effector_target.translation[1] = 0.5 + 0.05 * np.sin(2.0 * t) # Y: oscillation
        end_effector_target.translation[2] = 0.5 # Z: height

        # yaw_angle = 0.1 * np.sin(1.5 * t)  #
        # rot_z = pin.rpy.rpyToMatrix(0.0, 0.0, yaw_angle) # Yaw rotation
        # end_effector_target.rotation = end_effector_target.rotation @ rot_z

        # Update visualization frames
        viewer["end_effector_target"].set_transform(end_effector_target.np)
        viewer["end_effector"].set_transform(
            configuration.get_transform_frame_to_world(
                end_effector_task.frame
            ).np
        )

        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt

# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import warp as wp

from . import math
from .types import Data
from .types import DisableBit
from .types import JointType
from .types import Model
from .types import array2df
from .warp_util import event_scope


@wp.kernel
def _spring_passive(
  # Model
  jnt_dofadr: wp.array(dtype=int),
  jnt_qposadr: wp.array(dtype=int),
  jnt_stiffness: wp.array(dtype=float),
  jnt_type: wp.array(dtype=int),
  qpos_spring: wp.array(dtype=float),
  # Data in
  qpos: array2df,
  # Data out
  qfrc_spring: array2df,
):
  worldid, jntid = wp.tid()
  stiffness = jnt_stiffness[jntid]
  dofid = jnt_dofadr[jntid]

  if stiffness == 0.0:
    return

  jnttype = jnt_type[jntid]
  qposid = jnt_qposadr[jntid]

  if jnttype == wp.static(JointType.FREE.value):
    dif = wp.vec3(
      qpos[worldid, qposid + 0] - qpos_spring[qposid + 0],
      qpos[worldid, qposid + 1] - qpos_spring[qposid + 1],
      qpos[worldid, qposid + 2] - qpos_spring[qposid + 2],
    )
    qfrc_spring[worldid, dofid + 0] = -stiffness * dif[0]
    qfrc_spring[worldid, dofid + 1] = -stiffness * dif[1]
    qfrc_spring[worldid, dofid + 2] = -stiffness * dif[2]
    rot = wp.quat(
      qpos[worldid, qposid + 3],
      qpos[worldid, qposid + 4],
      qpos[worldid, qposid + 5],
      qpos[worldid, qposid + 6],
    )
    ref = wp.quat(
      qpos_spring[qposid + 3],
      qpos_spring[qposid + 4],
      qpos_spring[qposid + 5],
      qpos_spring[qposid + 6],
    )
    dif = math.quat_sub(rot, ref)
    qfrc_spring[worldid, dofid + 3] = -stiffness * dif[0]
    qfrc_spring[worldid, dofid + 4] = -stiffness * dif[1]
    qfrc_spring[worldid, dofid + 5] = -stiffness * dif[2]
  elif jnttype == wp.static(JointType.BALL.value):
    rot = wp.quat(
      qpos[worldid, qposid + 0],
      qpos[worldid, qposid + 1],
      qpos[worldid, qposid + 2],
      qpos[worldid, qposid + 3],
    )
    ref = wp.quat(
      qpos_spring[qposid + 0],
      qpos_spring[qposid + 1],
      qpos_spring[qposid + 2],
      qpos_spring[qposid + 3],
    )
    dif = math.quat_sub(rot, ref)
    qfrc_spring[worldid, dofid + 0] = -stiffness * dif[0]
    qfrc_spring[worldid, dofid + 1] = -stiffness * dif[1]
    qfrc_spring[worldid, dofid + 2] = -stiffness * dif[2]
  else:  # mjJNT_SLIDE, mjJNT_HINGE
    fdif = qpos[worldid, qposid] - qpos_spring[qposid]
    qfrc_spring[worldid, dofid] = -stiffness * fdif


@wp.kernel
def _damper_passive(
  # Model
  dof_damping: wp.array(dtype=float),
  # Data in
  qfrc_spring: array2df,
  qvel: array2df,
  # Data out
  qfrc_damper: array2df,
  qfrc_passive: array2df,
):
  worldid, dofid = wp.tid()

  damper = -dof_damping[dofid] * qvel[worldid, dofid]

  qfrc_damper[worldid, dofid] = damper
  qfrc_passive[worldid, dofid] = damper + qfrc_spring[worldid, dofid]


@event_scope
def passive(m: Model, d: Data):
  """Adds all passive forces."""
  if m.opt.disableflags & DisableBit.PASSIVE:
    d.qfrc_passive.zero_()
    # TODO(team): qfrc_gravcomp
    return

  # TODO(team): mj_gravcomp
  # TODO(team): mj_ellipsoidFluidModel
  # TODO(team): mj_inertiaBoxFluidModell

  d.qfrc_spring.zero_()
  wp.launch(
    _spring_passive,
    dim=(d.nworld, m.njnt),
    inputs=[
      m.jnt_dofadr,
      m.jnt_qposadr,
      m.jnt_stiffness,
      m.jnt_type,
      m.qpos_spring,
      d.qpos,
    ],
    outputs=[d.qfrc_spring],
  )
  wp.launch(
    _damper_passive,
    dim=(d.nworld, m.nv),
    inputs=[m.dof_damping, d.qfrc_spring, d.qvel],
    outputs=[d.qfrc_damper, d.qfrc_passive],
  )

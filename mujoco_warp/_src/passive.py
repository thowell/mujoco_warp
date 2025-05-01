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
from .warp_util import event_scope


@wp.kernel
def _spring_passive(
  # Model:
  qpos_spring: wp.array(dtype=float),
  jnt_type: wp.array(dtype=int),
  jnt_qposadr: wp.array(dtype=int),
  jnt_dofadr: wp.array(dtype=int),
  jnt_stiffness: wp.array(dtype=float),
  # Data in:
  qpos_in: wp.array2d(dtype=float),
  # Data out:
  qfrc_spring_out: wp.array2d(dtype=float),
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
      qpos_in[worldid, qposid + 0] - qpos_spring[qposid + 0],
      qpos_in[worldid, qposid + 1] - qpos_spring[qposid + 1],
      qpos_in[worldid, qposid + 2] - qpos_spring[qposid + 2],
    )
    qfrc_spring_out[worldid, dofid + 0] = -stiffness * dif[0]
    qfrc_spring_out[worldid, dofid + 1] = -stiffness * dif[1]
    qfrc_spring_out[worldid, dofid + 2] = -stiffness * dif[2]
    rot = wp.quat(
      qpos_in[worldid, qposid + 3],
      qpos_in[worldid, qposid + 4],
      qpos_in[worldid, qposid + 5],
      qpos_in[worldid, qposid + 6],
    )
    ref = wp.quat(
      qpos_spring[qposid + 3],
      qpos_spring[qposid + 4],
      qpos_spring[qposid + 5],
      qpos_spring[qposid + 6],
    )
    dif = math.quat_sub(rot, ref)
    qfrc_spring_out[worldid, dofid + 3] = -stiffness * dif[0]
    qfrc_spring_out[worldid, dofid + 4] = -stiffness * dif[1]
    qfrc_spring_out[worldid, dofid + 5] = -stiffness * dif[2]
  elif jnttype == wp.static(JointType.BALL.value):
    rot = wp.quat(
      qpos_in[worldid, qposid + 0],
      qpos_in[worldid, qposid + 1],
      qpos_in[worldid, qposid + 2],
      qpos_in[worldid, qposid + 3],
    )
    ref = wp.quat(
      qpos_spring[qposid + 0],
      qpos_spring[qposid + 1],
      qpos_spring[qposid + 2],
      qpos_spring[qposid + 3],
    )
    dif = math.quat_sub(rot, ref)
    qfrc_spring_out[worldid, dofid + 0] = -stiffness * dif[0]
    qfrc_spring_out[worldid, dofid + 1] = -stiffness * dif[1]
    qfrc_spring_out[worldid, dofid + 2] = -stiffness * dif[2]
  else:  # mjJNT_SLIDE, mjJNT_HINGE
    fdif = qpos_in[worldid, qposid] - qpos_spring[qposid]
    qfrc_spring_out[worldid, dofid] = -stiffness * fdif


@wp.kernel
def _damper_passive(
  # Model:
  dof_damping: wp.array(dtype=float),
  # Data in:
  qvel_in: wp.array2d(dtype=float),
  qfrc_spring_in: wp.array2d(dtype=float),
  # Data out:
  qfrc_damper_out: wp.array2d(dtype=float),
  qfrc_passive_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()

  qfrc_damper = -dof_damping[dofid] * qvel_in[worldid, dofid]

  qfrc_damper_out[worldid, dofid] = qfrc_damper
  qfrc_passive_out[worldid, dofid] = qfrc_damper + qfrc_spring_in[worldid, dofid]


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
      m.qpos_spring,
      m.jnt_type,
      m.jnt_qposadr,
      m.jnt_dofadr,
      m.jnt_stiffness,
      d.qpos,
    ],
    outputs=[d.qfrc_spring],
  )
  wp.launch(
    _damper_passive,
    dim=(d.nworld, m.nv),
    inputs=[m.dof_damping, d.qvel, d.qfrc_spring],
    outputs=[d.qfrc_damper, d.qfrc_passive],
  )

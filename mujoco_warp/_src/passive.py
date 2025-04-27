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

from typing import Tuple

import warp as wp

from . import math
from . import support
from .types import Data
from .types import DisableBit
from .types import JointType
from .types import ObjType
from .types import Model
from .types import MJ_MINVAL
from .warp_util import event_scope
from .warp_util import kernel


@event_scope
def passive(m: Model, d: Data):
  """Adds all passive forces."""
  if m.opt.disableflags & DisableBit.PASSIVE:
    d.qfrc_passive.zero_()
    # TODO(team): qfrc_gravcomp
    return

  @kernel
  def _spring(m: Model, d: Data):
    worldid, jntid = wp.tid()
    stiffness = m.jnt_stiffness[jntid]
    dofid = m.jnt_dofadr[jntid]

    if stiffness == 0.0:
      return

    jnt_type = m.jnt_type[jntid]
    qposid = m.jnt_qposadr[jntid]

    if jnt_type == wp.static(JointType.FREE.value):
      dif = wp.vec3(
        d.qpos[worldid, qposid + 0] - m.qpos_spring[qposid + 0],
        d.qpos[worldid, qposid + 1] - m.qpos_spring[qposid + 1],
        d.qpos[worldid, qposid + 2] - m.qpos_spring[qposid + 2],
      )
      d.qfrc_spring[worldid, dofid + 0] = -stiffness * dif[0]
      d.qfrc_spring[worldid, dofid + 1] = -stiffness * dif[1]
      d.qfrc_spring[worldid, dofid + 2] = -stiffness * dif[2]
      rot = wp.quat(
        d.qpos[worldid, qposid + 3],
        d.qpos[worldid, qposid + 4],
        d.qpos[worldid, qposid + 5],
        d.qpos[worldid, qposid + 6],
      )
      ref = wp.quat(
        m.qpos_spring[qposid + 3],
        m.qpos_spring[qposid + 4],
        m.qpos_spring[qposid + 5],
        m.qpos_spring[qposid + 6],
      )
      dif = math.quat_sub(rot, ref)
      d.qfrc_spring[worldid, dofid + 3] = -stiffness * dif[0]
      d.qfrc_spring[worldid, dofid + 4] = -stiffness * dif[1]
      d.qfrc_spring[worldid, dofid + 5] = -stiffness * dif[2]
    elif jnt_type == wp.static(JointType.BALL.value):
      rot = wp.quat(
        d.qpos[worldid, qposid + 0],
        d.qpos[worldid, qposid + 1],
        d.qpos[worldid, qposid + 2],
        d.qpos[worldid, qposid + 3],
      )
      ref = wp.quat(
        m.qpos_spring[qposid + 0],
        m.qpos_spring[qposid + 1],
        m.qpos_spring[qposid + 2],
        m.qpos_spring[qposid + 3],
      )
      dif = math.quat_sub(rot, ref)
      d.qfrc_spring[worldid, dofid + 0] = -stiffness * dif[0]
      d.qfrc_spring[worldid, dofid + 1] = -stiffness * dif[1]
      d.qfrc_spring[worldid, dofid + 2] = -stiffness * dif[2]
    else:  # mjJNT_SLIDE, mjJNT_HINGE
      fdif = d.qpos[worldid, qposid] - m.qpos_spring[qposid]
      d.qfrc_spring[worldid, dofid] = -stiffness * fdif

  @kernel
  def _damper_passive(m: Model, d: Data):
    worldid, dofid = wp.tid()
    damping = m.dof_damping[dofid]
    qfrc_damper = -damping * d.qvel[worldid, dofid]

    d.qfrc_damper[worldid, dofid] = qfrc_damper
    d.qfrc_passive[worldid, dofid] = qfrc_damper + d.qfrc_spring[worldid, dofid]

  # TODO(team): mj_gravcomp

  d.qfrc_spring.zero_()
  wp.launch(_spring, dim=(d.nworld, m.njnt), inputs=[m, d])
  wp.launch(_damper_passive, dim=(d.nworld, m.nv), inputs=[m, d])

  if (
    m.opt.density or m.opt.viscosity or m.opt.wind[0] or m.opt.wind[1] or m.opt.wind[2]
  ):
    _fluid(m, d)


def _fluid(m: Model, d: Data):
  @kernel
  def _box_fluid(m: Model, d: Data):
    worldid, bodyid = wp.tid()
    force, torque = _inertia_box_fluid_model(m, d, worldid, bodyid)
    d.fluid_applied[worldid, bodyid] = wp.spatial_vector(force, torque)

  wp.launch(_box_fluid, dim=(d.nworld, m.nbody), inputs=[m, d])

  # TODO(team): ellipsoid fluid model

  support.apply_ft(m, d, d.fluid_applied, d.qfrc_fluid)

  @kernel
  def _qfrc_passive_fluid(m: Model, d: Data):
    worldid, dofid = wp.tid()
    d.qfrc_passive[worldid, dofid] += d.qfrc_fluid[worldid, dofid]

  wp.launch(_qfrc_passive_fluid, dim=(d.nworld, m.nv), inputs=[m, d])


@wp.func
def _inertia_box_fluid_model(
  m: Model, d: Data, worldid: int, bodyid: int
) -> Tuple[wp.vec3f, wp.vec3f]:
  """Fluid forces based on inertia-box approximation."""

  # map from CoM-centered to local body-centered 6D velocity

  # body-inertial
  pos = d.xipos[worldid, bodyid]
  rot = d.ximat[worldid, bodyid]
  rotT = wp.transpose(rot)

  # transform velocity
  cvel = d.cvel[worldid, bodyid]
  torque = wp.spatial_top(cvel)
  force = wp.spatial_bottom(cvel)
  subtree_com = d.subtree_com[worldid, m.body_rootid[bodyid]]
  dif = pos - subtree_com
  force -= wp.cross(dif, torque)

  lvel_torque = rotT @ torque
  lvel_force = rotT @ force

  if m.opt.wind[0] or m.opt.wind[1] or m.opt.wind[2]:
    # subtract translational component from body velocity
    lvel_force -= rotT @ m.opt.wind

  lfrc_torque = wp.vec3(0.0)
  lfrc_force = wp.vec3(0.0)

  has_viscosity = m.opt.viscosity > 0.0
  has_density = m.opt.density > 0.0

  if has_viscosity or has_density:
    inertia = m.body_inertia[bodyid]
    mass = m.body_mass[bodyid]
    scl = 6.0 / mass
    box0 = wp.sqrt(wp.max(MJ_MINVAL, inertia[1] + inertia[2] - inertia[0]) * scl)
    box1 = wp.sqrt(wp.max(MJ_MINVAL, inertia[0] + inertia[2] - inertia[1]) * scl)
    box2 = wp.sqrt(wp.max(MJ_MINVAL, inertia[0] + inertia[1] - inertia[2]) * scl)

  if has_viscosity:
    # diameter of sphere approximation
    diam = (box0 + box1 + box2) / 3.0

    # angular viscosity
    lfrc_torque = lvel_torque * -wp.pi * diam * diam * diam * m.opt.viscosity

    # linear viscosity
    lfrc_force = lvel_force * -3.0 * wp.pi * diam * m.opt.viscosity

  if has_density:
    # force
    lfrc_force -= wp.vec3(
      0.5 * m.opt.density * box1 * box2 * wp.abs(lvel_force[0]) * lvel_force[0],
      0.5 * m.opt.density * box0 * box2 * wp.abs(lvel_force[1]) * lvel_force[1],
      0.5 * m.opt.density * box0 * box1 * wp.abs(lvel_force[2]) * lvel_force[2],
    )

    # torque
    scl = m.opt.density / 64.0
    box0_pow4 = box0 * box0 * box0 * box0
    box1_pow4 = box1 * box1 * box1 * box1
    box2_pow4 = box2 * box2 * box2 * box2
    lfrc_torque -= wp.vec3(
      box0 * (box1_pow4 + box2_pow4) * wp.abs(lvel_torque[0]) * lvel_torque[0] * scl,
      box1 * (box0_pow4 + box2_pow4) * wp.abs(lvel_torque[1]) * lvel_torque[1] * scl,
      box2 * (box0_pow4 + box1_pow4) * wp.abs(lvel_torque[2]) * lvel_torque[2] * scl,
    )

  # rotate to global orientation: lfrc -> bfrc
  bfrc_torque = rot @ lfrc_torque
  bfrc_force = rot @ lfrc_force

  # apply force and torque to body com
  return bfrc_force, bfrc_torque

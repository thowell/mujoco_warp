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

from mujoco_warp._src import math
from mujoco_warp._src import support
from mujoco_warp._src import util_misc
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import Model
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.func
def _pow2(val: float) -> float:
  return val * val


@wp.func
def _pow4(val: float) -> float:
  sq = val * val
  return sq * sq


@wp.func
def _geom_semiaxes(size: wp.vec3, geom_type: int) -> wp.vec3:  # kernel_analyzer: ignore
  if geom_type == GeomType.SPHERE:
    r = size[0]
    return wp.vec3(r, r, r)

  if geom_type == GeomType.CAPSULE:
    radius = size[0]
    half_length = size[1]
    return wp.vec3(radius, radius, half_length + radius)

  if geom_type == GeomType.CYLINDER:
    radius = size[0]
    half_length = size[1]
    return wp.vec3(radius, radius, half_length)

  # ellipsoid, box, mesh, sdf -> use size directly
  return size


@wp.func
def _ellipsoid_max_moment(size: wp.vec3, dir: int) -> float:
  d0 = size[dir]
  d1 = size[(dir + 1) % 3]
  d2 = size[(dir + 2) % 3]
  return wp.static(8.0 / 15.0 * wp.pi) * d0 * _pow4(wp.max(d1, d2))


@wp.kernel
def _spring_damper_dof_passive(
  # Model:
  opt_disableflags: int,
  qpos_spring: wp.array2d[float],
  jnt_type: wp.array[int],
  jnt_qposadr: wp.array[int],
  jnt_dofadr: wp.array[int],
  jnt_stiffness: wp.array2d[float],
  jnt_stiffnesspoly: wp.array2d[wp.vec2],
  dof_damping: wp.array2d[float],
  dof_dampingpoly: wp.array2d[wp.vec2],
  # Data in:
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  # Data out:
  qfrc_spring_out: wp.array2d[float],
  qfrc_damper_out: wp.array2d[float],
):
  worldid, jntid = wp.tid()
  dofid = jnt_dofadr[jntid]
  jnttype = jnt_type[jntid]
  stiffness = jnt_stiffness[worldid % jnt_stiffness.shape[0], jntid]
  spoly = jnt_stiffnesspoly[worldid % jnt_stiffnesspoly.shape[0], jntid]
  damping = dof_damping[worldid % dof_damping.shape[0], dofid]
  dpoly = dof_dampingpoly[worldid % dof_dampingpoly.shape[0], dofid]

  has_stiffness = (stiffness != 0.0 or spoly[0] != 0.0 or spoly[1] != 0.0) and not (opt_disableflags & DisableBit.SPRING)
  has_damping = (damping != 0.0 or dpoly[0] != 0.0 or dpoly[1] != 0.0) and not (opt_disableflags & DisableBit.DAMPER)

  if not has_stiffness:
    if jnttype == JointType.FREE:
      for i in range(6):
        qfrc_spring_out[worldid, dofid + i] = 0.0
    elif jnttype == JointType.BALL:
      for i in range(3):
        qfrc_spring_out[worldid, dofid + i] = 0.0
    else:
      qfrc_spring_out[worldid, dofid] = 0.0

  if not has_damping:
    if jnttype == JointType.FREE:
      for i in range(6):
        qfrc_damper_out[worldid, dofid + i] = 0.0
    elif jnttype == JointType.BALL:
      for i in range(3):
        qfrc_damper_out[worldid, dofid + i] = 0.0
    else:
      qfrc_damper_out[worldid, dofid] = 0.0

  if not (has_stiffness or has_damping):
    return
  qposid = jnt_qposadr[jntid]
  qpos_spring_id = worldid % qpos_spring.shape[0]

  if jnttype == JointType.FREE:
    # spring
    if has_stiffness:
      dif = wp.vec3(
        qpos_in[worldid, qposid + 0] - qpos_spring[qpos_spring_id, qposid + 0],
        qpos_in[worldid, qposid + 1] - qpos_spring[qpos_spring_id, qposid + 1],
        qpos_in[worldid, qposid + 2] - qpos_spring[qpos_spring_id, qposid + 2],
      )
      r = wp.length(dif)
      k = util_misc._poly_force(stiffness, spoly, r, 0)
      qfrc_spring_out[worldid, dofid + 0] = -k * dif[0]
      qfrc_spring_out[worldid, dofid + 1] = -k * dif[1]
      qfrc_spring_out[worldid, dofid + 2] = -k * dif[2]

      rot = wp.quat(
        qpos_in[worldid, qposid + 3],
        qpos_in[worldid, qposid + 4],
        qpos_in[worldid, qposid + 5],
        qpos_in[worldid, qposid + 6],
      )
      rot = wp.normalize(rot)
      ref = wp.quat(
        qpos_spring[qpos_spring_id, qposid + 3],
        qpos_spring[qpos_spring_id, qposid + 4],
        qpos_spring[qpos_spring_id, qposid + 5],
        qpos_spring[qpos_spring_id, qposid + 6],
      )
      dif = math.quat_sub(rot, ref)
      r_rot = wp.length(dif)
      k_rot = util_misc._poly_force(stiffness, spoly, r_rot, 0)
      qfrc_spring_out[worldid, dofid + 3] = -k_rot * dif[0]
      qfrc_spring_out[worldid, dofid + 4] = -k_rot * dif[1]
      qfrc_spring_out[worldid, dofid + 5] = -k_rot * dif[2]

    # damper
    if has_damping:
      for i in range(6):
        v = qvel_in[worldid, dofid + i]
        qfrc_damper_out[worldid, dofid + i] = -v * util_misc._poly_force(damping, dpoly, v, 1)

  elif jnttype == JointType.BALL:
    # spring
    if has_stiffness:
      rot = wp.quat(
        qpos_in[worldid, qposid + 0],
        qpos_in[worldid, qposid + 1],
        qpos_in[worldid, qposid + 2],
        qpos_in[worldid, qposid + 3],
      )
      rot = wp.normalize(rot)
      ref = wp.quat(
        qpos_spring[qpos_spring_id, qposid + 0],
        qpos_spring[qpos_spring_id, qposid + 1],
        qpos_spring[qpos_spring_id, qposid + 2],
        qpos_spring[qpos_spring_id, qposid + 3],
      )
      dif = math.quat_sub(rot, ref)
      r = wp.length(dif)
      k = util_misc._poly_force(stiffness, spoly, r, 0)
      qfrc_spring_out[worldid, dofid + 0] = -k * dif[0]
      qfrc_spring_out[worldid, dofid + 1] = -k * dif[1]
      qfrc_spring_out[worldid, dofid + 2] = -k * dif[2]

    # damper
    if has_damping:
      for i in range(3):
        v = qvel_in[worldid, dofid + i]
        qfrc_damper_out[worldid, dofid + i] = -v * util_misc._poly_force(damping, dpoly, v, 1)

  else:  # mjJNT_SLIDE, mjJNT_HINGE
    # spring
    if has_stiffness:
      fdif = qpos_in[worldid, qposid] - qpos_spring[qpos_spring_id, qposid]
      qfrc_spring_out[worldid, dofid] = -fdif * util_misc._poly_force(stiffness, spoly, fdif, 0)

    # damper
    if has_damping:
      v = qvel_in[worldid, dofid]
      qfrc_damper_out[worldid, dofid] = -v * util_misc._poly_force(damping, dpoly, v, 1)


@wp.kernel
def _spring_damper_tendon_passive(
  # Model:
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  tendon_stiffness: wp.array2d[float],
  tendon_stiffnesspoly: wp.array2d[wp.vec2],
  tendon_damping: wp.array2d[float],
  tendon_dampingpoly: wp.array2d[wp.vec2],
  tendon_lengthspring: wp.array2d[wp.vec2],
  # Data in:
  ten_J_in: wp.array2d[float],
  ten_length_in: wp.array2d[float],
  ten_velocity_in: wp.array2d[float],
  # In:
  dsbl_spring: bool,
  dsbl_damper: bool,
  # Data out:
  qfrc_spring_out: wp.array2d[float],
  qfrc_damper_out: wp.array2d[float],
):
  worldid, tenid, dofid_sparse = wp.tid()

  stiffness = tendon_stiffness[worldid % tendon_stiffness.shape[0], tenid]
  spoly = tendon_stiffnesspoly[worldid % tendon_stiffnesspoly.shape[0], tenid]
  damping = tendon_damping[worldid % tendon_damping.shape[0], tenid]
  dpoly = tendon_dampingpoly[worldid % tendon_dampingpoly.shape[0], tenid]

  has_stiffness = (stiffness != 0.0 or spoly[0] != 0.0 or spoly[1] != 0.0) and not dsbl_spring
  has_damping = (damping != 0.0 or dpoly[0] != 0.0 or dpoly[1] != 0.0) and not dsbl_damper

  if not has_stiffness and not has_damping:
    return

  rownnz = ten_J_rownnz[tenid]
  if dofid_sparse >= rownnz:
    return
  rowadr = ten_J_rowadr[tenid]
  sparseid = rowadr + dofid_sparse
  J = ten_J_in[worldid, sparseid]
  dofid = ten_J_colind[sparseid]

  if has_stiffness:
    # compute spring force along tendon
    length = ten_length_in[worldid, tenid]
    lengthspring = tendon_lengthspring[worldid % tendon_lengthspring.shape[0], tenid]
    lower = lengthspring[0]
    upper = lengthspring[1]

    x = wp.where(length > upper, length - upper, wp.where(length < lower, length - lower, 0.0))
    frc_spring = -x * util_misc._poly_force(stiffness, spoly, x, 0)

    # transform to joint torque
    wp.atomic_add(qfrc_spring_out[worldid], dofid, J * frc_spring)

  if has_damping:
    # compute damper force along tendon
    v = ten_velocity_in[worldid, tenid]
    frc_damper = -v * util_misc._poly_force(damping, dpoly, v, 1)

    # transform to joint torque
    wp.atomic_add(qfrc_damper_out[worldid], dofid, J * frc_damper)


@wp.kernel
def _gravity_force(
  # Model:
  opt_gravity: wp.array[wp.vec3],
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  body_mass: wp.array2d[float],
  body_gravcomp: wp.array2d[float],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # Data out:
  qfrc_gravcomp_out: wp.array2d[float],
):
  worldid, bodyid, dofid = wp.tid()
  bodyid += 1  # skip world body
  gravcomp = body_gravcomp[worldid % body_gravcomp.shape[0], bodyid]
  gravity = opt_gravity[worldid % opt_gravity.shape[0]]

  if gravcomp:
    force = -gravity * body_mass[worldid % body_mass.shape[0], bodyid] * gravcomp
    pos = xipos_in[worldid, bodyid]
    jac, _ = support.jac_dof(
      body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in, pos, bodyid, dofid, worldid
    )

    wp.atomic_add(qfrc_gravcomp_out[worldid], dofid, wp.dot(jac, force))


@wp.kernel
def _fluid_force(
  # Model:
  opt_wind: wp.array[wp.vec3],
  opt_density: wp.array[float],
  opt_viscosity: wp.array[float],
  body_rootid: wp.array[int],
  body_geomnum: wp.array[int],
  body_geomadr: wp.array[int],
  body_mass: wp.array2d[float],
  body_inertia: wp.array2d[wp.vec3],
  geom_type: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  geom_fluid: wp.array2d[float],
  body_fluid_ellipsoid: wp.array[bool],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  ximat_in: wp.array2d[wp.mat33],
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cvel_in: wp.array2d[wp.spatial_vector],
  # Out:
  fluid_applied_out: wp.array2d[wp.spatial_vector],
):
  """Computes body-space fluid forces for both inertia-box and ellipsoid models."""
  worldid, bodyid = wp.tid()
  zero_force = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))

  if bodyid == 0:
    fluid_applied_out[worldid, bodyid] = zero_force
    return

  # skip bodies with negligible mass
  mass = body_mass[worldid % body_mass.shape[0], bodyid]
  if mass < MJ_MINVAL:
    fluid_applied_out[worldid, bodyid] = zero_force
    return

  wind = opt_wind[worldid % opt_wind.shape[0]]
  density = opt_density[worldid % opt_density.shape[0]]
  viscosity = opt_viscosity[worldid % opt_viscosity.shape[0]]

  # Body kinematics
  xipos = xipos_in[worldid, bodyid]
  rot = ximat_in[worldid, bodyid]
  rotT = wp.transpose(rot)
  cvel = cvel_in[worldid, bodyid]
  ang_global = wp.spatial_top(cvel)
  lin_global = wp.spatial_bottom(cvel)
  subtree_root = subtree_com_in[worldid, body_rootid[bodyid]]
  lin_com = lin_global - wp.cross(xipos - subtree_root, ang_global)

  if body_fluid_ellipsoid[bodyid]:
    force_global = wp.vec3(0.0)
    torque_global = wp.vec3(0.0)

    start = body_geomadr[bodyid]
    count = body_geomnum[bodyid]

    for i in range(count):
      geomid = start + i
      coef = geom_fluid[geomid, 0]
      if coef <= 0.0:
        continue

      size = geom_size[worldid % geom_size.shape[0], geomid]
      semiaxes = _geom_semiaxes(size, geom_type[geomid])
      geom_rot = geom_xmat_in[worldid, geomid]
      geom_rotT = wp.transpose(geom_rot)
      geom_pos = geom_xpos_in[worldid, geomid]

      lin_point = lin_com + wp.cross(ang_global, geom_pos - xipos)

      l_ang = geom_rotT @ ang_global
      l_lin = geom_rotT @ lin_point

      if wind[0] or wind[1] or wind[2]:
        l_lin -= geom_rotT @ wind

      lfrc_torque = wp.vec3(0.0)
      lfrc_force = wp.vec3(0.0)

      if density > 0.0:
        # added-mass forces and torques
        virtual_mass = wp.vec3(geom_fluid[geomid, 6], geom_fluid[geomid, 7], geom_fluid[geomid, 8])
        virtual_inertia = wp.vec3(geom_fluid[geomid, 9], geom_fluid[geomid, 10], geom_fluid[geomid, 11])

        virtual_lin_mom = wp.vec3(
          density * virtual_mass[0] * l_lin[0],
          density * virtual_mass[1] * l_lin[1],
          density * virtual_mass[2] * l_lin[2],
        )
        virtual_ang_mom = wp.vec3(
          density * virtual_inertia[0] * l_ang[0],
          density * virtual_inertia[1] * l_ang[1],
          density * virtual_inertia[2] * l_ang[2],
        )

        added_mass_force = wp.cross(virtual_lin_mom, l_ang)
        added_mass_torque = wp.cross(virtual_lin_mom, l_lin) + wp.cross(virtual_ang_mom, l_ang)

        lfrc_force += added_mass_force
        lfrc_torque += added_mass_torque

      # lift force orthogonal to velocity from Kutta-Joukowski theorem
      magnus_coef = geom_fluid[geomid, 5]
      kutta_coef = geom_fluid[geomid, 4]
      blunt_drag_coef = geom_fluid[geomid, 1]
      slender_drag_coef = geom_fluid[geomid, 2]
      ang_drag_coef = geom_fluid[geomid, 3]

      volume = wp.static(4.0 / 3.0 * wp.pi) * semiaxes[0] * semiaxes[1] * semiaxes[2]
      d_max = wp.max(wp.max(semiaxes[0], semiaxes[1]), semiaxes[2])
      d_min = wp.min(wp.min(semiaxes[0], semiaxes[1]), semiaxes[2])
      d_mid = semiaxes[0] + semiaxes[1] + semiaxes[2] - d_max - d_min
      A_max = wp.pi * d_max * d_mid

      lin_speed = wp.length(l_lin)

      magnus_force = wp.cross(l_ang, l_lin) * (magnus_coef * density * volume)

      s12 = semiaxes[1] * semiaxes[2]
      s20 = semiaxes[2] * semiaxes[0]
      s01 = semiaxes[0] * semiaxes[1]

      proj_denom = _pow4(s12) * _pow2(l_lin[0]) + _pow4(s20) * _pow2(l_lin[1]) + _pow4(s01) * _pow2(l_lin[2])
      proj_num = _pow2(s12 * l_lin[0]) + _pow2(s20 * l_lin[1]) + _pow2(s01 * l_lin[2])

      A_proj = 0.0
      cos_alpha = 0.0
      if proj_num > MJ_MINVAL and proj_denom > MJ_MINVAL:
        A_proj = wp.pi * wp.sqrt(proj_denom / wp.max(MJ_MINVAL, proj_num))
        if lin_speed > MJ_MINVAL:
          cos_alpha = proj_num / wp.max(MJ_MINVAL, lin_speed * proj_denom)

      norm = wp.vec3(
        _pow2(s12) * l_lin[0],
        _pow2(s20) * l_lin[1],
        _pow2(s01) * l_lin[2],
      )

      kutta_force = wp.vec3(0.0)
      if density > 0.0 and kutta_coef != 0.0 and lin_speed > MJ_MINVAL:
        kutta_circ = wp.cross(norm, l_lin) * (kutta_coef * density * cos_alpha * A_proj)
        kutta_force = wp.cross(kutta_circ, l_lin)

      eq_sphere_D = wp.static(2.0 / 3.0) * (semiaxes[0] + semiaxes[1] + semiaxes[2])
      lin_visc_force_coef = wp.static(3.0 * wp.pi) * eq_sphere_D
      lin_visc_torq_coef = wp.pi * eq_sphere_D * eq_sphere_D * eq_sphere_D

      I_max = wp.static(8.0 / 15.0 * wp.pi) * d_mid * _pow4(d_max)
      II0 = _ellipsoid_max_moment(semiaxes, 0)
      II1 = _ellipsoid_max_moment(semiaxes, 1)
      II2 = _ellipsoid_max_moment(semiaxes, 2)

      mom_visc = wp.vec3(
        l_ang[0] * (ang_drag_coef * II0 + slender_drag_coef * (I_max - II0)),
        l_ang[1] * (ang_drag_coef * II1 + slender_drag_coef * (I_max - II1)),
        l_ang[2] * (ang_drag_coef * II2 + slender_drag_coef * (I_max - II2)),
      )

      drag_lin_coef = viscosity * lin_visc_force_coef + density * lin_speed * (
        A_proj * blunt_drag_coef + slender_drag_coef * (A_max - A_proj)
      )
      drag_ang_coef = viscosity * lin_visc_torq_coef + density * wp.length(mom_visc)

      lfrc_torque -= drag_ang_coef * l_ang
      lfrc_force += magnus_force + kutta_force - drag_lin_coef * l_lin

      lfrc_torque *= coef
      lfrc_force *= coef

      # map force/torque from local to world frame: lfrc -> bfrc
      torque_global += geom_rot @ lfrc_torque
      force_global += geom_rot @ lfrc_force

    fluid_applied_out[worldid, bodyid] = wp.spatial_vector(force_global, torque_global)
    return

  l_ang = rotT @ ang_global
  l_lin = rotT @ lin_com

  if wind[0] or wind[1] or wind[2]:
    l_lin -= rotT @ wind

  lfrc_torque = wp.vec3(0.0)
  lfrc_force = wp.vec3(0.0)

  has_viscosity = viscosity > 0.0
  has_density = density > 0.0

  if has_viscosity or has_density:
    inertia = body_inertia[worldid % body_inertia.shape[0], bodyid]
    mass = body_mass[worldid % body_mass.shape[0], bodyid]
    scl = 6.0 / mass
    box0 = wp.sqrt(wp.max(MJ_MINVAL, inertia[1] + inertia[2] - inertia[0]) * scl)
    box1 = wp.sqrt(wp.max(MJ_MINVAL, inertia[0] + inertia[2] - inertia[1]) * scl)
    box2 = wp.sqrt(wp.max(MJ_MINVAL, inertia[0] + inertia[1] - inertia[2]) * scl)

  if has_viscosity:
    diam = (box0 + box1 + box2) / 3.0
    lfrc_torque = -l_ang * wp.pow(diam, 3.0) * wp.pi * viscosity
    lfrc_force = -3.0 * l_lin * diam * wp.pi * viscosity

  if has_density:
    lfrc_force -= wp.vec3(
      0.5 * density * box1 * box2 * wp.abs(l_lin[0]) * l_lin[0],
      0.5 * density * box0 * box2 * wp.abs(l_lin[1]) * l_lin[1],
      0.5 * density * box0 * box1 * wp.abs(l_lin[2]) * l_lin[2],
    )

    scl = density / 64.0
    box0_pow4 = wp.pow(box0, 4.0)
    box1_pow4 = wp.pow(box1, 4.0)
    box2_pow4 = wp.pow(box2, 4.0)
    lfrc_torque -= wp.vec3(
      box0 * (box1_pow4 + box2_pow4) * wp.abs(l_ang[0]) * l_ang[0] * scl,
      box1 * (box0_pow4 + box2_pow4) * wp.abs(l_ang[1]) * l_ang[1] * scl,
      box2 * (box0_pow4 + box1_pow4) * wp.abs(l_ang[2]) * l_ang[2] * scl,
    )

  torque_global = rot @ lfrc_torque
  force_global = rot @ lfrc_force

  fluid_applied_out[worldid, bodyid] = wp.spatial_vector(force_global, torque_global)


def _fluid(m: Model, d: Data):
  fluid_applied = wp.empty((d.nworld, m.nbody), dtype=wp.spatial_vector)

  wp.launch(
    _fluid_force,
    dim=(d.nworld, m.nbody),
    inputs=[
      m.opt.wind,
      m.opt.density,
      m.opt.viscosity,
      m.body_rootid,
      m.body_geomnum,
      m.body_geomadr,
      m.body_mass,
      m.body_inertia,
      m.geom_type,
      m.geom_size,
      m.geom_fluid,
      m.body_fluid_ellipsoid,
      d.xipos,
      d.ximat,
      d.geom_xpos,
      d.geom_xmat,
      d.subtree_com,
      d.cvel,
    ],
    outputs=[fluid_applied],
  )

  support.apply_ft(m, d, fluid_applied, d.qfrc_fluid, False)


@wp.kernel
def _qfrc_passive(
  # Model:
  jnt_actgravcomp: wp.array[int],
  dof_jntid: wp.array[int],
  has_fluid: bool,
  # Data in:
  qfrc_spring_in: wp.array2d[float],
  qfrc_damper_in: wp.array2d[float],
  qfrc_gravcomp_in: wp.array2d[float],
  qfrc_fluid_in: wp.array2d[float],
  # In:
  gravcomp: bool,
  # Data out:
  qfrc_passive_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  qfrc_passive = qfrc_spring_in[worldid, dofid]
  qfrc_passive += qfrc_damper_in[worldid, dofid]

  # add gravcomp unless added by actuators
  if gravcomp and not jnt_actgravcomp[dof_jntid[dofid]]:
    qfrc_passive += qfrc_gravcomp_in[worldid, dofid]

  # add fluid force
  if has_fluid:
    qfrc_passive += qfrc_fluid_in[worldid, dofid]

  qfrc_passive_out[worldid, dofid] = qfrc_passive


@wp.kernel
def _flex_elasticity(
  # Model:
  nflex: int,
  opt_timestep: wp.array[float],
  body_dofadr: wp.array[int],
  flex_dim: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_edgeadr: wp.array[int],
  flex_elemadr: wp.array[int],
  flex_elemnum: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_stiffnessadr: wp.array[int],
  flex_elemedgeadr: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_elem: wp.array[int],
  flex_elemedge: wp.array[int],
  flexedge_length0: wp.array[float],
  flex_stiffness: wp.array[float],
  flex_damping: wp.array[float],
  # Data in:
  flexvert_xpos_in: wp.array2d[wp.vec3],
  flexedge_length_in: wp.array2d[float],
  flexedge_velocity_in: wp.array2d[float],
  # In:
  dsbl_damper: bool,
  # Data out:
  qfrc_spring_out: wp.array2d[float],
):
  worldid, elemid = wp.tid()
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]

  for i in range(nflex):
    locid = elemid - flex_elemadr[i]
    if locid >= 0 and locid < flex_elemnum[i]:
      f = i
      break

  local_elemid = elemid - flex_elemadr[f]
  dim = flex_dim[f]
  nvert = dim + 1
  nedge = nvert * (nvert - 1) / 2
  edges = wp.where(
    dim == 1,
    wp.matrix(0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, shape=(6, 2), dtype=int),
    wp.where(
      dim == 3,
      wp.matrix(0, 1, 1, 2, 2, 0, 2, 3, 0, 3, 1, 3, shape=(6, 2), dtype=int),
      wp.matrix(1, 2, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, shape=(6, 2), dtype=int),
    ),
  )
  if timestep > 0.0 and not dsbl_damper:
    kD = flex_damping[f] / timestep
  else:
    kD = 0.0

  elem_data_adr = flex_elemdataadr[f] + local_elemid * (dim + 1)
  vbase = flex_vertadr[f]

  # skip trilinear/interp elements (vertbodyid == -1, no simplex stiffness)
  vert0_check = flex_elem[elem_data_adr]
  if flex_vertbodyid[vbase + vert0_check] < 0:
    return
  gradient = wp.matrix(0.0, shape=(6, 6))
  for e in range(nedge):
    vert0 = flex_elem[elem_data_adr + edges[e, 0]]
    vert1 = flex_elem[elem_data_adr + edges[e, 1]]
    xpos0 = flexvert_xpos_in[worldid, vbase + vert0]
    xpos1 = flexvert_xpos_in[worldid, vbase + vert1]
    for i in range(3):
      gradient[e, 0 + i] = xpos0[i] - xpos1[i]
      gradient[e, 3 + i] = xpos1[i] - xpos0[i]

  elongation = wp.spatial_vectorf(0.0)
  for e in range(nedge):
    idx = flex_elemedge[flex_elemedgeadr[f] + local_elemid * nedge + e]
    vel = flexedge_velocity_in[worldid, flex_edgeadr[f] + idx]
    deformed = flexedge_length_in[worldid, flex_edgeadr[f] + idx]
    reference = flexedge_length0[flex_edgeadr[f] + idx]
    previous = deformed - vel * timestep
    elongation[e] = deformed * deformed - reference * reference + (deformed * deformed - previous * previous) * kD

  metric = wp.matrix(0.0, shape=(6, 6))
  stiffness_size = nedge * (nedge + 1) / 2
  stiffness_adr = flex_stiffnessadr[f] + local_elemid * stiffness_size
  id = int(0)
  for ed1 in range(nedge):
    for ed2 in range(ed1, nedge):
      metric[ed1, ed2] = flex_stiffness[stiffness_adr + id]
      metric[ed2, ed1] = flex_stiffness[stiffness_adr + id]
      id += 1

  force = wp.matrix(0.0, shape=(6, 3))
  for ed1 in range(nedge):
    for ed2 in range(nedge):
      for i in range(2):
        for x in range(3):
          force[edges[ed2, i], x] -= elongation[ed1] * gradient[ed2, 3 * i + x] * metric[ed1, ed2]

  for v in range(nvert):
    vert = flex_elem[elem_data_adr + v]
    bodyid = flex_vertbodyid[flex_vertadr[f] + vert]
    for x in range(3):
      wp.atomic_add(qfrc_spring_out, worldid, body_dofadr[bodyid] + x, force[v, x])


@wp.kernel
def _flex_bending(
  # Model:
  nflex: int,
  body_dofadr: wp.array[int],
  flex_dim: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_edgeadr: wp.array[int],
  flex_edgenum: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_edge: wp.array[wp.vec2i],
  flex_edgeflap: wp.array[wp.vec2i],
  flex_bending: wp.array2d[float],
  # Data in:
  flexvert_xpos_in: wp.array2d[wp.vec3],
  # Data out:
  qfrc_spring_out: wp.array2d[float],
):
  worldid, edgeid = wp.tid()
  nvert = 4

  for i in range(nflex):
    locid = edgeid - flex_edgeadr[i]
    if locid >= 0 and locid < flex_edgenum[i]:
      f = i
      break

  if flex_dim[f] != 2:
    return

  if flex_edgeflap[edgeid][1] == -1:
    return

  v = wp.vec4i(
    flex_vertadr[f] + flex_edge[edgeid][0],
    flex_vertadr[f] + flex_edge[edgeid][1],
    flex_vertadr[f] + flex_edgeflap[edgeid][0],
    flex_vertadr[f] + flex_edgeflap[edgeid][1],
  )

  frc = wp.matrix(0.0, shape=(4, 3))
  if flex_bending[edgeid, 16]:
    v0 = flexvert_xpos_in[worldid, v[0]]
    v1 = flexvert_xpos_in[worldid, v[1]]
    v2 = flexvert_xpos_in[worldid, v[2]]
    v3 = flexvert_xpos_in[worldid, v[3]]
    frc[1] = wp.cross(v2 - v0, v3 - v0)
    frc[2] = wp.cross(v3 - v0, v1 - v0)
    frc[3] = wp.cross(v1 - v0, v2 - v0)
    frc[0] = -(frc[1] + frc[2] + frc[3])

  force = wp.matrix(0.0, shape=(nvert, 3))
  for i in range(nvert):
    for x in range(3):
      acc = float(0.0)
      for j in range(nvert):
        acc += flex_bending[edgeid, 4 * i + j] * flexvert_xpos_in[worldid, v[j]][x]
      force[i, x] = -(acc + flex_bending[edgeid, 16] * frc[i, x])

  for i in range(nvert):
    bodyid = flex_vertbodyid[v[i]]
    for x in range(3):
      wp.atomic_add(qfrc_spring_out, worldid, body_dofadr[bodyid] + x, force[i, x])


@wp.kernel
def _flex_passive_interp(
  # Model:
  nflex: int,
  body_rootid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  flex_interp: wp.array[int],
  flex_cellnum: wp.array[wp.vec3i],
  flex_nodeadr: wp.array[int],
  flex_stiffnessadr: wp.array[int],
  flex_nodebodyid: wp.array[int],
  flex_node: wp.array[wp.vec3],
  flex_node0: wp.array[wp.vec3],
  flex_stiffness: wp.array[float],
  flex_damping: wp.array[float],
  flex_edgeequality: wp.array[int],
  flex_centered: wp.array[bool],
  # Data in:
  subtree_com_in: wp.array2d[wp.vec3],
  cvel_in: wp.array2d[wp.spatial_vector],
  flexnode_xpos_in: wp.array2d[wp.vec3],
  # In:
  dsbl_spring: bool,
  dsbl_damper: bool,
  # Data out:
  qfrc_spring_out: wp.array2d[float],
  qfrc_damper_out: wp.array2d[float],
  # Out:
  displ_scratch_out: wp.array3d[wp.vec3],
  vel_corot_scratch_out: wp.array3d[wp.vec3],
):
  """Corotational passive forces for interpolated flex (trilinear/quadratic)."""
  worldid, cellid = wp.tid()

  # Find which flex and which cell this thread handles
  cell_offset = int(0)
  f = int(0)
  ci = int(0)
  cj = int(0)
  ck = int(0)
  found = int(0)
  for fi in range(nflex):
    if found:
      break
    order_fi = flex_interp[fi]
    if order_fi < 0:
      order_fi = -order_fi
    if order_fi == 0:
      continue
    # skip flex with strain constraints (stiffness in constraint solver)
    if flex_edgeequality[fi] == 3:
      continue
    cellnum_fi = flex_cellnum[fi]
    cx_fi = cellnum_fi[0]
    cy_fi = cellnum_fi[1]
    cz_fi = cellnum_fi[2]
    ncells_fi = cx_fi * cy_fi * cz_fi
    if cellid < cell_offset + ncells_fi:
      f = fi
      local_cell = cellid - cell_offset
      ci = local_cell / (cy_fi * cz_fi)
      cj = (local_cell - ci * cy_fi * cz_fi) / cz_fi
      ck = local_cell - ci * cy_fi * cz_fi - cj * cz_fi
      found = 1
    cell_offset += ncells_fi

  if not found:
    return

  order = flex_interp[f]
  if order < 0:
    order = -order

  npc = (order + 1) * (order + 1) * (order + 1)
  ndof_cell = 3 * npc

  cellnum = flex_cellnum[f]
  cy = cellnum[1]
  cz = cellnum[2]
  nstart = flex_nodeadr[f]
  ny_g = cy * order + 1
  nz_g = cz * order + 1

  # Cell stiffness matrix address
  cell_idx = ci * cy * cz + cj * cz + ck
  k_base = flex_stiffnessadr[f] + cell_idx * ndof_cell * ndof_cell

  # Skip empty cells (zero stiffness)
  if flex_stiffness[k_base] == 0.0:
    return

  # Compute corotational frame via polar decomposition at cell center
  F = wp.mat33(0.0)
  idx = int(0)
  for li in range(order + 1):
    for lj in range(order + 1):
      for lk in range(order + 1):
        if idx < npc:
          gi = ci * order + li
          gj = cj * order + lj
          gk = ck * order + lk
          gidx = gi * ny_g * nz_g + gj * nz_g + gk

          node_pos = flexnode_xpos_in[worldid, nstart + gidx]

          # Shape function derivatives at center (0.5)
          if order == 1:
            dphi_x = float(-1) if li == 0 else float(1)
            dphi_y = float(-1) if lj == 0 else float(1)
            dphi_z = float(-1) if lk == 0 else float(1)
            phi_x = float(0.5)
            phi_y = float(0.5)
            phi_z = float(0.5)
          else:
            if li == 0:
              dphi_x = -1.0
            elif li == 1:
              dphi_x = 0.0
            else:
              dphi_x = 1.0
            if lj == 0:
              dphi_y = -1.0
            elif lj == 1:
              dphi_y = 0.0
            else:
              dphi_y = 1.0
            if lk == 0:
              dphi_z = -1.0
            elif lk == 1:
              dphi_z = 0.0
            else:
              dphi_z = 1.0
            phi_x = 0.5 if li == 0 or li == 2 else 1.0
            phi_y = 0.5 if lj == 0 or lj == 2 else 1.0
            phi_z = 0.5 if lk == 0 or lk == 2 else 1.0

          grad_x = dphi_x * phi_y * phi_z
          grad_y = phi_x * dphi_y * phi_z
          grad_z = phi_x * phi_y * dphi_z

          for r in range(3):
            F[r, 0] += node_pos[r] * grad_x
            F[r, 1] += node_pos[r] * grad_y
            F[r, 2] += node_pos[r] * grad_z

          idx += 1

  # Extract rotation via Müller et al. (2016) quaternion iteration (mju_mat2Rot)
  cell_quat = wp.quat(0.0, 0.0, 0.0, 1.0)  # identity quaternion (x,y,z,w)
  for _iter in range(10):
    rot = wp.quat_to_matrix(cell_quat)

    # columns of rot and F
    col1_rot = wp.vec3(rot[0, 0], rot[1, 0], rot[2, 0])
    col2_rot = wp.vec3(rot[0, 1], rot[1, 1], rot[2, 1])
    col3_rot = wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])
    col1_mat = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    col2_mat = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    col3_mat = wp.vec3(F[0, 2], F[1, 2], F[2, 2])

    # omega = (cross(r1, m1) + cross(r2, m2) + cross(r3, m3)) / (|dot| + eps)
    omega = wp.cross(col1_rot, col1_mat) + wp.cross(col2_rot, col2_mat) + wp.cross(col3_rot, col3_mat)
    denom = wp.abs(wp.dot(col1_rot, col1_mat) + wp.dot(col2_rot, col2_mat) + wp.dot(col3_rot, col3_mat)) + 1.0e-10
    omega = omega / denom

    w = wp.length(omega)
    if w < 1.0e-6:
      break

    # normalize axis
    axis = omega / w

    # axis-angle to quaternion
    half_w = 0.5 * w
    qrot = wp.quat(
      axis[0] * wp.sin(half_w),
      axis[1] * wp.sin(half_w),
      axis[2] * wp.sin(half_w),
      wp.cos(half_w),
    )
    cell_quat = wp.normalize(qrot * cell_quat)

  # mju_negQuat: conjugate (R⁻¹) — negate xyz, keep w
  cell_quat_inv = wp.quat(-cell_quat[0], -cell_quat[1], -cell_quat[2], cell_quat[3])

  # Pre-compute displacements and velocities in corotational frame
  # (matches C: rotate all positions/velocities once, then K*u)
  idx_j = int(0)
  for li_j in range(order + 1):
    for lj_j in range(order + 1):
      for lk_j in range(order + 1):
        if idx_j < npc:
          gi_j = ci * order + li_j
          gj_j = cj * order + lj_j
          gk_j = ck * order + lk_j
          gidx_j = gi_j * ny_g * nz_g + gj_j * nz_g + gk_j

          xpos_j = flexnode_xpos_in[worldid, nstart + gidx_j]

          if not dsbl_spring:
            refpos_j = flex_node0[nstart + gidx_j]
            xrot_j = wp.quat_rotate(cell_quat_inv, xpos_j)
            displ_scratch_out[worldid, cellid, idx_j] = xrot_j - refpos_j

          if not dsbl_damper:
            bodyid_j = flex_nodebodyid[nstart + gidx_j]
            cvel_j = cvel_in[worldid, bodyid_j]
            omega_j = wp.spatial_top(cvel_j)
            vcom_j = wp.spatial_bottom(cvel_j)
            com_j = subtree_com_in[worldid, body_rootid[bodyid_j]]
            r_j = xpos_j - com_j
            vel_world_j = vcom_j + wp.cross(omega_j, r_j)
            vel_corot_scratch_out[worldid, cellid, idx_j] = wp.quat_rotate(cell_quat_inv, vel_world_j)

          idx_j += 1

  # Compute K*displacement and K*velocity per output node, then scatter forces
  idx_i = int(0)
  for li_i in range(order + 1):
    for lj_i in range(order + 1):
      for lk_i in range(order + 1):
        if idx_i < npc:
          gi_i = ci * order + li_i
          gj_i = cj * order + lj_i
          gk_i = ck * order + lk_i
          gidx_i = gi_i * ny_g * nz_g + gj_i * nz_g + gk_i
          bodyid_i = flex_nodebodyid[nstart + gidx_i]

          frc_spring = wp.vec3(0.0)
          frc_damper = wp.vec3(0.0)

          for comp_i in range(3):
            row = idx_i * 3 + comp_i
            val_spring = float(0.0)
            val_damper = float(0.0)

            for idx_j in range(npc):
              for comp_j in range(3):
                col = idx_j * 3 + comp_j
                K_ij = flex_stiffness[k_base + row * ndof_cell + col]

                if not dsbl_spring:
                  val_spring += K_ij * displ_scratch_out[worldid, cellid, idx_j][comp_j]

                if not dsbl_damper:
                  val_damper += K_ij * vel_corot_scratch_out[worldid, cellid, idx_j][comp_j]

            frc_spring[comp_i] = val_spring
            frc_damper[comp_i] = val_damper

          # Rotate forces back to world frame (R)
          frc_spring_world = wp.quat_rotate(cell_quat, frc_spring)
          frc_damper_world = wp.quat_rotate(cell_quat, frc_damper)

          # Scale damper force by damping coefficient
          frc_damper_world = frc_damper_world * flex_damping[f]

          # Apply forces to body DOFs (fast path: nodes at body origin)
          dofnum_i = body_dofnum[bodyid_i]
          dofadr_i = body_dofadr[bodyid_i]
          if dofnum_i > 0:
            centered = flex_centered[f]
            node_local = flex_node[nstart + gidx_i]
            at_origin = node_local[0] == 0.0 and node_local[1] == 0.0 and node_local[2] == 0.0

            if centered or at_origin:
              for x in range(3):
                if x < dofnum_i:
                  if not dsbl_spring:
                    wp.atomic_add(
                      qfrc_spring_out,
                      worldid,
                      dofadr_i + x,
                      frc_spring_world[x],
                    )
                  if not dsbl_damper:
                    wp.atomic_add(
                      qfrc_damper_out,
                      worldid,
                      dofadr_i + x,
                      frc_damper_world[x],
                    )

          idx_i += 1


@event_scope
def passive(m: Model, d: Data):
  """Adds all passive forces."""
  dsbl_spring = m.opt.disableflags & DisableBit.SPRING
  dsbl_damper = m.opt.disableflags & DisableBit.DAMPER

  if dsbl_spring and dsbl_damper:
    d.qfrc_spring.zero_()
    d.qfrc_damper.zero_()
    d.qfrc_gravcomp.zero_()
    d.qfrc_fluid.zero_()
    d.qfrc_passive.zero_()
    return

  wp.launch(
    _spring_damper_dof_passive,
    dim=(d.nworld, m.njnt),
    inputs=[
      m.opt.disableflags,
      m.qpos_spring,
      m.jnt_type,
      m.jnt_qposadr,
      m.jnt_dofadr,
      m.jnt_stiffness,
      m.jnt_stiffnesspoly,
      m.dof_damping,
      m.dof_dampingpoly,
      d.qpos,
      d.qvel,
    ],
    outputs=[d.qfrc_spring, d.qfrc_damper],
  )

  if m.ntendon:
    wp.launch(
      _spring_damper_tendon_passive,
      dim=(d.nworld, m.ntendon, m.max_ten_J_rownnz),
      inputs=[
        m.ten_J_rownnz,
        m.ten_J_rowadr,
        m.ten_J_colind,
        m.tendon_stiffness,
        m.tendon_stiffnesspoly,
        m.tendon_damping,
        m.tendon_dampingpoly,
        m.tendon_lengthspring,
        d.ten_J,
        d.ten_length,
        d.ten_velocity,
        dsbl_spring,
        dsbl_damper,
      ],
      outputs=[
        d.qfrc_spring,
        d.qfrc_damper,
      ],
    )

  if not dsbl_spring:
    wp.launch(
      _flex_elasticity,
      dim=(d.nworld, m.nflexelem),
      inputs=[
        m.nflex,
        m.opt.timestep,
        m.body_dofadr,
        m.flex_dim,
        m.flex_vertadr,
        m.flex_edgeadr,
        m.flex_elemadr,
        m.flex_elemnum,
        m.flex_elemdataadr,
        m.flex_stiffnessadr,
        m.flex_elemedgeadr,
        m.flex_vertbodyid,
        m.flex_elem,
        m.flex_elemedge,
        m.flexedge_length0,
        m.flex_stiffness,
        m.flex_damping,
        d.flexvert_xpos,
        d.flexedge_length,
        d.flexedge_velocity,
        dsbl_damper,
      ],
      outputs=[d.qfrc_spring],
    )
    wp.launch(
      _flex_bending,
      dim=(d.nworld, m.nflexedge),
      inputs=[
        m.nflex,
        m.body_dofadr,
        m.flex_dim,
        m.flex_vertadr,
        m.flex_edgeadr,
        m.flex_edgenum,
        m.flex_vertbodyid,
        m.flex_edge,
        m.flex_edgeflap,
        m.flex_bending,
        d.flexvert_xpos,
      ],
      outputs=[d.qfrc_spring],
    )

  # Launch passive interp kernel for interpolated flex (trilinear/quadratic)
  if m.nflex and m.nflexintcell > 0:
    displ_scratch = wp.empty((d.nworld, m.nflexintcell, 27), dtype=wp.vec3)
    vel_corot_scratch = wp.empty((d.nworld, m.nflexintcell, 27), dtype=wp.vec3)
    wp.launch(
      _flex_passive_interp,
      dim=(d.nworld, m.nflexintcell),
      inputs=[
        m.nflex,
        m.body_rootid,
        m.body_dofnum,
        m.body_dofadr,
        m.flex_interp,
        m.flex_cellnum,
        m.flex_nodeadr,
        m.flex_stiffnessadr,
        m.flex_nodebodyid,
        m.flex_node,
        m.flex_node0,
        m.flex_stiffness,
        m.flex_damping,
        m.flex_edgeequality,
        m.flex_centered,
        d.subtree_com,
        d.cvel,
        d.flexnode_xpos,
        dsbl_spring,
        dsbl_damper,
      ],
      outputs=[
        d.qfrc_spring,
        d.qfrc_damper,
        displ_scratch,
        vel_corot_scratch,
      ],
    )

  gravcomp = m.ngravcomp and not (m.opt.disableflags & DisableBit.GRAVITY)

  if gravcomp:
    d.qfrc_gravcomp.zero_()
    wp.launch(
      _gravity_force,
      dim=(d.nworld, m.nbody - 1, m.nv),
      inputs=[
        m.opt.gravity,
        m.body_parentid,
        m.body_rootid,
        m.body_mass,
        m.body_gravcomp,
        m.dof_bodyid,
        m.body_isdofancestor,
        d.xipos,
        d.subtree_com,
        d.cdof,
      ],
      outputs=[d.qfrc_gravcomp],
    )

  if m.has_fluid:
    _fluid(m, d)

  wp.launch(
    _qfrc_passive,
    dim=(d.nworld, m.nv),
    inputs=[
      m.jnt_actgravcomp,
      m.dof_jntid,
      m.has_fluid,
      d.qfrc_spring,
      d.qfrc_damper,
      d.qfrc_gravcomp,
      d.qfrc_fluid,
      gravcomp,
    ],
    outputs=[
      d.qfrc_passive,
    ],
  )

  if m.callback.passive:
    m.callback.passive(m, d)

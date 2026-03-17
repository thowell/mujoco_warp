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
"""Complementarity-free analytical contact forces.

Implements dual-cone impedance contact model from ComFree-Sim (arXiv:2603.12185).

Key equations (paper notation, adapted for MuJoCo Warp pipeline):
  v_contact = J * qvel                                      (contact-space velocity)
  lambda_n = impedance * max(-(v_n*dt + phi), 0)             (normal force)
  lambda_j = impedance * max(mu*d_j*v_t*dt - (v_n*dt+phi),0) (friction per face)

Where impedance = (kuser*dt + duser) / M(phi), with M(phi) = invweight*(1-r)/r.
"""

import warp as wp

from mujoco_warp._src import support
from mujoco_warp._src import types
from mujoco_warp._src.types import ContactType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import vec5
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.func
def _gap_impedance(
  solimp: vec5,
  invweight: float,
  phi: float,
) -> float:
  """Compute gap-dependent impedance 1/M(phi).

  Returns:  1 / max(invweight * (1-r)/r, MJ_MINVAL)

  where r(|phi|) is MuJoCo's solimp function (Eq. 10 in paper).
  """
  dmin = solimp[0]
  dmax = solimp[1]
  width = solimp[2]
  mid = solimp[3]
  power = solimp[4]

  dmin = wp.clamp(dmin, types.MJ_MINIMP, types.MJ_MAXIMP)
  dmax = wp.clamp(dmax, types.MJ_MINIMP, types.MJ_MAXIMP)
  width = wp.max(MJ_MINVAL, width)
  mid = wp.clamp(mid, types.MJ_MINIMP, types.MJ_MAXIMP)
  power = wp.max(1.0, power)

  # r(|phi|) — MuJoCo impedance function
  imp_x = wp.abs(phi) / width
  imp_a = (1.0 / wp.pow(mid, power - 1.0)) * wp.pow(imp_x, power)
  imp_b = 1.0 - (1.0 / wp.pow(1.0 - mid, power - 1.0)) * wp.pow(1.0 - imp_x, power)
  imp_y = wp.where(imp_x < mid, imp_a, imp_b)
  r = dmin + imp_y * (dmax - dmin)
  r = wp.clamp(r, dmin, dmax)
  r = wp.where(imp_x > 1.0, dmax, r)

  # 1 / M(phi)
  inv_M = 1.0 / wp.max(invweight * (1.0 - r) / r, MJ_MINVAL)

  return inv_M


@wp.kernel
def _passive_contact_force(
  # Model:
  body_parentid: wp.array(dtype=int),
  body_rootid: wp.array(dtype=int),
  body_weldid: wp.array(dtype=int),
  body_dofnum: wp.array(dtype=int),
  body_dofadr: wp.array(dtype=int),
  body_invweight0: wp.array2d(dtype=wp.vec2),
  dof_bodyid: wp.array(dtype=int),
  dof_parentid: wp.array(dtype=int),
  geom_bodyid: wp.array(dtype=int),
  flex_vertadr: wp.array(dtype=int),
  flex_vertbodyid: wp.array(dtype=int),
  opt_timestep: wp.array(dtype=float),
  # Data in:
  qvel_in: wp.array2d(dtype=float),
  qacc_smooth_in: wp.array2d(dtype=float),
  subtree_com_in: wp.array2d(dtype=wp.vec3),
  cdof_in: wp.array2d(dtype=wp.spatial_vector),
  nacon_in: wp.array(dtype=int),
  # Contact in:
  dist_in: wp.array(dtype=float),
  dim_in: wp.array(dtype=int),
  includemargin_in: wp.array(dtype=float),
  worldid_in: wp.array(dtype=int),
  geom_in: wp.array(dtype=wp.vec2i),
  flex_in: wp.array(dtype=wp.vec2i),
  vert_in: wp.array(dtype=wp.vec2i),
  pos_in: wp.array(dtype=wp.vec3),
  frame_in: wp.array(dtype=wp.mat33),
  friction_in: wp.array(dtype=vec5),
  solimp_in: wp.array(dtype=vec5),
  type_in: wp.array(dtype=int),
  # Params:
  kuser_in: wp.array(dtype=float),
  duser_in: wp.array(dtype=float),
  # Data out:
  qfrc_contact_out: wp.array2d(dtype=float),
):
  conid = wp.tid()

  if conid >= nacon_in[0]:
    return

  if not type_in[conid] & ContactType.CONSTRAINT:
    return

  condim = dim_in[conid]

  # only condim == 3 supported (normal + 2 tangential)
  if condim != 3:
    return

  includemargin = includemargin_in[conid]
  phi = dist_in[conid] - includemargin  # signed gap (negative = penetrating)

  if phi >= 0.0:
    return

  worldid = worldid_in[conid]
  dt = opt_timestep[worldid % opt_timestep.shape[0]]

  # identify contacting bodies
  geom = geom_in[conid]

  if geom[0] >= 0:
    body1 = geom_bodyid[geom[0]]
  else:
    flex = flex_in[conid]
    vert = vert_in[conid]
    body1 = flex_vertbodyid[flex_vertadr[flex[0]] + vert[0]]

  if geom[1] >= 0:
    body2 = geom_bodyid[geom[1]]
  else:
    flex = flex_in[conid]
    vert = vert_in[conid]
    body2 = flex_vertbodyid[flex_vertadr[flex[1]] + vert[1]]

  body1 = body_weldid[body1]
  body2 = body_weldid[body2]

  con_pos = pos_in[conid]
  frame = frame_in[conid]
  normal = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2])
  tangent1 = wp.vec3(frame[1, 0], frame[1, 1], frame[1, 2])
  tangent2 = wp.vec3(frame[2, 0], frame[2, 1], frame[2, 2])
  mu = friction_in[conid][0]

  # compute invweight
  body_invweight0_id = worldid % body_invweight0.shape[0]
  invweight = body_invweight0[body_invweight0_id, body1][0] + body_invweight0[body_invweight0_id, body2][0]

  # gap-dependent impedance (Eq. 9-10)
  inv_M = _gap_impedance(solimp_in[conid], invweight, phi)

  # impedance = (kuser*dt + duser) / M(phi)
  kuser = kuser_in[worldid % kuser_in.shape[0]]
  duser = duser_in[worldid % duser_in.shape[0]]
  impedance = (kuser * dt + duser) * inv_M

  # --- Phase 1: contact-space velocity from current qvel ---
  v_n = float(0.0)
  v_t1 = float(0.0)
  v_t2 = float(0.0)

  da1 = int(body_dofadr[body1] + body_dofnum[body1] - 1)
  da2 = int(body_dofadr[body2] + body_dofnum[body2] - 1)

  while da1 >= 0 or da2 >= 0:
    da = wp.max(da1, da2)

    jac1p, jac1r = support.jac_dof(
      body_parentid, body_rootid, dof_bodyid,
      subtree_com_in, cdof_in,
      con_pos, body1, da, worldid,
    )
    jac2p, jac2r = support.jac_dof(
      body_parentid, body_rootid, dof_bodyid,
      subtree_com_in, cdof_in,
      con_pos, body2, da, worldid,
    )

    # predicted smooth velocity for this DOF (paper Eq. 2)
    v_dof = qvel_in[worldid, da] + dt * qacc_smooth_in[worldid, da]

    jacp_dif = wp.vec3(
      jac2p[0] - jac1p[0],
      jac2p[1] - jac1p[1],
      jac2p[2] - jac1p[2],
    )

    v_n += wp.dot(normal, jacp_dif) * v_dof
    v_t1 += wp.dot(tangent1, jacp_dif) * v_dof
    v_t2 += wp.dot(tangent2, jacp_dif) * v_dof

    if da1 == da:
      da1 = dof_parentid[da1]
    if da2 == da:
      da2 = dof_parentid[da2]

  # --- Phase 2: dual-cone polyhedral violations ---
  # gap_term = v_n * dt + phi  (negative when penetrating)
  gap_term = v_n * dt + phi

  # Normal: max(-(v_n*dt + phi), 0)
  lambda_n = impedance * wp.max(-gap_term, 0.0)

  # Tangential: 4 polyhedral faces (+t1, -t1, +t2, -t2)
  # violation_j = mu * (d_j . v_t) * dt - (v_n*dt + phi)
  lam_t1_pos = impedance * wp.max(mu * v_t1 * dt - gap_term, 0.0)
  lam_t1_neg = impedance * wp.max(-mu * v_t1 * dt - gap_term, 0.0)
  lam_t2_pos = impedance * wp.max(mu * v_t2 * dt - gap_term, 0.0)
  lam_t2_neg = impedance * wp.max(-mu * v_t2 * dt - gap_term, 0.0)

  # Total force (Eq. 7): face +d_j contributes along (n - mu*d_j), face -d_j along (n + mu*d_j)
  # Normal: all faces contribute positively
  f_normal = (lambda_n + lam_t1_pos + lam_t1_neg + lam_t2_pos + lam_t2_neg) * normal
  # Tangential: J̃ᵀ gives mu*(lam_neg - lam_pos) to oppose sliding
  f_tangent = mu * ((lam_t1_neg - lam_t1_pos) * tangent1 + (lam_t2_neg - lam_t2_pos) * tangent2)
  f_total = f_normal + f_tangent

  # --- Phase 3: accumulate J^T * f_total ---
  da1 = int(body_dofadr[body1] + body_dofnum[body1] - 1)
  da2 = int(body_dofadr[body2] + body_dofnum[body2] - 1)

  while da1 >= 0 or da2 >= 0:
    da = wp.max(da1, da2)

    jac1p, jac1r = support.jac_dof(
      body_parentid, body_rootid, dof_bodyid,
      subtree_com_in, cdof_in,
      con_pos, body1, da, worldid,
    )
    jac2p, jac2r = support.jac_dof(
      body_parentid, body_rootid, dof_bodyid,
      subtree_com_in, cdof_in,
      con_pos, body2, da, worldid,
    )

    jacp_dif = wp.vec3(
      jac2p[0] - jac1p[0],
      jac2p[1] - jac1p[1],
      jac2p[2] - jac1p[2],
    )

    qfrc = wp.dot(f_total, jacp_dif)
    wp.atomic_add(qfrc_contact_out[worldid], da, qfrc)

    if da1 == da:
      da1 = dof_parentid[da1]
    if da2 == da:
      da2 = dof_parentid[da2]


@event_scope
def passive_contact(m: Model, d: Data):
  """Compute analytical complementarity-free contact forces."""
  d.qfrc_contact.zero_()

  wp.launch(
    _passive_contact_force,
    dim=d.naconmax,
    inputs=[
      m.body_parentid,
      m.body_rootid,
      m.body_weldid,
      m.body_dofnum,
      m.body_dofadr,
      m.body_invweight0,
      m.dof_bodyid,
      m.dof_parentid,
      m.geom_bodyid,
      m.flex_vertadr,
      m.flex_vertbodyid,
      m.opt.timestep,
      d.qvel,
      d.qacc_smooth,
      d.subtree_com,
      d.cdof,
      d.nacon,
      d.contact.dist,
      d.contact.dim,
      d.contact.includemargin,
      d.contact.worldid,
      d.contact.geom,
      d.contact.flex,
      d.contact.vert,
      d.contact.pos,
      d.contact.frame,
      d.contact.friction,
      d.contact.solimp,
      d.contact.type,
      m.opt.passive_kuser,
      m.opt.passive_duser,
    ],
    outputs=[d.qfrc_contact],
  )

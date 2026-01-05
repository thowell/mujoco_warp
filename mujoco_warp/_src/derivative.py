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

from mujoco_warp._src import forward
from mujoco_warp._src import math
from mujoco_warp._src import util_misc
from mujoco_warp._src.support import next_act
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import BiasType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import DynType
from mujoco_warp._src.types import GainType
from mujoco_warp._src.types import IntegratorType
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import vec10
from mujoco_warp._src.types import vec10f
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _qderiv_actuator_passive_vel(
  # Model:
  opt_timestep: wp.array[float],
  actuator_dyntype: wp.array[int],
  actuator_gaintype: wp.array[int],
  actuator_biastype: wp.array[int],
  actuator_actadr: wp.array[int],
  actuator_actnum: wp.array[int],
  actuator_forcelimited: wp.array[bool],
  actuator_actlimited: wp.array[bool],
  actuator_dynprm: wp.array2d[vec10f],
  actuator_gainprm: wp.array2d[vec10f],
  actuator_biasprm: wp.array2d[vec10f],
  actuator_actearly: wp.array[bool],
  actuator_forcerange: wp.array2d[wp.vec2],
  actuator_actrange: wp.array2d[wp.vec2],
  # Data in:
  act_in: wp.array2d[float],
  ctrl_in: wp.array2d[float],
  act_dot_in: wp.array2d[float],
  actuator_force_in: wp.array2d[float],
  # Out:
  vel_out: wp.array2d[float],
):
  worldid, actid = wp.tid()

  actuator_gainprm_id = worldid % actuator_gainprm.shape[0]
  actuator_biasprm_id = worldid % actuator_biasprm.shape[0]

  if actuator_gaintype[actid] == GainType.AFFINE:
    gain = actuator_gainprm[actuator_gainprm_id, actid][2]
  else:
    gain = 0.0

  if actuator_biastype[actid] == BiasType.AFFINE:
    bias = actuator_biasprm[actuator_biasprm_id, actid][2]
  elif actuator_biastype[actid] == BiasType.DCMOTOR:
    dynprm = actuator_dynprm[worldid % actuator_dynprm.shape[0], actid]
    te = dynprm[0]
    if te <= 0.0:
      gainprm = actuator_gainprm[actuator_gainprm_id, actid]
      R = gainprm[0]
      K = gainprm[1]

      slots = util_misc.dcmotor_slots(dynprm, gainprm)
      slot_Ta = slots[2]

      if slot_Ta >= 0:
        adr = actuator_actadr[actid] + slot_Ta
        T = act_in[worldid, adr]
        alpha = gainprm[2]
        T0 = gainprm[3]
        Ta = dynprm[4]
        R *= 1.0 + alpha * (T + Ta - T0)

      bias = -K * K / wp.max(MJ_MINVAL, R)
    else:
      bias = 0.0
  else:
    bias = 0.0

  if bias == 0.0 and gain == 0.0:
    vel_out[worldid, actid] = 0.0
    return

  # skip if force is clamped by forcerange
  if actuator_forcelimited[actid]:
    force = actuator_force_in[worldid, actid]
    forcerange = actuator_forcerange[worldid % actuator_forcerange.shape[0], actid]
    if force <= forcerange[0] or force >= forcerange[1]:
      vel_out[worldid, actid] = 0.0
      return

  vel = float(bias)
  if actuator_dyntype[actid] != DynType.NONE:
    if gain != 0.0:
      act_adr = actuator_actadr[actid] + actuator_actnum[actid] - 1

      # use next activation if actearly is set (matching forward pass)
      if actuator_actearly[actid]:
        act = next_act(
          opt_timestep[worldid % opt_timestep.shape[0]],
          actuator_dyntype[actid],
          actuator_dynprm[worldid % actuator_dynprm.shape[0], actid],
          actuator_actrange[worldid % actuator_actrange.shape[0], actid],
          act_in[worldid, act_adr],
          act_dot_in[worldid, act_adr],
          1.0,
          actuator_actlimited[actid],
        )
      else:
        act = act_in[worldid, act_adr]

      vel += gain * act
  else:
    if gain != 0.0:
      vel += gain * ctrl_in[worldid, actid]

  vel_out[worldid, actid] = vel


@wp.func
def _nonzero_mask(x: float) -> float:
  """Returns 1.0 for non-zero input, 0.0 otherwise."""
  if x != 0.0:
    return 1.0
  return 0.0


@wp.kernel
def _qderiv_actuator_passive_actuation_dense(
  # Model:
  nu: int,
  # Data in:
  moment_rownnz_in: wp.array2d[int],
  moment_rowadr_in: wp.array2d[int],
  moment_colind_in: wp.array2d[int],
  actuator_moment_in: wp.array2d[float],
  # In:
  vel_in: wp.array2d[float],
  qMi: wp.array[int],
  qMj: wp.array[int],
  # Out:
  qDeriv_out: wp.array3d[float],
):
  worldid, elemid = wp.tid()

  dofiid = qMi[elemid]
  dofjid = qMj[elemid]
  qderiv_contrib = float(0.0)
  for actid in range(nu):
    vel = vel_in[worldid, actid]
    if vel == 0.0:
      continue

    # TODO(team): restructure sparse version for better parallelism?
    moment_i = float(0.0)
    moment_j = float(0.0)

    rownnz = moment_rownnz_in[worldid, actid]
    rowadr = moment_rowadr_in[worldid, actid]
    for i in range(rownnz):
      sparseid = rowadr + i
      colind = moment_colind_in[worldid, sparseid]
      if colind == dofiid:
        moment_i = actuator_moment_in[worldid, sparseid]
      if colind == dofjid:
        moment_j = actuator_moment_in[worldid, sparseid]
      if moment_i != 0.0 and moment_j != 0.0:
        break

    if moment_i == 0 and moment_j == 0:
      continue

    qderiv_contrib += moment_i * moment_j * vel

  qDeriv_out[worldid, dofiid, dofjid] = qderiv_contrib
  if dofiid != dofjid:
    qDeriv_out[worldid, dofjid, dofiid] = qderiv_contrib


@wp.kernel
def _qderiv_actuator_passive_actuation_sparse(
  # Model:
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  # Data in:
  moment_rownnz_in: wp.array2d[int],
  moment_rowadr_in: wp.array2d[int],
  moment_colind_in: wp.array2d[int],
  actuator_moment_in: wp.array2d[float],
  # In:
  vel_in: wp.array2d[float],
  qMj: wp.array[int],
  # Out:
  qDeriv_out: wp.array3d[float],
):
  worldid, actid = wp.tid()

  vel = vel_in[worldid, actid]
  if vel == 0.0:
    return

  rownnz = moment_rownnz_in[worldid, actid]
  rowadr = moment_rowadr_in[worldid, actid]

  for i in range(rownnz):
    rowadri = rowadr + i
    moment_i = actuator_moment_in[worldid, rowadri]
    if moment_i == 0.0:
      continue
    dofi = moment_colind_in[worldid, rowadri]

    for j in range(i + 1):
      rowadrj = rowadr + j
      moment_j = actuator_moment_in[worldid, rowadrj]
      if moment_j == 0.0:
        continue
      dofj = moment_colind_in[worldid, rowadrj]

      contrib = moment_i * moment_j * vel

      # Search the corresponding elemid
      # TODO: This could be precalculated for improved performance
      row = dofi
      col = dofj
      row_startk = M_rowadr[row] - 1
      row_nnz = M_rownnz[row]
      for k in range(row_nnz):
        row_startk += 1
        if qMj[row_startk] == col:
          wp.atomic_add(qDeriv_out[worldid, 0], row_startk, contrib)
          break


@wp.kernel
def _qderiv_actuator_passive(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  dof_damping: wp.array2d[float],
  dof_dampingpoly: wp.array2d[wp.vec2],
  is_sparse: bool,
  # Data in:
  qvel_in: wp.array2d[float],
  qM_in: wp.array3d[float],
  # In:
  qMi: wp.array[int],
  qMj: wp.array[int],
  qDeriv_in: wp.array3d[float],
  # Out:
  qDeriv_out: wp.array3d[float],
):
  worldid, elemid = wp.tid()

  dofiid = qMi[elemid]
  dofjid = qMj[elemid]

  if is_sparse:
    qderiv = qDeriv_in[worldid, 0, elemid]
  else:
    qderiv = qDeriv_in[worldid, dofiid, dofjid]

  if not (opt_disableflags & DisableBit.DAMPER) and dofiid == dofjid:
    damping = dof_damping[worldid % dof_damping.shape[0], dofiid]
    dpoly = dof_dampingpoly[worldid % dof_dampingpoly.shape[0], dofiid]
    v = qvel_in[worldid, dofiid]
    qderiv -= util_misc._poly_force_deriv(damping, dpoly, v, 1)

  qderiv *= opt_timestep[worldid % opt_timestep.shape[0]]

  if is_sparse:
    qDeriv_out[worldid, 0, elemid] = qM_in[worldid, 0, elemid] - qderiv
  else:
    qM = qM_in[worldid, dofiid, dofjid] - qderiv
    qDeriv_out[worldid, dofiid, dofjid] = qM
    if dofiid != dofjid:
      qDeriv_out[worldid, dofjid, dofiid] = qM


# TODO(team): improve performance with tile operations?
@wp.kernel
def _qderiv_tendon_damping(
  # Model:
  ntendon: int,
  opt_timestep: wp.array[float],
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  tendon_damping: wp.array2d[float],
  tendon_dampingpoly: wp.array2d[wp.vec2],
  is_sparse: bool,
  # Data in:
  ten_J_in: wp.array2d[float],
  ten_velocity_in: wp.array2d[float],
  # In:
  qMi: wp.array[int],
  qMj: wp.array[int],
  # Out:
  qDeriv_out: wp.array3d[float],
):
  worldid, elemid = wp.tid()
  dofiid = qMi[elemid]
  dofjid = qMj[elemid]

  qderiv = float(0.0)
  tendon_damping_id = worldid % tendon_damping.shape[0]
  for tenid in range(ntendon):
    damping = tendon_damping[tendon_damping_id, tenid]
    dpoly = tendon_dampingpoly[worldid % tendon_dampingpoly.shape[0], tenid]
    if damping == 0.0 and dpoly[0] == 0.0 and dpoly[1] == 0.0:
      continue

    rownnz = ten_J_rownnz[tenid]
    rowadr = ten_J_rowadr[tenid]
    Ji = float(0.0)
    Jj = float(0.0)
    for k in range(rownnz):
      if Ji != 0.0 and Jj != 0.0:
        break
      sparseid = rowadr + k
      colind = ten_J_colind[sparseid]
      if colind == dofiid:
        Ji = ten_J_in[worldid, sparseid]
      if colind == dofjid:
        Jj = ten_J_in[worldid, sparseid]

    v = ten_velocity_in[worldid, tenid]
    qderiv -= Ji * Jj * util_misc._poly_force_deriv(damping, dpoly, v, 1)

  qderiv *= opt_timestep[worldid % opt_timestep.shape[0]]

  if is_sparse:
    qDeriv_out[worldid, 0, elemid] -= qderiv
  else:
    qDeriv_out[worldid, dofiid, dofjid] -= qderiv
    if dofiid != dofjid:
      qDeriv_out[worldid, dofjid, dofiid] -= qderiv


@wp.kernel
def rne_deriv_cvel_cdof_dot(
  # Model:
  body_parentid: wp.array[int],
  body_jntnum: wp.array[int],
  body_jntadr: wp.array[int],
  body_dofadr: wp.array[int],
  jnt_type: wp.array[int],
  # Data in:
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  body_tree_: wp.array[int],
  # Out:
  Dcvel_out: wp.array3d[wp.spatial_vector],
  Dcdof_dot_out: wp.array3d[wp.spatial_vector],
):
  """Forward pass: compute d(cvel)/d(qvel_k) and d(cdof_dot)/d(qvel_k).

  Mirrors the accumulation order of comvel for each joint type.

  Dcdof_dot for rotation DOFs of free joints (dofid+0..2) is zero because the
  forward pass sets cdof_dot[dofid+0..2] = 0.  The Dcdof_dot array is
  zero-initialized so no explicit write is needed.
  """
  worldid, nodeid, dofid = wp.tid()
  bodyid = body_tree_[nodeid]
  dofadr = body_dofadr[bodyid]
  jntid = body_jntadr[bodyid]
  jntnum = body_jntnum[bodyid]
  pid = body_parentid[bodyid]

  cdof = cdof_in[worldid]

  # Initialize from parent
  cvel_k = Dcvel_out[worldid, pid, dofid]

  if jntnum == 0:
    Dcvel_out[worldid, bodyid, dofid] = cvel_k
    return

  dof_i = dofadr

  for j in range(jntid, jntid + jntnum):
    jnttype = jnt_type[j]

    if jnttype == 0:  # FREE
      # rotation DOFs (dof_i+0..2) contribute to cvel
      if dofid >= dof_i and dofid < dof_i + 3:
        cvel_k += cdof[dofid]

      # cdof_dot for rotation DOFs is zero (set in forward kinematics),
      # so Dcdof_dot for rotation DOFs is zero (from wp.zeros init)

      # derivative of cdof_dot for translation DOFs 3,4,5
      Dcdof_dot_out[worldid, dof_i + 3, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 3])
      Dcdof_dot_out[worldid, dof_i + 4, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 4])
      Dcdof_dot_out[worldid, dof_i + 5, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 5])

      # translation DOFs (dof_i+3..5) contribute to cvel
      if dofid >= dof_i + 3 and dofid < dof_i + 6:
        cvel_k += cdof[dofid]

      dof_i += 6

    elif jnttype == 1:  # BALL
      Dcdof_dot_out[worldid, dof_i + 0, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 0])
      Dcdof_dot_out[worldid, dof_i + 1, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 1])
      Dcdof_dot_out[worldid, dof_i + 2, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 2])

      if dofid >= dof_i and dofid < dof_i + 3:
        cvel_k += cdof[dofid]

      dof_i += 3
    else:  # HINGE or SLIDE
      Dcdof_dot_out[worldid, dof_i, dofid] = math.motion_cross(cvel_k, cdof[dof_i])

      if dofid == dof_i:
        cvel_k += cdof[dof_i]

      dof_i += 1

  Dcvel_out[worldid, bodyid, dofid] = cvel_k


@wp.kernel
def rne_deriv_cacc_cfrcbody_forward(
  # Model:
  body_parentid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  cinert_in: wp.array2d[vec10],
  cvel_in: wp.array2d[wp.spatial_vector],
  cdof_dot_in: wp.array2d[wp.spatial_vector],
  # In:
  body_tree_: wp.array[int],
  Dcvel_in: wp.array3d[wp.spatial_vector],
  Dcdof_dot_in: wp.array3d[wp.spatial_vector],
  # Out:
  Dcacc_out: wp.array3d[wp.spatial_vector],
  Dcfrcbody_out: wp.array3d[wp.spatial_vector],
):
  """Forward pass: compute d(cacc)/d(qvel_k) and d(cfrc_body)/d(qvel_k)."""
  worldid, nodeid, dofid = wp.tid()
  bodyid = body_tree_[nodeid]
  dofadr = body_dofadr[bodyid]
  dofnum = body_dofnum[bodyid]
  pid = body_parentid[bodyid]

  qvel = qvel_in[worldid]

  dcacc = Dcacc_out[worldid, pid, dofid]

  for j in range(dofadr, dofadr + dofnum):
    # Term 1: d(cdof_dot * qvel)/d(qvel_k) when j == dofid
    if j == dofid:
      dcacc += cdof_dot_in[worldid, j]

    # Term 2: cdof_dot depends on cvel which depends on qvel_k
    dcdofdot = Dcdof_dot_in[worldid, j, dofid]
    dcacc += dcdofdot * qvel[j]

  Dcacc_out[worldid, bodyid, dofid] = dcacc

  # d(cfrc_body)/d(qvel_k)
  cinert = cinert_in[worldid, bodyid]
  cvel = cvel_in[worldid, bodyid]
  dcvel = Dcvel_in[worldid, bodyid, dofid]

  # term1 = cinert * d(cacc)/d(qvel_k)
  term1 = math.inert_vec(cinert, dcacc)

  # term2 = d(cvel x* (cinert * cvel))/d(qvel_k)
  cinert_cvel = math.inert_vec(cinert, cvel)
  cinert_dcvel = math.inert_vec(cinert, dcvel)
  term2 = math.motion_cross_force(dcvel, cinert_cvel) + math.motion_cross_force(cvel, cinert_dcvel)

  Dcfrcbody_out[worldid, bodyid, dofid] = term1 + term2


@wp.kernel
def rne_deriv_cfrcbody_backward(
  # Model:
  body_parentid: wp.array[int],
  # In:
  body_tree_: wp.array[int],
  # Out:
  Dcfrcbody_out: wp.array3d[wp.spatial_vector],
):
  """Backward pass: accumulate d(cfrc_body) from children to parents."""
  worldid, nodeid, dofid = wp.tid()
  bodyid = body_tree_[nodeid]
  pid = body_parentid[bodyid]

  # body_tree never contains bodyid=0 (worldbody), so pid >= 0 is always valid.
  # Siblings at the same level may share a parent; atomic_add handles this.
  val = Dcfrcbody_out[worldid, bodyid, dofid]
  wp.atomic_add(Dcfrcbody_out[worldid, pid], dofid, val)


@wp.kernel
def rne_deriv_body2jnt_sparse(
  # Model:
  dof_bodyid: wp.array[int],
  # Data in:
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  timestep: wp.array[float],
  qMi: wp.array[int],
  qMj: wp.array[int],
  Dcfrcbody_in: wp.array3d[wp.spatial_vector],
  # Out:
  qDeriv_out: wp.array3d[float],
):
  """Project body-space RNE derivatives into joint-space qDeriv (sparse)."""
  worldid, elemid = wp.tid()
  dt = timestep[worldid % timestep.shape[0]]

  i = qMi[elemid]
  j = qMj[elemid]

  body_i = dof_bodyid[i]
  dcfrc = Dcfrcbody_in[worldid, body_i, j]
  term = wp.dot(cdof_in[worldid, i], dcfrc)

  wp.atomic_add(qDeriv_out[worldid, 0], elemid, dt * term)


@wp.kernel
def rne_deriv_body2jnt_dense(
  # Model:
  dof_bodyid: wp.array[int],
  # Data in:
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  timestep: wp.array[float],
  Dcfrcbody_in: wp.array3d[wp.spatial_vector],
  # Out:
  qDeriv_out: wp.array3d[float],
):
  """Project body-space RNE derivatives into joint-space qDeriv (dense)."""
  worldid, i, j = wp.tid()
  dt = timestep[worldid % timestep.shape[0]]

  body_i = dof_bodyid[i]
  dcfrc = Dcfrcbody_in[worldid, body_i, j]
  term = wp.dot(cdof_in[worldid, i], dcfrc)

  qDeriv_out[worldid, i, j] += dt * term


def rne_vel_deriv(m: Model, d: Data, out: wp.array2d[float]):
  """Compute RNE velocity derivatives and add to qDeriv output.

  Implements the analytical derivative of inverse-dynamics Coriolis/centrifugal
  forces with respect to joint velocities (equivalent to mjd_rne_vel in the C
  MuJoCo engine).

  Args:
    m: The model (device).
    d: The data (device).
    out: qDeriv-like output array to accumulate RNE terms into.
  """
  # TODO(team): consider caching these allocations
  Dcvel = wp.zeros((d.nworld, m.nbody, m.nv), dtype=wp.spatial_vector)
  Dcdof_dot = wp.zeros((d.nworld, m.nv, m.nv), dtype=wp.spatial_vector)
  Dcacc = wp.zeros((d.nworld, m.nbody, m.nv), dtype=wp.spatial_vector)
  Dcfrcbody = wp.zeros((d.nworld, m.nbody, m.nv), dtype=wp.spatial_vector)

  # Forward pass 1: compute Dcvel and Dcdof_dot
  for body_tree in m.body_tree:
    wp.launch(
      rne_deriv_cvel_cdof_dot,
      dim=(d.nworld, body_tree.size, m.nv),
      inputs=[
        m.body_parentid,
        m.body_jntnum,
        m.body_jntadr,
        m.body_dofadr,
        m.jnt_type,
        d.cdof,
        body_tree,
      ],
      outputs=[Dcvel, Dcdof_dot],
    )

  # Forward pass 2: compute Dcacc and Dcfrcbody
  for body_tree in m.body_tree:
    wp.launch(
      rne_deriv_cacc_cfrcbody_forward,
      dim=(d.nworld, body_tree.size, m.nv),
      inputs=[
        m.body_parentid,
        m.body_dofnum,
        m.body_dofadr,
        d.qvel,
        d.cinert,
        d.cvel,
        d.cdof_dot,
        body_tree,
        Dcvel,
        Dcdof_dot,
      ],
      outputs=[Dcacc, Dcfrcbody],
    )

  # Backward pass: accumulate Dcfrcbody from children to parents
  for body_tree in reversed(m.body_tree):
    wp.launch(
      rne_deriv_cfrcbody_backward,
      dim=(d.nworld, body_tree.size, m.nv),
      inputs=[m.body_parentid, body_tree],
      outputs=[Dcfrcbody],
    )

  # Project body-space derivatives into joint-space qDeriv
  if m.is_sparse:
    wp.launch(
      rne_deriv_body2jnt_sparse,
      dim=(d.nworld, m.qD_fullm_i.size),
      inputs=[m.dof_bodyid, d.cdof, m.opt.timestep, m.qD_fullm_i, m.qD_fullm_j, Dcfrcbody],
      outputs=[out],
    )
  else:
    wp.launch(
      rne_deriv_body2jnt_dense,
      dim=(d.nworld, m.nv, m.nv),
      inputs=[m.dof_bodyid, d.cdof, m.opt.timestep, Dcfrcbody],
      outputs=[out],
    )


@event_scope
def deriv_smooth_vel(m: Model, d: Data, out: wp.array2d[float]):
  """Analytical derivative of smooth forces w.r.t. velocities.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state and output arrays (device).
    out: qM - dt * qDeriv (derivatives of smooth forces w.r.t velocities).
  """
  qMi = m.qM_fullm_i
  qMj = m.qM_fullm_j

  if ~(m.opt.disableflags & (DisableBit.ACTUATION | DisableBit.DAMPER)):
    # TODO(team): only clear elements not set by _qderiv_actuator_passive
    out.zero_()
    if m.nu > 0 and not (m.opt.disableflags & DisableBit.ACTUATION):
      vel = wp.empty((d.nworld, m.nu), dtype=float)
      wp.launch(
        _qderiv_actuator_passive_vel,
        dim=(d.nworld, m.nu),
        inputs=[
          m.opt.timestep,
          m.actuator_dyntype,
          m.actuator_gaintype,
          m.actuator_biastype,
          m.actuator_actadr,
          m.actuator_actnum,
          m.actuator_forcelimited,
          m.actuator_actlimited,
          m.actuator_dynprm,
          m.actuator_gainprm,
          m.actuator_biasprm,
          m.actuator_actearly,
          m.actuator_forcerange,
          m.actuator_actrange,
          d.act,
          d.ctrl,
          d.act_dot,
          d.actuator_force,
        ],
        outputs=[vel],
      )
      if m.is_sparse:
        wp.launch(
          _qderiv_actuator_passive_actuation_sparse,
          dim=(d.nworld, m.nu),
          inputs=[m.M_rownnz, m.M_rowadr, d.moment_rownnz, d.moment_rowadr, d.moment_colind, d.actuator_moment, vel, qMj],
          outputs=[out],
        )
      else:
        wp.launch(
          _qderiv_actuator_passive_actuation_dense,
          dim=(d.nworld, qMi.size),
          inputs=[m.nu, d.moment_rownnz, d.moment_rowadr, d.moment_colind, d.actuator_moment, vel, qMi, qMj],
          outputs=[out],
        )
    wp.launch(
      _qderiv_actuator_passive,
      dim=(d.nworld, qMi.size),
      inputs=[
        m.opt.timestep,
        m.opt.disableflags,
        m.dof_damping,
        m.dof_dampingpoly,
        m.is_sparse,
        d.qvel,
        d.qM,
        qMi,
        qMj,
        out,
      ],
      outputs=[out],
    )
  else:
    # TODO(team): directly utilize qM for these settings
    wp.copy(out, d.qM)

  if not (m.opt.disableflags & DisableBit.DAMPER):
    wp.launch(
      _qderiv_tendon_damping,
      dim=(d.nworld, qMi.size),
      inputs=[
        m.ntendon,
        m.opt.timestep,
        m.ten_J_rownnz,
        m.ten_J_rowadr,
        m.ten_J_colind,
        m.tendon_damping,
        m.tendon_dampingpoly,
        m.is_sparse,
        d.ten_J,
        d.ten_velocity,
        qMi,
        qMj,
      ],
      outputs=[out],
    )

  # TODO(team): rne derivative


@wp.kernel
def _get_state(
  # Model:
  nq: int,
  nv: int,
  na: int,
  # Data in:
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  act_in: wp.array2d[float],
  # Out:
  state_out: wp.array2d[float],
):
  # get state = [qpos, qvel, act]
  worldid = wp.tid()
  for i in range(nq):
    state_out[worldid, i] = qpos_in[worldid, i]
    if i < nv:
      state_out[worldid, nq + i] = qvel_in[worldid, i]
  for i in range(na):
    state_out[worldid, nq + nv + i] = act_in[worldid, i]


@wp.kernel
def _set_state(
  # Model:
  nq: int,
  nv: int,
  na: int,
  # In:
  state_in: wp.array2d[float],
  # Data out:
  qpos_out: wp.array2d[float],
  qvel_out: wp.array2d[float],
  act_out: wp.array2d[float],
):
  # set state = [qpos, qvel, act]
  worldid = wp.tid()
  for i in range(nq):
    qpos_out[worldid, i] = state_in[worldid, i]
    if i < nv:
      qvel_out[worldid, i] = state_in[worldid, nq + i]
  for i in range(na):
    act_out[worldid, i] = state_in[worldid, nq + nv + i]


@wp.kernel
def _state_diff_to_col(
  # Model:
  nq: int,
  nv: int,
  na: int,
  njnt: int,
  jnt_type: wp.array[int],
  jnt_qposadr: wp.array[int],
  jnt_dofadr: wp.array[int],
  # In:
  state1_in: wp.array2d[float],
  state2_in: wp.array2d[float],
  inv_h: float,
  col_idx: int,
  # Out:
  jac_out: wp.array3d[float],
):
  # finite difference two state vectors and write to Jacobian column
  worldid = wp.tid()

  # position difference via joint type
  for jntid in range(njnt):
    jnttype = jnt_type[jntid]
    qpos_adr = jnt_qposadr[jntid]
    dof_adr = jnt_dofadr[jntid]

    if jnttype == JointType.FREE:
      # linear position difference
      jac_out[worldid, dof_adr + 0, col_idx] = (state2_in[worldid, qpos_adr + 0] - state1_in[worldid, qpos_adr + 0]) * inv_h
      jac_out[worldid, dof_adr + 1, col_idx] = (state2_in[worldid, qpos_adr + 1] - state1_in[worldid, qpos_adr + 1]) * inv_h
      jac_out[worldid, dof_adr + 2, col_idx] = (state2_in[worldid, qpos_adr + 2] - state1_in[worldid, qpos_adr + 2]) * inv_h
      # quaternion difference
      q1 = wp.quat(
        state1_in[worldid, qpos_adr + 3],
        state1_in[worldid, qpos_adr + 4],
        state1_in[worldid, qpos_adr + 5],
        state1_in[worldid, qpos_adr + 6],
      )
      q2 = wp.quat(
        state2_in[worldid, qpos_adr + 3],
        state2_in[worldid, qpos_adr + 4],
        state2_in[worldid, qpos_adr + 5],
        state2_in[worldid, qpos_adr + 6],
      )
      dq = math.quat_sub(q2, q1)
      jac_out[worldid, dof_adr + 3, col_idx] = dq[0] * inv_h
      jac_out[worldid, dof_adr + 4, col_idx] = dq[1] * inv_h
      jac_out[worldid, dof_adr + 5, col_idx] = dq[2] * inv_h
    elif jnttype == JointType.BALL:
      q1 = wp.quat(
        state1_in[worldid, qpos_adr + 0],
        state1_in[worldid, qpos_adr + 1],
        state1_in[worldid, qpos_adr + 2],
        state1_in[worldid, qpos_adr + 3],
      )
      q2 = wp.quat(
        state2_in[worldid, qpos_adr + 0],
        state2_in[worldid, qpos_adr + 1],
        state2_in[worldid, qpos_adr + 2],
        state2_in[worldid, qpos_adr + 3],
      )
      dq = math.quat_sub(q2, q1)
      jac_out[worldid, dof_adr + 0, col_idx] = dq[0] * inv_h
      jac_out[worldid, dof_adr + 1, col_idx] = dq[1] * inv_h
      jac_out[worldid, dof_adr + 2, col_idx] = dq[2] * inv_h
    else:  # SLIDE, HINGE
      jac_out[worldid, dof_adr, col_idx] = (state2_in[worldid, qpos_adr] - state1_in[worldid, qpos_adr]) * inv_h

  # velocity and activation difference
  for i in range(nv):
    jac_out[worldid, nv + i, col_idx] = (state2_in[worldid, nq + i] - state1_in[worldid, nq + i]) * inv_h
  for i in range(na):
    jac_out[worldid, 2 * nv + i, col_idx] = (state2_in[worldid, nq + nv + i] - state1_in[worldid, nq + nv + i]) * inv_h


@wp.kernel
def _perturb_position(
  # Model:
  nq: int,
  njnt: int,
  jnt_type: wp.array[int],
  jnt_qposadr: wp.array[int],
  jnt_dofadr: wp.array[int],
  # Data in:
  qpos_in: wp.array2d[float],
  # In:
  dof_idx: int,
  eps: float,
  # Data out:
  qpos_out: wp.array2d[float],
):
  worldid = wp.tid()

  # copy qpos_in to qpos_out
  for i in range(nq):
    qpos_out[worldid, i] = qpos_in[worldid, i]

  # find joint for this dof and perturb
  for jntid in range(njnt):
    jnttype = jnt_type[jntid]
    qpos_adr = jnt_qposadr[jntid]
    dof_adr = jnt_dofadr[jntid]

    if jnttype == JointType.FREE:
      if dof_idx >= dof_adr and dof_idx < dof_adr + 3:
        qpos_out[worldid, qpos_adr + (dof_idx - dof_adr)] += eps
      elif dof_idx >= dof_adr + 3 and dof_idx < dof_adr + 6:
        q = wp.quat(
          qpos_in[worldid, qpos_adr + 3],
          qpos_in[worldid, qpos_adr + 4],
          qpos_in[worldid, qpos_adr + 5],
          qpos_in[worldid, qpos_adr + 6],
        )
        local_idx = dof_idx - dof_adr - 3
        if local_idx == 0:
          v = wp.vec3(1.0, 0.0, 0.0)
        elif local_idx == 1:
          v = wp.vec3(0.0, 1.0, 0.0)
        else:
          v = wp.vec3(0.0, 0.0, 1.0)
        q_new = math.quat_integrate(q, v, eps)
        qpos_out[worldid, qpos_adr + 3] = q_new[0]
        qpos_out[worldid, qpos_adr + 4] = q_new[1]
        qpos_out[worldid, qpos_adr + 5] = q_new[2]
        qpos_out[worldid, qpos_adr + 6] = q_new[3]
    elif jnttype == JointType.BALL:
      if dof_idx >= dof_adr and dof_idx < dof_adr + 3:
        q = wp.quat(
          qpos_in[worldid, qpos_adr + 0],
          qpos_in[worldid, qpos_adr + 1],
          qpos_in[worldid, qpos_adr + 2],
          qpos_in[worldid, qpos_adr + 3],
        )
        local_idx = dof_idx - dof_adr
        if local_idx == 0:
          v = wp.vec3(1.0, 0.0, 0.0)
        elif local_idx == 1:
          v = wp.vec3(0.0, 1.0, 0.0)
        else:
          v = wp.vec3(0.0, 0.0, 1.0)
        q_new = math.quat_integrate(q, v, eps)
        qpos_out[worldid, qpos_adr + 0] = q_new[0]
        qpos_out[worldid, qpos_adr + 1] = q_new[1]
        qpos_out[worldid, qpos_adr + 2] = q_new[2]
        qpos_out[worldid, qpos_adr + 3] = q_new[3]
    else:  # SLIDE, HINGE
      if dof_idx == dof_adr:
        qpos_out[worldid, qpos_adr] += eps


@wp.kernel
def _perturb_array(
  # In:
  idx: int,
  eps: float,
  arr_in: wp.array2d[float],
  # Out:
  arr_out: wp.array2d[float],
):
  worldid = wp.tid()
  for i in range(arr_in.shape[1]):
    if i == idx:
      arr_out[worldid, i] = arr_in[worldid, i] + eps
    else:
      arr_out[worldid, i] = arr_in[worldid, i]


@wp.kernel
def _diff_vectors_to_col(
  # In:
  x1_in: wp.array2d[float],
  x2_in: wp.array2d[float],
  inv_h: float,
  n: int,
  col_idx: int,
  # Out:
  jac_out: wp.array3d[float],
):
  # dx = (x2 - x1) / h, written to Jacobian column
  worldid = wp.tid()
  for i in range(n):
    jac_out[worldid, i, col_idx] = (x2_in[worldid, i] - x1_in[worldid, i]) * inv_h


@event_scope
def transition_fd(
  m: Model,
  d: Data,
  eps: float,
  centered: bool = False,
  A: wp.array3d[float] = None,
  B: wp.array3d[float] = None,
  C: wp.array3d[float] = None,
  D: wp.array3d[float] = None,
):
  """Finite differenced transition matrices (control theory notation).

  Computes: d(x_next) = A*dx + B*du, d(sensor) = C*dx + D*du
  where x = [qvel_diff, qvel, act] is the state in tangent space.

  Args:
    m: model
    d: data
    eps: finite difference epsilon
    centered: if True, use centered differences
    A: output state transition matrix (nworld, ndx, ndx) where ndx = 2*nv+na
    B: output control transition matrix (nworld, ndx, nu)
    C: output state observation matrix (nworld, nsensordata, ndx)
    D: output control observation matrix (nworld, nsensordata, nu)
  """
  if m.opt.integrator == IntegratorType.RK4:
    raise ValueError("RK4 integrator is not supported by transition_fd")

  # TODO(team): add option for scratch memory

  nq, nv, na, nu = m.nq, m.nv, m.na, m.nu
  ns = m.nsensordata
  ndx = 2 * nv + na
  nworld = d.nworld

  # skip sensor computations if not requested
  skip_sensor = C is None and D is None

  # save current state (matches mjSTATE_FULLPHYSICS | mjSTATE_CTRL)
  state_size = nq + nv + na
  state0 = wp.empty((nworld, state_size), dtype=float)
  ctrl0 = wp.empty((nworld, nu), dtype=float) if nu > 0 else None
  warmstart0 = wp.empty((nworld, nv), dtype=float)
  act_dot0 = wp.empty((nworld, na), dtype=float) if na > 0 else None
  time0 = wp.empty(nworld, dtype=float)
  wp.launch(_get_state, dim=nworld, inputs=[nq, nv, na, d.qpos, d.qvel, d.act], outputs=[state0])
  if nu > 0:
    wp.copy(ctrl0, d.ctrl)
  if na > 0:
    wp.copy(act_dot0, d.act_dot)
  wp.copy(warmstart0, d.qacc_warmstart)
  wp.copy(time0, d.time)

  # baseline step
  forward.step(m, d)

  # save baseline next state and sensors
  next_state = wp.empty((nworld, state_size), dtype=float)
  wp.launch(_get_state, dim=nworld, inputs=[nq, nv, na, d.qpos, d.qvel, d.act], outputs=[next_state])
  sensor0 = None
  if not skip_sensor:
    sensor0 = wp.empty((nworld, ns), dtype=float)
    wp.copy(sensor0, d.sensordata)

  # restore state
  wp.launch(_set_state, dim=nworld, inputs=[nq, nv, na, state0], outputs=[d.qpos, d.qvel, d.act])
  if nu > 0:
    wp.copy(d.ctrl, ctrl0)
  if na > 0:
    wp.copy(d.act_dot, act_dot0)
  wp.copy(d.qacc_warmstart, warmstart0)
  wp.copy(d.time, time0)

  # recompute all intermediate quantities (xpos, qfrc_bias, etc.) for the
  # restored state — needed for skip-stage optimization to work correctly
  forward.forward(m, d)

  # allocate work arrays
  # note: next_minus/sensor_minus are always allocated because ctrl clamping
  # may require backward-only differencing even when centered=False
  next_plus = wp.empty((nworld, state_size), dtype=float)
  next_minus = wp.empty((nworld, state_size), dtype=float)
  sensor_plus = wp.empty((nworld, ns), dtype=float) if not skip_sensor else None
  sensor_minus = wp.empty((nworld, ns), dtype=float) if not skip_sensor else None

  inv_eps = 1.0 / eps
  inv_2eps = 1.0 / (2.0 * eps) if centered else inv_eps

  # --- helper: restore full state ---
  def _restore(restore_ctrl=True):
    wp.launch(_set_state, dim=nworld, inputs=[nq, nv, na, state0], outputs=[d.qpos, d.qvel, d.act])
    if restore_ctrl and nu > 0:
      wp.copy(d.ctrl, ctrl0)
    if na > 0:
      wp.copy(d.act_dot, act_dot0)
    wp.copy(d.qacc_warmstart, warmstart0)
    wp.copy(d.time, time0)

  # --- helper: write state/sensor column derivative ---
  def _write_state_col(state_jac, s1, s2, inv_h, col_idx):
    if state_jac is not None:
      wp.launch(
        _state_diff_to_col,
        dim=nworld,
        inputs=[nq, nv, na, m.njnt, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, s1, s2, inv_h, col_idx],
        outputs=[state_jac],
      )

  def _write_sensor_col(sensor_jac, x1, x2, inv_h, col_idx):
    if sensor_jac is not None:
      wp.launch(_diff_vectors_to_col, dim=nworld, inputs=[x1, x2, inv_h, ns, col_idx], outputs=[sensor_jac])

  # --- helper: zero a Jacobian column ---
  def _zero_state_col(state_jac, col_idx):
    if state_jac is not None:
      wp.launch(
        _state_diff_to_col,
        dim=nworld,
        inputs=[nq, nv, na, m.njnt, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, next_state, next_state, inv_eps, col_idx],
        outputs=[state_jac],
      )

  def _zero_sensor_col(sensor_jac, col_idx):
    if sensor_jac is not None and sensor0 is not None:
      wp.launch(_diff_vectors_to_col, dim=nworld, inputs=[sensor0, sensor0, inv_eps, ns, col_idx], outputs=[sensor_jac])

  # --- ctrl clamping helper: check if nudged ctrl is in range ---
  # read ctrl limits on host (only used when B or D is requested and nu > 0)
  ctrl_limited = None
  ctrl_range = None
  ctrl_host = None
  if (B is not None or D is not None) and nu > 0:
    ctrl_limited = m.actuator_ctrllimited.numpy()
    ctrl_range = m.actuator_ctrlrange.numpy()
    ctrl_host = ctrl0.numpy()

  def _ctrl_in_range(ctrl_val, nudged_val, idx):
    """Check if both ctrl_val and nudged_val are inside ctrlrange for actuator idx."""
    if not ctrl_limited[idx]:
      return True
    lo, hi = ctrl_range[0, idx, 0], ctrl_range[0, idx, 1]
    return lo <= ctrl_val <= hi and lo <= nudged_val <= hi

  # finite difference controls: skipstage = STAGE_VEL
  if (B is not None or D is not None) and nu > 0:
    ctrl_temp = wp.empty((nworld, nu), dtype=float)
    for i in range(nu):
      c = ctrl_host[0, i]

      # check if forward/backward nudges are in range
      nudge_fwd = _ctrl_in_range(c, c + eps, i)
      nudge_back = (centered or not nudge_fwd) and _ctrl_in_range(c - eps, c, i)

      if nudge_fwd:
        # nudge forward
        wp.launch(_perturb_array, dim=nworld, inputs=[i, eps, ctrl0], outputs=[ctrl_temp])
        wp.copy(d.ctrl, ctrl_temp)
        forward.step_skip(m, d, forward._STAGE_VEL, skip_sensor)
        wp.launch(_get_state, dim=nworld, inputs=[nq, nv, na, d.qpos, d.qvel, d.act], outputs=[next_plus])
        if not skip_sensor:
          wp.copy(sensor_plus, d.sensordata)
        _restore()

      if nudge_back:
        # nudge backward
        wp.launch(_perturb_array, dim=nworld, inputs=[i, -eps, ctrl0], outputs=[ctrl_temp])
        wp.copy(d.ctrl, ctrl_temp)
        forward.step_skip(m, d, forward._STAGE_VEL, skip_sensor)
        wp.launch(_get_state, dim=nworld, inputs=[nq, nv, na, d.qpos, d.qvel, d.act], outputs=[next_minus])
        if not skip_sensor:
          wp.copy(sensor_minus, d.sensordata)
        _restore()

      # compute derivatives using available nudges
      if nudge_fwd and not nudge_back:
        # forward-only differencing
        _write_state_col(B, next_state, next_plus, inv_eps, i)
        _write_sensor_col(D, sensor0, sensor_plus, inv_eps, i)
      elif not nudge_fwd and nudge_back:
        # backward-only differencing
        _write_state_col(B, next_minus, next_state, inv_eps, i)
        _write_sensor_col(D, sensor_minus, sensor0, inv_eps, i)
      elif nudge_fwd and nudge_back:
        # centered differencing
        _write_state_col(B, next_minus, next_plus, inv_2eps, i)
        _write_sensor_col(D, sensor_plus, sensor_minus, inv_2eps, i)
      else:
        # both nudges failed (ctrl clamped at both limits) — write zeros
        _zero_state_col(B, i)
        _zero_sensor_col(D, i)

  # finite difference activations: skipstage = STAGE_VEL
  if (A is not None or C is not None) and na > 0:
    act0 = wp.empty((nworld, na), dtype=float)
    wp.copy(act0, d.act)
    act_temp = wp.empty((nworld, na), dtype=float)
    for i in range(na):
      # nudge forward
      wp.launch(_perturb_array, dim=nworld, inputs=[i, eps, act0], outputs=[act_temp])
      wp.copy(d.act, act_temp)
      forward.step_skip(m, d, forward._STAGE_VEL, skip_sensor)
      wp.launch(_get_state, dim=nworld, inputs=[nq, nv, na, d.qpos, d.qvel, d.act], outputs=[next_plus])
      if not skip_sensor:
        wp.copy(sensor_plus, d.sensordata)

      # restore
      _restore(restore_ctrl=False)

      if centered:
        wp.launch(_perturb_array, dim=nworld, inputs=[i, -eps, act0], outputs=[act_temp])
        wp.copy(d.act, act_temp)
        forward.step_skip(m, d, forward._STAGE_VEL, skip_sensor)
        wp.launch(_get_state, dim=nworld, inputs=[nq, nv, na, d.qpos, d.qvel, d.act], outputs=[next_minus])
        if not skip_sensor:
          wp.copy(sensor_minus, d.sensordata)
        _restore(restore_ctrl=False)

      # compute derivatives
      col_idx = 2 * nv + i
      if centered:
        _write_state_col(A, next_minus, next_plus, inv_2eps, col_idx)
        _write_sensor_col(C, sensor_minus, sensor_plus, inv_2eps, col_idx)
      else:
        _write_state_col(A, next_state, next_plus, inv_eps, col_idx)
        _write_sensor_col(C, sensor0, sensor_plus, inv_eps, col_idx)

  # finite difference velocities: skipstage = STAGE_POS
  if A is not None or C is not None:
    qvel0 = wp.empty((nworld, nv), dtype=float)
    wp.copy(qvel0, d.qvel)
    qvel_temp = wp.empty((nworld, nv), dtype=float)
    for i in range(nv):
      # nudge forward
      wp.launch(_perturb_array, dim=nworld, inputs=[i, eps, qvel0], outputs=[qvel_temp])
      wp.copy(d.qvel, qvel_temp)
      forward.step_skip(m, d, forward._STAGE_POS, skip_sensor)
      wp.launch(_get_state, dim=nworld, inputs=[nq, nv, na, d.qpos, d.qvel, d.act], outputs=[next_plus])
      if not skip_sensor:
        wp.copy(sensor_plus, d.sensordata)

      # restore
      _restore(restore_ctrl=False)

      if centered:
        wp.launch(_perturb_array, dim=nworld, inputs=[i, -eps, qvel0], outputs=[qvel_temp])
        wp.copy(d.qvel, qvel_temp)
        forward.step_skip(m, d, forward._STAGE_POS, skip_sensor)
        wp.launch(_get_state, dim=nworld, inputs=[nq, nv, na, d.qpos, d.qvel, d.act], outputs=[next_minus])
        if not skip_sensor:
          wp.copy(sensor_minus, d.sensordata)
        _restore(restore_ctrl=False)

      # compute derivatives
      col_idx = nv + i
      if centered:
        _write_state_col(A, next_minus, next_plus, inv_2eps, col_idx)
        _write_sensor_col(C, sensor_minus, sensor_plus, inv_2eps, col_idx)
      else:
        _write_state_col(A, next_state, next_plus, inv_eps, col_idx)
        _write_sensor_col(C, sensor0, sensor_plus, inv_eps, col_idx)

  # finite difference positions: skipstage = STAGE_NONE
  if A is not None or C is not None:
    qpos_perturbed = wp.empty((nworld, nq), dtype=float)
    for i in range(nv):
      # nudge position forward
      wp.launch(
        _perturb_position,
        dim=nworld,
        inputs=[nq, m.njnt, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, i, eps],
        outputs=[qpos_perturbed],
      )
      wp.copy(d.qpos, qpos_perturbed)
      forward.step_skip(m, d, forward._STAGE_NONE, skip_sensor)
      wp.launch(_get_state, dim=nworld, inputs=[nq, nv, na, d.qpos, d.qvel, d.act], outputs=[next_plus])
      if not skip_sensor:
        wp.copy(sensor_plus, d.sensordata)

      # restore
      _restore(restore_ctrl=False)

      if centered:
        wp.launch(
          _perturb_position,
          dim=nworld,
          inputs=[nq, m.njnt, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, i, -eps],
          outputs=[qpos_perturbed],
        )
        wp.copy(d.qpos, qpos_perturbed)
        forward.step_skip(m, d, forward._STAGE_NONE, skip_sensor)
        wp.launch(_get_state, dim=nworld, inputs=[nq, nv, na, d.qpos, d.qvel, d.act], outputs=[next_minus])
        if not skip_sensor:
          wp.copy(sensor_minus, d.sensordata)
        _restore(restore_ctrl=False)

      # compute derivatives
      col_idx = i
      if centered:
        _write_state_col(A, next_minus, next_plus, inv_2eps, col_idx)
        _write_sensor_col(C, sensor_minus, sensor_plus, inv_2eps, col_idx)
      else:
        _write_state_col(A, next_state, next_plus, inv_eps, col_idx)
        _write_sensor_col(C, sensor0, sensor_plus, inv_eps, col_idx)

  # restore final state
  _restore()

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

import dataclasses
from math import ceil
from math import sqrt

import warp as wp

from mujoco_warp._src import math
from mujoco_warp._src import smooth
from mujoco_warp._src import support
from mujoco_warp._src import types
from mujoco_warp._src import util_solve
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_func
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_solve_func
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope
from mujoco_warp._src.warp_util import scoped_mathdx_gemm_disabled

wp.set_module_options({"enable_backward": False})

_BLOCK_CHOLESKY_DIM = 32


@dataclasses.dataclass
class InverseContext:
  """Workspace arrays for inverse dynamics."""

  Jaref: wp.array2d[float]
  search_dot: wp.array[float]
  gauss: wp.array[float]
  cost: wp.array[float]
  prev_cost: wp.array[float]
  done: wp.array[bool]
  changed_efc_ids: wp.array2d[int]
  changed_efc_count: wp.array[int]


@dataclasses.dataclass
class SolverContext:
  """Workspace arrays for constraint solver."""

  Jaref: wp.array2d[float]
  search_dot: wp.array[float]
  gauss: wp.array[float]
  cost: wp.array[float]
  prev_cost: wp.array[float]
  done: wp.array[bool]
  grad: wp.array2d[float]
  grad_dot: wp.array[float]
  Mgrad: wp.array2d[float]
  search: wp.array2d[float]
  mv: wp.array2d[float]
  jv: wp.array2d[float]
  quad: wp.array2d[wp.vec3]
  quad_gauss: wp.array[wp.vec3]
  alpha: wp.array[float]
  prev_grad: wp.array2d[float]
  prev_Mgrad: wp.array2d[float]
  beta: wp.array[float]
  h: wp.array3d[float]
  hfactor: wp.array3d[float]
  # Incremental Hessian update (Newton only)
  changed_efc_ids: wp.array2d[int]
  changed_efc_count: wp.array[int]


def create_inverse_context(m: types.Model, d: types.Data) -> InverseContext:
  """Create an InverseContext with allocated workspace arrays.

  Args:
    m: Model containing nv, nv_pad, and solver type.
    d: Data containing nworld and njmax.

  Returns:
    InverseContext with allocated arrays.
  """
  nworld = d.nworld
  njmax = d.njmax

  return InverseContext(
    Jaref=wp.empty((nworld, njmax), dtype=float),
    search_dot=wp.empty((nworld,), dtype=float),
    gauss=wp.empty((nworld,), dtype=float),
    cost=wp.empty((nworld,), dtype=float),
    prev_cost=wp.empty((nworld,), dtype=float),
    done=wp.empty((nworld,), dtype=bool),
    changed_efc_ids=wp.empty((nworld, 0), dtype=int),
    changed_efc_count=wp.empty((0,), dtype=int),
  )


def create_solver_context(m: types.Model, d: types.Data) -> SolverContext:
  """Create a SolverContext with allocated workspace arrays.

  Args:
    m: Model containing nv, nv_pad, and solver type.
    d: Data containing nworld and njmax.

  Returns:
    SolverContext with allocated arrays.
  """
  nworld = d.nworld
  nv = m.nv
  nv_pad = m.nv_pad
  njmax = d.njmax

  alloc_h = m.opt.solver == types.SolverType.NEWTON
  alloc_hfactor = alloc_h and nv > _BLOCK_CHOLESKY_DIM

  return SolverContext(
    Jaref=wp.empty((nworld, njmax), dtype=float),
    search_dot=wp.empty((nworld,), dtype=float),
    gauss=wp.empty((nworld,), dtype=float),
    cost=wp.empty((nworld,), dtype=float),
    prev_cost=wp.empty((nworld,), dtype=float),
    done=wp.empty((nworld,), dtype=bool),
    grad=wp.zeros((nworld, nv_pad), dtype=float),
    grad_dot=wp.empty((nworld,), dtype=float),
    Mgrad=wp.zeros((nworld, nv_pad), dtype=float),
    search=wp.empty((nworld, nv), dtype=float),
    mv=wp.empty((nworld, nv), dtype=float),
    jv=wp.empty((nworld, njmax), dtype=float),
    quad=wp.empty((nworld, njmax), dtype=wp.vec3),
    quad_gauss=wp.empty((nworld,), dtype=wp.vec3),
    alpha=wp.empty((nworld,), dtype=float),
    prev_grad=wp.empty((nworld, nv), dtype=float),
    prev_Mgrad=wp.empty((nworld, nv), dtype=float),
    beta=wp.empty((nworld,), dtype=float),
    h=wp.zeros((nworld, nv_pad, nv_pad), dtype=float) if alloc_h else wp.empty((nworld, 0, 0), dtype=float),
    hfactor=wp.zeros((nworld, nv_pad, nv_pad), dtype=float) if alloc_hfactor else wp.empty((nworld, 0, 0), dtype=float),
    changed_efc_ids=wp.empty((nworld, njmax), dtype=int) if alloc_h else wp.empty((nworld, 0), dtype=int),
    changed_efc_count=wp.empty((nworld,), dtype=int) if alloc_h else wp.empty((0,), dtype=int),
  )


@wp.func
def _rescale(nv: int, meaninertia: float, value: float) -> float:
  return value / (meaninertia * float(nv))


@wp.func
def _in_bracket(x: wp.vec3, y: wp.vec3) -> bool:
  return (x[1] < y[1] and y[1] < 0.0) or (x[1] > y[1] and y[1] > 0.0)


@wp.func
def _eval_pt_direct(jaref: float, jv: float, d: float, alpha: float) -> wp.vec3:
  """Eval quadratic constraint, return (cost, grad, hessian)."""
  x = jaref + alpha * jv
  jvD = jv * d
  return wp.vec3(0.5 * d * x * x, jvD * x, jv * jvD)


@wp.func
def _eval_pt_direct_alpha_zero(jaref: float, jv: float, d: float) -> wp.vec3:
  """Eval quadratic constraint at alpha=0."""
  jvD = jv * d
  return wp.vec3(0.5 * d * jaref * jaref, jvD * jaref, jv * jvD)


@wp.func
def _eval_pt_direct_3alphas(
  jaref: float, jv: float, d: float, lo_alpha: float, hi_alpha: float, mid_alpha: float
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  """Eval quadratic constraint for 3 alphas."""
  x_lo = jaref + lo_alpha * jv
  x_hi = jaref + hi_alpha * jv
  x_mid = jaref + mid_alpha * jv
  jvD = jv * d
  hessian = jv * jvD
  half_d = 0.5 * d
  return (
    wp.vec3(half_d * x_lo * x_lo, jvD * x_lo, hessian),
    wp.vec3(half_d * x_hi * x_hi, jvD * x_hi, hessian),
    wp.vec3(half_d * x_mid * x_mid, jvD * x_mid, hessian),
  )


@wp.func
def _eval_cost(quad: wp.vec3, alpha: float) -> float:
  return alpha * alpha * quad[2] + alpha * quad[1] + quad[0]


@wp.func
def _eval_pt(quad: wp.vec3, alpha: float) -> wp.vec3:
  """Eval quad polynomial at alpha, return (cost, grad, hessian)."""
  aq2 = alpha * quad[2]
  return wp.vec3(
    alpha * aq2 + alpha * quad[1] + quad[0],
    2.0 * aq2 + quad[1],
    2.0 * quad[2],
  )


@wp.func
def _eval_pt_3alphas(quad: wp.vec3, lo_alpha: float, hi_alpha: float, mid_alpha: float) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  """Eval quad polynomial for 3 alphas."""
  q0, q1, q2 = quad[0], quad[1], quad[2]
  hessian = 2.0 * q2
  lo_aq2 = lo_alpha * q2
  hi_aq2 = hi_alpha * q2
  mid_aq2 = mid_alpha * q2
  return (
    wp.vec3(lo_alpha * lo_aq2 + lo_alpha * q1 + q0, 2.0 * lo_aq2 + q1, hessian),
    wp.vec3(hi_alpha * hi_aq2 + hi_alpha * q1 + q0, 2.0 * hi_aq2 + q1, hessian),
    wp.vec3(mid_alpha * mid_aq2 + mid_alpha * q1 + q0, 2.0 * mid_aq2 + q1, hessian),
  )


@wp.func
def _eval_frictionloss_pt(x: float, f: float, rf: float, jv: float, d: float) -> wp.vec3:
  """Eval frictionloss and return (cost, grad, hessian). x = Jaref + alpha * jv."""
  if (-rf < x) and (x < rf):
    jvD = jv * d
    return wp.vec3(0.5 * d * x * x, jvD * x, jv * jvD)
  elif x <= -rf:
    return wp.vec3(f * (-0.5 * rf - x), -f * jv, 0.0)
  else:
    return wp.vec3(f * (-0.5 * rf + x), f * jv, 0.0)


@wp.func
def _eval_frictionloss_pt_one(x: float, f: float, rf: float, half_d: float, jvD: float, hessian: float, f_jv: float) -> wp.vec3:
  """Eval frictionloss with precomputed shared values."""
  if (-rf < x) and (x < rf):
    return wp.vec3(half_d * x * x, jvD * x, hessian)
  elif x <= -rf:
    return wp.vec3(f * (-0.5 * rf - x), -f_jv, 0.0)
  else:
    return wp.vec3(f * (-0.5 * rf + x), f_jv, 0.0)


@wp.func
def _eval_frictionloss_pt_3alphas(
  x_lo: float, x_hi: float, x_mid: float, f: float, rf: float, jv: float, d: float
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  """Eval frictionloss for 3 x values with shared precomputation."""
  jvD = jv * d
  half_d = 0.5 * d
  hessian = jv * jvD
  f_jv = f * jv
  return (
    _eval_frictionloss_pt_one(x_lo, f, rf, half_d, jvD, hessian, f_jv),
    _eval_frictionloss_pt_one(x_hi, f, rf, half_d, jvD, hessian, f_jv),
    _eval_frictionloss_pt_one(x_mid, f, rf, half_d, jvD, hessian, f_jv),
  )


@wp.func
def _eval_elliptic(
  # In:
  impratio_invsqrt: float,
  friction: types.vec5,
  quad: wp.vec3,
  quad1: wp.vec3,
  quad2: wp.vec3,
  alpha: float,
) -> wp.vec3:
  mu = friction[0] * impratio_invsqrt

  u0 = quad1[0]
  v0 = quad1[1]
  uu = quad1[2]
  uv = quad2[0]
  vv = quad2[1]
  dm = quad2[2]

  # compute N, Tsqr
  N = u0 + alpha * v0
  Tsqr = uu + alpha * (2.0 * uv + alpha * vv)

  # no tangential force: top or bottom zone
  if Tsqr <= 0.0:
    # bottom zone: quadratic cost
    if N < 0.0:
      return _eval_pt(quad, alpha)

    # top zone: nothing to do
  # otherwise regular processing
  else:
    # tangential force
    T = wp.sqrt(Tsqr)

    # N >= mu * T : top zone
    if N >= mu * T:
      # nothing to do
      pass
    # mu * N + T <= 0 : bottom zone
    elif mu * N + T <= 0.0:
      return _eval_pt(quad, alpha)

    # otherwise middle zone
    else:
      # derivatives
      N1 = v0
      T1 = (uv + alpha * vv) / T
      T2 = vv / T - (uv + alpha * vv) * T1 / (T * T)

      # add to cost
      cost = wp.vec3(
        0.5 * dm * (N - mu * T) * (N - mu * T),
        dm * (N - mu * T) * (N1 - mu * T1),
        dm * ((N1 - mu * T1) * (N1 - mu * T1) + (N - mu * T) * (-mu * T2)),
      )

      return cost

  return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def _log_scale(min_value: float, max_value: float, num_values: int, i: int) -> float:
  step = (wp.log(max_value) - wp.log(min_value)) / wp.max(1.0, float(num_values - 1))
  return wp.exp(wp.log(min_value) + float(i) * step)


@wp.kernel
def linesearch_parallel_fused(
  # Model:
  opt_ls_iterations: int,
  opt_impratio_invsqrt: wp.array[float],
  opt_ls_parallel_min_step: float,
  # Data in:
  ne_in: wp.array[int],
  nf_in: wp.array[int],
  nefc_in: wp.array[int],
  contact_friction_in: wp.array[types.vec5],
  contact_efc_address_in: wp.array2d[int],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  efc_D_in: wp.array2d[float],
  efc_frictionloss_in: wp.array2d[float],
  njmax_in: int,
  nacon_in: wp.array[int],
  # In:
  ctx_Jaref_in: wp.array2d[float],
  ctx_jv_in: wp.array2d[float],
  ctx_quad_in: wp.array2d[wp.vec3],
  ctx_quad_gauss_in: wp.array[wp.vec3],
  ctx_done_in: wp.array[bool],
  # Out:
  cost_out: wp.array2d[float],
):
  worldid, alphaid = wp.tid()

  if ctx_done_in[worldid]:
    return

  alpha = _log_scale(opt_ls_parallel_min_step, 1.0, opt_ls_iterations, alphaid)

  out = _eval_cost(ctx_quad_gauss_in[worldid], alpha)

  ne = ne_in[worldid]
  nf = nf_in[worldid]

  # TODO(team): _eval with option to only compute cost
  for efcid in range(min(njmax_in, nefc_in[worldid])):
    # equality
    if efcid < ne:
      out += _eval_cost(ctx_quad_in[worldid, efcid], alpha)
    # friction
    elif efcid < ne + nf:
      # search point, friction loss, bound (rf)
      start = ctx_Jaref_in[worldid, efcid]
      dir = ctx_jv_in[worldid, efcid]
      x = start + alpha * dir
      f = efc_frictionloss_in[worldid, efcid]
      rf = math.safe_div(f, efc_D_in[worldid, efcid])

      # -bound < x < bound : quadratic
      if (-rf < x) and (x < rf):
        quad = ctx_quad_in[worldid, efcid]
      # x < -bound: linear negative
      elif x <= -rf:
        quad = wp.vec3(f * (-0.5 * rf - start), -f * dir, 0.0)
      # bound < x : linear positive
      else:
        quad = wp.vec3(f * (-0.5 * rf + start), f * dir, 0.0)

      out += _eval_cost(quad, alpha)
    # limit and contact
    elif efc_type_in[worldid, efcid] == types.ConstraintType.CONTACT_ELLIPTIC:
      # extract contact info
      conid = efc_id_in[worldid, efcid]

      if conid >= nacon_in[0]:
        continue

      efcid0 = contact_efc_address_in[conid, 0]
      if efcid != efcid0:
        continue

      friction = contact_friction_in[conid]
      mu = friction[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]

      # unpack quad
      efcid1 = contact_efc_address_in[conid, 1]
      efcid2 = contact_efc_address_in[conid, 2]
      u0 = ctx_quad_in[worldid, efcid1][0]
      v0 = ctx_quad_in[worldid, efcid1][1]
      uu = ctx_quad_in[worldid, efcid1][2]
      uv = ctx_quad_in[worldid, efcid2][0]
      vv = ctx_quad_in[worldid, efcid2][1]
      dm = ctx_quad_in[worldid, efcid2][2]

      # compute N, Tsqr
      N = u0 + alpha * v0
      Tsqr = uu + alpha * (2.0 * uv + alpha * vv)

      # no tangential force: top or bottom zone
      if Tsqr <= 0.0:
        # bottom zone: quadratic cost
        if N < 0.0:
          out += _eval_cost(ctx_quad_in[worldid, efcid], alpha)
      # otherwise regular processing
      else:
        # tangential force
        T = wp.sqrt(Tsqr)

        # N >= mu * T : top zone
        if N >= mu * T:
          # nothing to do
          pass
        # mu * N + T <= 0 : bottom zone
        elif mu * N + T <= 0.0:
          out += _eval_cost(ctx_quad_in[worldid, efcid], alpha)
        # otherwise middle zone
        else:
          out += 0.5 * dm * (N - mu * T) * (N - mu * T)
    else:
      # search point
      x = ctx_Jaref_in[worldid, efcid] + alpha * ctx_jv_in[worldid, efcid]

      # active
      if x < 0.0:
        out += _eval_cost(ctx_quad_in[worldid, efcid], alpha)

  cost_out[worldid, alphaid] = out


@wp.kernel
def linesearch_parallel_best_alpha(
  # Model:
  opt_ls_iterations: int,
  opt_ls_parallel_min_step: float,
  # In:
  ctx_done_in: wp.array[bool],
  cost_in: wp.array2d[float],
  # Out:
  ctx_alpha_out: wp.array[float],
):
  worldid = wp.tid()

  if ctx_done_in[worldid]:
    return

  bestid = int(0)
  best_cost = float(types.MJ_MAXVAL)
  for i in range(opt_ls_iterations):
    cost = cost_in[worldid, i]
    if cost < best_cost:
      best_cost = cost
      bestid = i

  ctx_alpha_out[worldid] = _log_scale(opt_ls_parallel_min_step, 1.0, opt_ls_iterations, bestid)


def _linesearch_parallel(m: types.Model, d: types.Data, ctx: SolverContext, cost: wp.array2d[float]):
  """Parallel linesearch with setup and teardown kernels."""
  dofs_per_thread = 20 if m.nv > 50 else 50
  threads_per_efc = ceil(m.nv / dofs_per_thread)

  # quad_gauss = [gauss, search.T @ Ma - search.T @ qfrc_smooth, 0.5 * search.T @ mv]
  if threads_per_efc > 1:
    ctx.quad_gauss.zero_()

  wp.launch(
    linesearch_prepare_gauss(m.nv, dofs_per_thread),
    dim=(d.nworld, threads_per_efc),
    inputs=[d.qfrc_smooth, d.efc.Ma, ctx.search, ctx.gauss, ctx.mv, ctx.done],
    outputs=[ctx.quad_gauss],
  )

  # quad = [0.5 * Jaref * Jaref * efc_D, jv * Jaref * efc_D, 0.5 * jv * jv * efc_D]

  wp.launch(
    linesearch_prepare_quad,
    dim=(d.nworld, d.njmax),
    inputs=[
      m.opt.impratio_invsqrt,
      d.nefc,
      d.contact.friction,
      d.contact.dim,
      d.contact.efc_address,
      d.efc.type,
      d.efc.id,
      d.efc.D,
      d.nacon,
      ctx.Jaref,
      ctx.jv,
      ctx.done,
    ],
    outputs=[ctx.quad],
  )

  wp.launch(
    linesearch_parallel_fused,
    dim=(d.nworld, m.opt.ls_iterations),
    inputs=[
      m.opt.ls_iterations,
      m.opt.impratio_invsqrt,
      m.opt.ls_parallel_min_step,
      d.ne,
      d.nf,
      d.nefc,
      d.contact.friction,
      d.contact.efc_address,
      d.efc.type,
      d.efc.id,
      d.efc.D,
      d.efc.frictionloss,
      d.njmax,
      d.nacon,
      ctx.Jaref,
      ctx.jv,
      ctx.quad,
      ctx.quad_gauss,
      ctx.done,
    ],
    outputs=[cost],
  )

  wp.launch(
    linesearch_parallel_best_alpha,
    dim=(d.nworld),
    inputs=[m.opt.ls_iterations, m.opt.ls_parallel_min_step, ctx.done, cost],
    outputs=[ctx.alpha],
  )

  # Teardown: update qacc, Ma, Jaref
  wp.launch(
    linesearch_qacc_ma,
    dim=(d.nworld, m.nv),
    inputs=[ctx.search, ctx.mv, ctx.alpha, ctx.done],
    outputs=[d.qacc, d.efc.Ma],
  )

  wp.launch(
    linesearch_jaref,
    dim=(d.nworld, d.njmax),
    inputs=[d.nefc, ctx.jv, ctx.alpha, ctx.done],
    outputs=[ctx.Jaref],
  )


@wp.func
def _compute_efc_eval_pt_pyramidal(  # kernel_analyzer: ignore
  # In:
  efcid: int,
  alpha: float,
  ne: int,  # kernel_analyzer: ignore
  nf: int,  # kernel_analyzer: ignore
  efc_D: float,  # kernel_analyzer: ignore
  efc_frictionloss: wp.array[float],  # kernel_analyzer: ignore
  ctx_Jaref: float,
  ctx_jv: float,
) -> wp.vec3:
  """Compute for pyramidal cones (no elliptic contact data needed)."""
  # Limit/other constraint
  if efcid >= ne + nf:
    x = ctx_Jaref + alpha * ctx_jv
    if x < 0.0:
      return _eval_pt_direct(ctx_Jaref, ctx_jv, efc_D, alpha)
    return wp.vec3(0.0)

  # Friction constraint - needs quad for frictionloss computation
  if efcid >= ne:
    f = efc_frictionloss[efcid]
    x = ctx_Jaref + alpha * ctx_jv
    rf = math.safe_div(f, efc_D)
    return _eval_frictionloss_pt(x, f, rf, ctx_jv, efc_D)

  # Equality constraint
  return _eval_pt_direct(ctx_Jaref, ctx_jv, efc_D, alpha)


@wp.func
def _compute_efc_eval_pt_elliptic(  # kernel_analyzer: ignore
  # In:
  efcid: int,
  alpha: float,
  ne: int,  # kernel_analyzer: ignore
  nf: int,  # kernel_analyzer: ignore
  impratio_invsqrt: float,
  efc_type: int,  # kernel_analyzer: ignore
  efc_D_in: wp.array[float],  # kernel_analyzer: ignore
  efc_frictionloss: wp.array[float],  # kernel_analyzer: ignore
  ctx_Jaref: float,
  ctx_jv: float,
  ctx_quad: wp.vec3,
  contact_friction: types.vec5,  # kernel_analyzer: ignore
  efc_address0: int,
  quad1: wp.vec3,
  quad2: wp.vec3,
) -> wp.vec3:
  """Compute for elliptic cones (includes elliptic contact data)."""
  # Contact/limit/other constraints
  if efcid >= ne + nf:
    # Contact elliptic
    if efc_type == types.ConstraintType.CONTACT_ELLIPTIC:
      if efcid != efc_address0:  # Not primary row
        return wp.vec3(0.0)
      return _eval_elliptic(impratio_invsqrt, contact_friction, ctx_quad, quad1, quad2, alpha)

    # Limit/other constraint — direct eval (no quad read)
    x = ctx_Jaref + alpha * ctx_jv
    if x < 0.0:
      return _eval_pt_direct(ctx_Jaref, ctx_jv, efc_D_in[efcid], alpha)
    return wp.vec3(0.0)

  # Friction constraint - load D and frictionloss only here
  if efcid >= ne:
    efc_D = efc_D_in[efcid]
    f = efc_frictionloss[efcid]
    x = ctx_Jaref + alpha * ctx_jv
    rf = math.safe_div(f, efc_D)
    return _eval_frictionloss_pt(x, f, rf, ctx_jv, efc_D)

  # Equality constraint — direct eval (no quad read)
  return _eval_pt_direct(ctx_Jaref, ctx_jv, efc_D_in[efcid], alpha)


@wp.func
def _compute_efc_eval_pt_alpha_zero_pyramidal(  # kernel_analyzer: ignore
  # In:
  efcid: int,
  ne: int,  # kernel_analyzer: ignore
  nf: int,  # kernel_analyzer: ignore
  efc_D: float,  # kernel_analyzer: ignore
  efc_frictionloss: wp.array[float],  # kernel_analyzer: ignore
  ctx_Jaref: float,
  ctx_jv: float,
) -> wp.vec3:
  """Optimized version for alpha=0.0, pyramidal cones."""
  # Limit/other constraint
  if efcid >= ne + nf:
    if ctx_Jaref < 0.0:
      return _eval_pt_direct_alpha_zero(ctx_Jaref, ctx_jv, efc_D)
    return wp.vec3(0.0)

  # Friction constraint - needs quad for frictionloss computation
  if efcid >= ne:
    f = efc_frictionloss[efcid]
    rf = math.safe_div(f, efc_D)
    return _eval_frictionloss_pt(ctx_Jaref, f, rf, ctx_jv, efc_D)

  # Equality constraint
  return _eval_pt_direct_alpha_zero(ctx_Jaref, ctx_jv, efc_D)


@wp.func
def _compute_efc_eval_pt_alpha_zero_elliptic(  # kernel_analyzer: ignore
  # In:
  efcid: int,
  ne: int,  # kernel_analyzer: ignore
  nf: int,  # kernel_analyzer: ignore
  impratio_invsqrt: float,
  efc_type: int,  # kernel_analyzer: ignore
  efc_D_in: wp.array[float],  # kernel_analyzer: ignore
  efc_frictionloss: wp.array[float],  # kernel_analyzer: ignore
  ctx_Jaref: float,
  ctx_jv: float,
  ctx_quad: wp.vec3,
  contact_friction: types.vec5,  # kernel_analyzer: ignore
  efc_address0: int,
  quad1: wp.vec3,
  quad2: wp.vec3,
) -> wp.vec3:
  """Optimized version for alpha=0.0, elliptic cones."""
  # Contact/limit/other constraints
  if efcid >= ne + nf:
    # Contact elliptic
    if efc_type == types.ConstraintType.CONTACT_ELLIPTIC:
      if efcid != efc_address0:  # Not primary row
        return wp.vec3(0.0)
      return _eval_elliptic(impratio_invsqrt, contact_friction, ctx_quad, quad1, quad2, 0.0)

    # Limit/other constraint — direct eval (no quad read)
    if ctx_Jaref < 0.0:
      return _eval_pt_direct_alpha_zero(ctx_Jaref, ctx_jv, efc_D_in[efcid])
    return wp.vec3(0.0)

  # Friction constraint - load D and frictionloss only here
  if efcid >= ne:
    efc_D = efc_D_in[efcid]
    f = efc_frictionloss[efcid]
    rf = math.safe_div(f, efc_D)
    return _eval_frictionloss_pt(ctx_Jaref, f, rf, ctx_jv, efc_D)

  # Equality constraint — direct eval (no quad read)
  return _eval_pt_direct_alpha_zero(ctx_Jaref, ctx_jv, efc_D_in[efcid])


@wp.func
def _compute_efc_eval_pt_3alphas_pyramidal(  # kernel_analyzer: ignore
  # In:
  efcid: int,
  lo_alpha: float,
  hi_alpha: float,
  mid_alpha: float,
  ne: int,  # kernel_analyzer: ignore
  nf: int,  # kernel_analyzer: ignore
  efc_D: float,  # kernel_analyzer: ignore
  efc_frictionloss: wp.array[float],  # kernel_analyzer: ignore
  ctx_Jaref: float,
  ctx_jv: float,
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  """Compute (cost, gradient, hessian) for 3 alphas, pyramidal cones.

  Returns a tuple of 3 vec3s for (lo_alpha, hi_alpha, mid_alpha).
  Constraint types checked in order: limit/other -> friction -> equality.
  """
  # Limit/other constraints: active only when x < 0
  if efcid >= ne + nf:
    x_lo = ctx_Jaref + lo_alpha * ctx_jv
    x_hi = ctx_Jaref + hi_alpha * ctx_jv
    x_mid = ctx_Jaref + mid_alpha * ctx_jv
    pt_lo, pt_hi, pt_mid = _eval_pt_direct_3alphas(ctx_Jaref, ctx_jv, efc_D, lo_alpha, hi_alpha, mid_alpha)
    r_lo = wp.where(x_lo < 0.0, pt_lo, wp.vec3(0.0))
    r_hi = wp.where(x_hi < 0.0, pt_hi, wp.vec3(0.0))
    r_mid = wp.where(x_mid < 0.0, pt_mid, wp.vec3(0.0))
    return (r_lo, r_hi, r_mid)

  # Friction constraint - needs quad for frictionloss computation
  if efcid >= ne:
    x_lo = ctx_Jaref + lo_alpha * ctx_jv
    x_hi = ctx_Jaref + hi_alpha * ctx_jv
    x_mid = ctx_Jaref + mid_alpha * ctx_jv
    f = efc_frictionloss[efcid]
    rf = math.safe_div(f, efc_D)
    return _eval_frictionloss_pt_3alphas(x_lo, x_hi, x_mid, f, rf, ctx_jv, efc_D)

  # Equality constraint: always active
  return _eval_pt_direct_3alphas(ctx_Jaref, ctx_jv, efc_D, lo_alpha, hi_alpha, mid_alpha)


@wp.func
def _compute_efc_eval_pt_3alphas_elliptic(  # kernel_analyzer: ignore
  # In:
  efcid: int,
  lo_alpha: float,
  hi_alpha: float,
  mid_alpha: float,
  ne: int,  # kernel_analyzer: ignore
  nf: int,  # kernel_analyzer: ignore
  impratio_invsqrt: float,
  efc_type: int,  # kernel_analyzer: ignore
  efc_D_in: wp.array[float],  # kernel_analyzer: ignore
  efc_frictionloss: wp.array[float],  # kernel_analyzer: ignore
  ctx_Jaref: float,
  ctx_jv: float,
  ctx_quad: wp.vec3,
  contact_friction: types.vec5,  # kernel_analyzer: ignore
  efc_address0: int,
  quad1: wp.vec3,
  quad2: wp.vec3,
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  """Compute (cost, gradient, hessian) for 3 alphas, elliptic cones.

  Returns a tuple of 3 vec3s for (lo_alpha, hi_alpha, mid_alpha).
  Constraint types checked in order: contact elliptic/limit/other -> friction -> equality.
  """
  # x = search point, needed for friction and limit constraints
  x_lo = ctx_Jaref + lo_alpha * ctx_jv
  x_hi = ctx_Jaref + hi_alpha * ctx_jv
  x_mid = ctx_Jaref + mid_alpha * ctx_jv

  # Contact/limit/other constraints
  if efcid >= ne + nf:
    # Contact elliptic: uses special elliptic cone evaluation
    if efc_type == types.ConstraintType.CONTACT_ELLIPTIC:
      if efcid != efc_address0:  # secondary rows contribute nothing
        return (wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0))
      return (
        _eval_elliptic(impratio_invsqrt, contact_friction, ctx_quad, quad1, quad2, lo_alpha),
        _eval_elliptic(impratio_invsqrt, contact_friction, ctx_quad, quad1, quad2, hi_alpha),
        _eval_elliptic(impratio_invsqrt, contact_friction, ctx_quad, quad1, quad2, mid_alpha),
      )

    # Limit/other constraints — direct eval (no quad read)
    efc_D = efc_D_in[efcid]
    pt_lo, pt_hi, pt_mid = _eval_pt_direct_3alphas(ctx_Jaref, ctx_jv, efc_D, lo_alpha, hi_alpha, mid_alpha)
    r_lo = wp.where(x_lo < 0.0, pt_lo, wp.vec3(0.0))
    r_hi = wp.where(x_hi < 0.0, pt_hi, wp.vec3(0.0))
    r_mid = wp.where(x_mid < 0.0, pt_mid, wp.vec3(0.0))
    return (r_lo, r_hi, r_mid)

  # Friction constraint - load D and frictionloss only here
  if efcid >= ne:
    efc_D = efc_D_in[efcid]
    f = efc_frictionloss[efcid]
    rf = math.safe_div(f, efc_D)
    return _eval_frictionloss_pt_3alphas(x_lo, x_hi, x_mid, f, rf, ctx_jv, efc_D)

  # Equality constraint — direct eval (no quad read)
  return _eval_pt_direct_3alphas(ctx_Jaref, ctx_jv, efc_D_in[efcid], lo_alpha, hi_alpha, mid_alpha)


# =============================================================================
# Iterative Linesearch
# =============================================================================
#
# Iterative linesearch implementation using Warp's tiled execution model with
# parallel reductions over constraint (EFC) rows.
#
# Key optimizations:
#
# 1. KERNEL FUSION - Reduces kernel launch overhead by combining:
#    - linesearch_jv_fused: jv = J @ search (for small nv <= 50)
#    - linesearch_prepare_quad: quad coefficients (pyramidal: computed directly,
#      elliptic: computed in a prepare phase with __syncthreads barrier)
#    - linesearch_prepare_gauss: quad_gauss via tile reduction over DOFs
#    - linesearch_qacc_ma: qacc and Ma updates at kernel end
#    - linesearch_jaref: Jaref update at kernel end
#
# 2. PARALLEL REDUCTIONS - Uses wp.tile_reduce for summing cost/gradient/hessian
#    contributions across EFC rows within each world. The main iteration loop
#    packs 3 vec3 reductions into a single mat33 reduction for efficiency.
#
# 3. COMPILE-TIME SPECIALIZATION via factory parameters:
#    - cone_type: Eliminates elliptic cone branches for pyramidal-only models
#    - ls_iterations: Enables loop unrolling for the main bracket search
#    - fuse_jv: Conditionally includes jv computation based on nv size
#
# 4. DIRECT EVALUATION (pyramidal only) - For equality and limit constraints,
#    computes cost/gradient/hessian directly from (Jaref, jv, efc_D, alpha)
#    without intermediate quad coefficients, using _eval_pt_direct functions.
#
# 5. BATCHED 3-ALPHA EVALUATION - The main iteration loop evaluates 3 alpha
#    values per iteration (lo_next, hi_next, mid). Instead of calling
#    _compute_efc_eval_pt 3 times per constraint row (which would repeat
#    constraint type checks and data loads), we use _compute_efc_eval_pt_3alphas
#    which:
#    - Performs constraint type branching once per row
#    - Loads efc_D, efc_frictionloss, contact data once
#    - Computes x = Jaref + alpha * jv for all 3 alphas
#    - For pyramidal direct evaluation: shares jvD = jv * efc_D, hessian = jv * jvD
#    - For quad-based evaluation: uses _eval_pt_3alphas which computes the
#      constant hessian (2.0 * quad[2]) once and reuses for all 3 alphas
#
# 6. DEFERRED DATA LOADING - efc_D and efc_frictionloss are only loaded
#    inside the constraint branches where they're needed, reducing register
#    pressure for other constraint types.
#
# Trade-offs:
# - Requires block synchronization (__syncthreads) for elliptic quad preparation
# - Separate kernel compilation for each (block_dim, ls_iterations, cone_type,
#   fuse_jv) combination (cached by Warp)
#
# Optimizations attempted but not beneficial:
# - Caching EFC data (Jaref, jv, quad, etc.) in shared memory tiles for reuse
#   across the p0, lo_in, and main iteration loops.
#
# =============================================================================


@cache_kernel
def linesearch_iterative(ls_iterations: int, cone_type: types.ConeType, fuse_jv: bool, is_sparse: bool):
  """Factory for iterative linesearch kernel.

  Args:
    ls_iterations: Max linesearch iterations (compile-time constant for loop optimization).
    cone_type: Friction cone type (PYRAMIDAL or ELLIPTIC) for compile-time optimization.
    fuse_jv: Whether to compute jv = J @ search in-kernel (efficient for small nv).
    is_sparse: Use sparse matrix representation for constraint Jacobian.
  """
  LS_ITERATIONS = ls_iterations
  IS_ELLIPTIC = cone_type == types.ConeType.ELLIPTIC
  FUSE_JV = fuse_jv
  IS_SPARSE = is_sparse

  # Native snippet for CUDA __syncthreads()
  @wp.func_native(snippet="WP_TILE_SYNC();")
  def _syncthreads():
    pass

  # Select specialized helper functions based on cone type
  if IS_ELLIPTIC:
    _compute_efc_eval_pt = _compute_efc_eval_pt_elliptic
    _compute_efc_eval_pt_alpha_zero = _compute_efc_eval_pt_alpha_zero_elliptic
    _compute_efc_eval_pt_3alphas = _compute_efc_eval_pt_3alphas_elliptic
  else:
    _compute_efc_eval_pt = _compute_efc_eval_pt_pyramidal
    _compute_efc_eval_pt_alpha_zero = _compute_efc_eval_pt_alpha_zero_pyramidal
    _compute_efc_eval_pt_3alphas = _compute_efc_eval_pt_3alphas_pyramidal

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    nv: int,
    opt_tolerance: wp.array[float],
    opt_ls_tolerance: wp.array[float],
    opt_impratio_invsqrt: wp.array[float],
    stat_meaninertia: wp.array[float],
    # Data in:
    ne_in: wp.array[int],
    nf_in: wp.array[int],
    nefc_in: wp.array[int],
    qfrc_smooth_in: wp.array2d[float],
    contact_friction_in: wp.array[types.vec5],
    contact_dim_in: wp.array[int],
    contact_efc_address_in: wp.array2d[int],
    efc_type_in: wp.array2d[int],
    efc_id_in: wp.array2d[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_frictionloss_in: wp.array2d[float],
    njmax_in: int,
    nacon_in: wp.array[int],
    # In:
    ctx_Jaref_in: wp.array2d[float],
    ctx_search_in: wp.array2d[float],
    ctx_search_dot_in: wp.array[float],
    ctx_gauss_in: wp.array[float],
    ctx_mv_in: wp.array2d[float],
    ctx_jv_in: wp.array2d[float],
    ctx_quad_in: wp.array2d[wp.vec3],
    ctx_done_in: wp.array[bool],
    # Data out:
    qacc_out: wp.array2d[float],
    efc_Ma_out: wp.array2d[float],
    # Out:
    ctx_Jaref_out: wp.array2d[float],
    ctx_jv_out: wp.array2d[float],
    ctx_quad_out: wp.array2d[wp.vec3],
  ):
    worldid, tid = wp.tid()

    if ctx_done_in[worldid]:
      return

    ne = ne_in[worldid]
    nf = nf_in[worldid]
    nefc = wp.min(njmax_in, nefc_in[worldid])

    # jv = J @ search (fused for small nv)
    if wp.static(FUSE_JV):
      for efcid in range(tid, nefc, wp.block_dim()):
        jv = float(0.0)
        if wp.static(IS_SPARSE):
          rownnz = efc_J_rownnz_in[worldid, efcid]
          rowadr = efc_J_rowadr_in[worldid, efcid]
          for k in range(rownnz):
            sparseid = rowadr + k
            colind = efc_J_colind_in[worldid, 0, sparseid]
            jv += efc_J_in[worldid, 0, sparseid] * ctx_search_in[worldid, colind]
        else:
          for i in range(nv):
            jv += efc_J_in[worldid, efcid, i] * ctx_search_in[worldid, i]
        ctx_jv_out[worldid, efcid] = jv

      _syncthreads()  # ensure all jv values are written before reading

    # quad coefficients (elliptic contacts only, requires barrier sync)
    # Non-elliptic constraints (equality, friction, limit) now use direct
    # evaluation from (Jaref, jv, efc_D), avoiding quad reads entirely.
    if wp.static(IS_ELLIPTIC):
      # elliptic-only config values
      impratio_invsqrt = opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]
      nacon = nacon_in[0]

      for efcid in range(tid, nefc, wp.block_dim()):
        # Only compute and store quad for CONTACT_ELLIPTIC (needs inter-row data)
        if efc_type_in[worldid, efcid] == types.ConstraintType.CONTACT_ELLIPTIC:
          conid = efc_id_in[worldid, efcid]
          if conid < nacon:
            efcid0 = contact_efc_address_in[conid, 0]
            if efcid == efcid0:
              Jaref = ctx_Jaref_in[worldid, efcid]
              jv = ctx_jv_in[worldid, efcid]
              efc_D = efc_D_in[worldid, efcid]

              jvD = jv * efc_D
              quad = wp.vec3(0.5 * Jaref * Jaref * efc_D, jvD * Jaref, 0.5 * jv * jvD)

              # primary row: accumulate secondary rows and write quad, quad1, quad2
              dim = contact_dim_in[conid]
              friction = contact_friction_in[conid]
              mu = friction[0] * impratio_invsqrt

              u0 = Jaref * mu
              v0 = jv * mu

              uu = float(0.0)
              uv = float(0.0)
              vv = float(0.0)
              for j in range(1, dim):
                efcidj = contact_efc_address_in[conid, j]
                if efcidj >= 0:
                  jvj = ctx_jv_in[worldid, efcidj]
                  jarefj = ctx_Jaref_in[worldid, efcidj]
                  dj = efc_D_in[worldid, efcidj]
                  DJj = dj * jarefj

                  quad += wp.vec3(0.5 * jarefj * DJj, jvj * DJj, 0.5 * jvj * dj * jvj)

                  # rescale to make primal cone circular
                  frictionj = friction[j - 1]
                  uj = jarefj * frictionj
                  vj = jvj * frictionj

                  uu += uj * uj
                  uv += uj * vj
                  vv += vj * vj

              ctx_quad_out[worldid, efcid] = quad

              efcid1 = contact_efc_address_in[conid, 1]
              ctx_quad_out[worldid, efcid1] = wp.vec3(u0, v0, uu)

              mu2 = mu * mu
              efcid2 = contact_efc_address_in[conid, 2]
              ctx_quad_out[worldid, efcid2] = wp.vec3(uv, vv, efc_D / (mu2 * (1.0 + mu2)))

      _syncthreads()  # ensure all quads are written before reading

    # gtol (tolerance values loaded here, deferred from kernel start)
    tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]
    ls_tolerance = opt_ls_tolerance[worldid % opt_ls_tolerance.shape[0]]
    snorm = wp.sqrt(ctx_search_dot_in[worldid])
    meaninertia = stat_meaninertia[worldid % stat_meaninertia.shape[0]]
    scale = meaninertia * wp.float(nv)
    gtol = wp.max(tolerance * ls_tolerance * snorm * scale, 1e-6)

    # p0 via parallel reduction
    local_p0 = wp.vec3(0.0)
    for efcid in range(tid, nefc, wp.block_dim()):
      if wp.static(IS_ELLIPTIC):
        efc_type = efc_type_in[worldid, efcid]
        efc_id = 0
        contact_friction = types.vec5(0.0)
        efc_addr0 = int(0)
        ctx_quad = wp.vec3(0.0)
        quad1 = wp.vec3(0.0)
        quad2 = wp.vec3(0.0)

        if efc_type == types.ConstraintType.CONTACT_ELLIPTIC:
          efc_id = efc_id_in[worldid, efcid]
          contact_friction = contact_friction_in[efc_id]
          efc_addr0 = contact_efc_address_in[efc_id, 0]
          efc_addr1 = contact_efc_address_in[efc_id, 1]
          efc_addr2 = contact_efc_address_in[efc_id, 2]
          ctx_quad = ctx_quad_in[worldid, efcid]
          quad1 = ctx_quad_in[worldid, efc_addr1]
          quad2 = ctx_quad_in[worldid, efc_addr2]

        local_p0 += _compute_efc_eval_pt_alpha_zero(
          efcid,
          ne,
          nf,
          impratio_invsqrt,
          efc_type,
          efc_D_in[worldid],
          efc_frictionloss_in[worldid],
          ctx_Jaref_in[worldid, efcid],
          ctx_jv_in[worldid, efcid],
          ctx_quad,
          contact_friction,
          efc_addr0,
          quad1,
          quad2,
        )
      else:
        # direct evaluation for pyramidal cones (no intermediate quad)
        local_p0 += _compute_efc_eval_pt_alpha_zero(
          efcid,
          ne,
          nf,
          efc_D_in[worldid, efcid],
          efc_frictionloss_in[worldid],
          ctx_Jaref_in[worldid, efcid],
          ctx_jv_in[worldid, efcid],
        )

    # at this point, every thread has computed some contributions to p0 in local_p0
    # we now create a tile of all local_p0 contributions and reduce them to a single value
    # this is done in parallel using a tile reduction
    p0_tile = wp.tile(local_p0, preserve_type=True)
    p0_sum = wp.tile_reduce(wp.add, p0_tile)

    # quad_gauss = [gauss, search.T @ Ma - search.T @ qfrc_smooth, 0.5 * search.T @ mv]
    local_gauss = wp.vec2(0.0)  # vec2 since component 0 is constant (ctx_gauss_in)
    for dofid in range(tid, nv, wp.block_dim()):
      search = ctx_search_in[worldid, dofid]
      local_gauss += wp.vec2(
        search * (efc_Ma_out[worldid, dofid] - qfrc_smooth_in[worldid, dofid]),
        0.5 * search * ctx_mv_in[worldid, dofid],
      )

    gauss_tile = wp.tile(local_gauss, preserve_type=True)
    gauss_sum = wp.tile_reduce(wp.add, gauss_tile)
    gauss_reduced = gauss_sum[0]
    ctx_quad_gauss = wp.vec3(ctx_gauss_in[worldid], gauss_reduced[0], gauss_reduced[1])

    # add quad_gauss contribution to p0
    p0 = wp.vec3(ctx_quad_gauss[0], ctx_quad_gauss[1], 2.0 * ctx_quad_gauss[2]) + p0_sum[0]

    # lo_in at lo_alpha_in = -p0[1] / p0[2]
    lo_alpha_in = -math.safe_div(p0[1], p0[2])

    local_lo_in = wp.vec3(0.0)
    for efcid in range(tid, nefc, wp.block_dim()):
      if wp.static(IS_ELLIPTIC):
        efc_type = efc_type_in[worldid, efcid]
        efc_id = 0
        contact_friction = types.vec5(0.0)
        efc_addr0 = int(0)
        ctx_quad = wp.vec3(0.0)
        quad1 = wp.vec3(0.0)
        quad2 = wp.vec3(0.0)

        if efc_type == types.ConstraintType.CONTACT_ELLIPTIC:
          efc_id = efc_id_in[worldid, efcid]
          contact_friction = contact_friction_in[efc_id]
          efc_addr0 = contact_efc_address_in[efc_id, 0]
          efc_addr1 = contact_efc_address_in[efc_id, 1]
          efc_addr2 = contact_efc_address_in[efc_id, 2]
          ctx_quad = ctx_quad_in[worldid, efcid]
          quad1 = ctx_quad_in[worldid, efc_addr1]
          quad2 = ctx_quad_in[worldid, efc_addr2]

        local_lo_in += _compute_efc_eval_pt(
          efcid,
          lo_alpha_in,
          ne,
          nf,
          impratio_invsqrt,
          efc_type,
          efc_D_in[worldid],
          efc_frictionloss_in[worldid],
          ctx_Jaref_in[worldid, efcid],
          ctx_jv_in[worldid, efcid],
          ctx_quad,
          contact_friction,
          efc_addr0,
          quad1,
          quad2,
        )
      else:
        # direct evaluation for pyramidal cones (no intermediate quad)
        local_lo_in += _compute_efc_eval_pt(
          efcid,
          lo_alpha_in,
          ne,
          nf,
          efc_D_in[worldid, efcid],
          efc_frictionloss_in[worldid],
          ctx_Jaref_in[worldid, efcid],
          ctx_jv_in[worldid, efcid],
        )

    lo_in_tile = wp.tile(local_lo_in, preserve_type=True)
    lo_in_sum = wp.tile_reduce(wp.add, lo_in_tile)
    lo_in = _eval_pt(ctx_quad_gauss, lo_alpha_in) + lo_in_sum[0]

    # accept Newton step if derivative is small and cost improved
    initial_converged = wp.abs(lo_in[1]) < gtol and lo_in[0] < p0[0]

    # main iterative loop - skip if already converged
    if not initial_converged:
      alpha = float(0.0)

      # initialize bounds
      lo_less = lo_in[1] < p0[1]
      lo = wp.where(lo_less, lo_in, p0)
      lo_alpha = wp.where(lo_less, lo_alpha_in, 0.0)
      hi = wp.where(lo_less, p0, lo_in)
      hi_alpha = wp.where(lo_less, 0.0, lo_alpha_in)

      for _ in range(LS_ITERATIONS):
        lo_next_alpha = lo_alpha - math.safe_div(lo[1], lo[2])
        hi_next_alpha = hi_alpha - math.safe_div(hi[1], hi[2])
        mid_alpha = 0.5 * (lo_alpha + hi_alpha)

        local_lo = wp.vec3(0.0)
        local_hi = wp.vec3(0.0)
        local_mid = wp.vec3(0.0)

        for efcid in range(tid, nefc, wp.block_dim()):
          if wp.static(IS_ELLIPTIC):
            efc_type = efc_type_in[worldid, efcid]
            efc_id = 0
            contact_friction = types.vec5(0.0)
            efc_addr0 = int(0)
            ctx_quad = wp.vec3(0.0)
            quad1 = wp.vec3(0.0)
            quad2 = wp.vec3(0.0)

            if efc_type == types.ConstraintType.CONTACT_ELLIPTIC:
              efc_id = efc_id_in[worldid, efcid]
              contact_friction = contact_friction_in[efc_id]
              efc_addr0 = contact_efc_address_in[efc_id, 0]
              efc_addr1 = contact_efc_address_in[efc_id, 1]
              efc_addr2 = contact_efc_address_in[efc_id, 2]
              ctx_quad = ctx_quad_in[worldid, efcid]
              quad1 = ctx_quad_in[worldid, efc_addr1]
              quad2 = ctx_quad_in[worldid, efc_addr2]

            r_lo, r_hi, r_mid = _compute_efc_eval_pt_3alphas(
              efcid,
              lo_next_alpha,
              hi_next_alpha,
              mid_alpha,
              ne,
              nf,
              impratio_invsqrt,
              efc_type,
              efc_D_in[worldid],
              efc_frictionloss_in[worldid],
              ctx_Jaref_in[worldid, efcid],
              ctx_jv_in[worldid, efcid],
              ctx_quad,
              contact_friction,
              efc_addr0,
              quad1,
              quad2,
            )
          else:
            # direct evaluation for pyramidal cones (no intermediate quad)
            r_lo, r_hi, r_mid = _compute_efc_eval_pt_3alphas(
              efcid,
              lo_next_alpha,
              hi_next_alpha,
              mid_alpha,
              ne,
              nf,
              efc_D_in[worldid, efcid],
              efc_frictionloss_in[worldid],
              ctx_Jaref_in[worldid, efcid],
              ctx_jv_in[worldid, efcid],
            )
          local_lo += r_lo
          local_hi += r_hi
          local_mid += r_mid

        # reduce with packed mat33 (3 vec3s into columns: col0=lo, col1=hi, col2=mid)
        local_combined = wp.mat33(
          local_lo[0],
          local_hi[0],
          local_mid[0],
          local_lo[1],
          local_hi[1],
          local_mid[1],
          local_lo[2],
          local_hi[2],
          local_mid[2],
        )

        # reduce with packed mat33 (3 vec3s into columns: col0=lo, col1=hi, col2=mid)
        # this is faster than 3 vec3 reductions because it avoids synchronization barriers
        combined_tile = wp.tile(local_combined, preserve_type=True)
        combined_sum = wp.tile_reduce(wp.add, combined_tile)
        result = combined_sum[0]

        # extract columns back to vec3s and add quad_gauss contributions
        gauss_lo, gauss_hi, gauss_mid = _eval_pt_3alphas(ctx_quad_gauss, lo_next_alpha, hi_next_alpha, mid_alpha)
        lo_next = gauss_lo + wp.vec3(result[0, 0], result[1, 0], result[2, 0])
        hi_next = gauss_hi + wp.vec3(result[0, 1], result[1, 1], result[2, 1])
        mid = gauss_mid + wp.vec3(result[0, 2], result[1, 2], result[2, 2])

        # bracket swapping
        # swap lo:
        swap_lo_lo_next = _in_bracket(lo, lo_next)
        lo = wp.where(swap_lo_lo_next, lo_next, lo)
        lo_alpha = wp.where(swap_lo_lo_next, lo_next_alpha, lo_alpha)
        swap_lo_mid = _in_bracket(lo, mid)
        lo = wp.where(swap_lo_mid, mid, lo)
        lo_alpha = wp.where(swap_lo_mid, mid_alpha, lo_alpha)
        swap_lo_hi_next = _in_bracket(lo, hi_next)
        lo = wp.where(swap_lo_hi_next, hi_next, lo)
        lo_alpha = wp.where(swap_lo_hi_next, hi_next_alpha, lo_alpha)
        swap_lo = swap_lo_lo_next or swap_lo_mid or swap_lo_hi_next

        # swap hi:
        swap_hi_hi_next = _in_bracket(hi, hi_next)
        hi = wp.where(swap_hi_hi_next, hi_next, hi)
        hi_alpha = wp.where(swap_hi_hi_next, hi_next_alpha, hi_alpha)
        swap_hi_mid = _in_bracket(hi, mid)
        hi = wp.where(swap_hi_mid, mid, hi)
        hi_alpha = wp.where(swap_hi_mid, mid_alpha, hi_alpha)
        swap_hi_lo_next = _in_bracket(hi, lo_next)
        hi = wp.where(swap_hi_lo_next, lo_next, hi)
        hi_alpha = wp.where(swap_hi_lo_next, lo_next_alpha, hi_alpha)
        swap_hi = swap_hi_hi_next or swap_hi_mid or swap_hi_lo_next

        # check for convergence
        ls_done = (not swap_lo and not swap_hi) or (lo[1] < 0.0 and lo[1] > -gtol) or (hi[1] > 0.0 and hi[1] < gtol)

        # update alpha if improved
        improved = lo[0] < p0[0] or hi[0] < p0[0]
        lo_better = lo[0] < hi[0]
        alpha = wp.where(improved and lo_better, lo_alpha, alpha)
        alpha = wp.where(improved and not lo_better, hi_alpha, alpha)

        if ls_done:
          break
    else:
      alpha = lo_alpha_in

    # qacc and Ma update
    for dofid in range(tid, nv, wp.block_dim()):
      qacc_out[worldid, dofid] += alpha * ctx_search_in[worldid, dofid]
      efc_Ma_out[worldid, dofid] += alpha * ctx_mv_in[worldid, dofid]

    # Jaref update
    for efcid in range(tid, nefc, wp.block_dim()):
      ctx_Jaref_out[worldid, efcid] += alpha * ctx_jv_in[worldid, efcid]

  return kernel


def _linesearch_iterative(m: types.Model, d: types.Data, ctx: SolverContext, fuse_jv: bool):
  """Iterative linesearch with parallel reductions over efc rows and dofs.

  Args:
    m: Model.
    d: Data.
    ctx: SolverContext.
    fuse_jv: Whether jv is computed in-kernel (True) or pre-computed (False).
  """
  wp.launch_tiled(
    linesearch_iterative(m.opt.ls_iterations, m.opt.cone, fuse_jv, m.is_sparse),
    dim=d.nworld,
    inputs=[
      m.nv,
      m.opt.tolerance,
      m.opt.ls_tolerance,
      m.opt.impratio_invsqrt,
      m.stat.meaninertia,
      d.ne,
      d.nf,
      d.nefc,
      d.qfrc_smooth,
      d.contact.friction,
      d.contact.dim,
      d.contact.efc_address,
      d.efc.type,
      d.efc.id,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      d.efc.J,
      d.efc.D,
      d.efc.frictionloss,
      d.njmax,
      d.nacon,
      ctx.Jaref,
      ctx.search,
      ctx.search_dot,
      ctx.gauss,
      ctx.mv,
      ctx.jv,
      ctx.quad,
      ctx.done,
    ],
    outputs=[d.qacc, d.efc.Ma, ctx.Jaref, ctx.jv, ctx.quad],
    block_dim=m.block_dim.linesearch_iterative,
  )


@wp.kernel
def linesearch_zero_jv(
  # Data in:
  nefc_in: wp.array[int],
  # In:
  ctx_done_in: wp.array[bool],
  # Out:
  ctx_jv_out: wp.array2d[float],
):
  worldid, efcid = wp.tid()

  if efcid >= nefc_in[worldid]:
    return

  if ctx_done_in[worldid]:
    return

  ctx_jv_out[worldid, efcid] = 0.0


@cache_kernel
def linesearch_jv_fused(is_sparse: bool, nv: int, dofs_per_thread: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    # In:
    ctx_search_in: wp.array2d[float],
    ctx_done_in: wp.array[bool],
    # Out:
    ctx_jv_out: wp.array2d[float],
  ):
    worldid, efcid, dofstart = wp.tid()

    if efcid >= nefc_in[worldid]:
      return

    if ctx_done_in[worldid]:
      return

    jv_out = float(0.0)

    if wp.static(dofs_per_thread >= nv):
      if wp.static(is_sparse):
        # Sparse: iterate over non-zero entries in the row
        rownnz = efc_J_rownnz_in[worldid, efcid]
        rowadr = efc_J_rowadr_in[worldid, efcid]
        for k in range(rownnz):
          sparseid = rowadr + k
          colind = efc_J_colind_in[worldid, 0, sparseid]
          jv_out += efc_J_in[worldid, 0, sparseid] * ctx_search_in[worldid, colind]
      else:
        for i in range(wp.static(min(dofs_per_thread, nv))):
          jv_out += efc_J_in[worldid, efcid, i] * ctx_search_in[worldid, i]
      ctx_jv_out[worldid, efcid] = jv_out

    else:
      if wp.static(is_sparse):
        # Sparse: thread 0 handles entire row (sparse entries << nv typically)
        if dofstart == 0:
          rownnz = efc_J_rownnz_in[worldid, efcid]
          rowadr = efc_J_rowadr_in[worldid, efcid]
          for k in range(rownnz):
            sparseid = rowadr + k
            colind = efc_J_colind_in[worldid, 0, sparseid]
            jv_out += efc_J_in[worldid, 0, sparseid] * ctx_search_in[worldid, colind]
          ctx_jv_out[worldid, efcid] = jv_out
      else:
        for i in range(wp.static(dofs_per_thread)):
          ii = dofstart * wp.static(dofs_per_thread) + i
          if ii < nv:
            jv_out += efc_J_in[worldid, efcid, ii] * ctx_search_in[worldid, ii]
        wp.atomic_add(ctx_jv_out, worldid, efcid, jv_out)

  return kernel


@cache_kernel
def linesearch_prepare_gauss(nv: int, dofs_per_thread: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    qfrc_smooth_in: wp.array2d[float],
    efc_Ma_in: wp.array2d[float],
    # In:
    ctx_search_in: wp.array2d[float],
    ctx_gauss_in: wp.array[float],
    ctx_mv_in: wp.array2d[float],
    ctx_done_in: wp.array[bool],
    # Out:
    ctx_quad_gauss_out: wp.array[wp.vec3],
  ):
    worldid, dofstart = wp.tid()

    if ctx_done_in[worldid]:
      return

    quad_gauss_1 = float(0.0)
    quad_gauss_2 = float(0.0)

    if wp.static(dofs_per_thread >= nv):
      for i in range(wp.static(nv)):
        search = ctx_search_in[worldid, i]
        quad_gauss_1 += search * (efc_Ma_in[worldid, i] - qfrc_smooth_in[worldid, i])
        quad_gauss_2 += 0.5 * search * ctx_mv_in[worldid, i]

      quad_gauss_0 = ctx_gauss_in[worldid]
      ctx_quad_gauss_out[worldid] = wp.vec3(quad_gauss_0, quad_gauss_1, quad_gauss_2)

    else:
      for i in range(wp.static(dofs_per_thread)):
        ii = dofstart * wp.static(dofs_per_thread) + i
        if ii < nv:
          search = ctx_search_in[worldid, ii]
          quad_gauss_1 += search * (efc_Ma_in[worldid, ii] - qfrc_smooth_in[worldid, ii])
          quad_gauss_2 += 0.5 * search * ctx_mv_in[worldid, ii]

      if dofstart == 0:
        quad_gauss_0 = ctx_gauss_in[worldid]
        wp.atomic_add(ctx_quad_gauss_out, worldid, wp.vec3(quad_gauss_0, quad_gauss_1, quad_gauss_2))
      else:
        wp.atomic_add(ctx_quad_gauss_out, worldid, wp.vec3(0.0, quad_gauss_1, quad_gauss_2))

  return kernel


@wp.kernel
def linesearch_prepare_quad(
  # Model:
  opt_impratio_invsqrt: wp.array[float],
  # Data in:
  nefc_in: wp.array[int],
  contact_friction_in: wp.array[types.vec5],
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array2d[int],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  efc_D_in: wp.array2d[float],
  nacon_in: wp.array[int],
  # In:
  ctx_Jaref_in: wp.array2d[float],
  ctx_jv_in: wp.array2d[float],
  ctx_done_in: wp.array[bool],
  # Out:
  ctx_quad_out: wp.array2d[wp.vec3],
):
  worldid, efcid = wp.tid()

  if efcid >= nefc_in[worldid]:
    return

  if ctx_done_in[worldid]:
    return

  Jaref = ctx_Jaref_in[worldid, efcid]
  jv = ctx_jv_in[worldid, efcid]
  efc_D = efc_D_in[worldid, efcid]

  # init with scalar quadratic
  quad = wp.vec3(0.5 * Jaref * Jaref * efc_D, jv * Jaref * efc_D, 0.5 * jv * jv * efc_D)

  # elliptic cone: extra processing
  if efc_type_in[worldid, efcid] == types.ConstraintType.CONTACT_ELLIPTIC:
    # extract contact info
    conid = efc_id_in[worldid, efcid]

    if conid >= nacon_in[0]:
      return

    efcid0 = contact_efc_address_in[conid, 0]

    if efcid != efcid0:
      return

    dim = contact_dim_in[conid]
    friction = contact_friction_in[conid]
    mu = friction[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]

    u0 = Jaref * mu
    v0 = jv * mu

    uu = float(0.0)
    uv = float(0.0)
    vv = float(0.0)
    for j in range(1, dim):
      # complete vector quadratic (for bottom zone)
      efcidj = contact_efc_address_in[conid, j]
      if efcidj < 0:
        return
      jvj = ctx_jv_in[worldid, efcidj]
      jarefj = ctx_Jaref_in[worldid, efcidj]
      dj = efc_D_in[worldid, efcidj]
      DJj = dj * jarefj

      quad += wp.vec3(
        0.5 * jarefj * DJj,
        jvj * DJj,
        0.5 * jvj * dj * jvj,
      )

      # rescale to make primal cone circular
      frictionj = friction[j - 1]
      uj = jarefj * frictionj
      vj = jvj * frictionj

      # accumulate sums of squares
      uu += uj * uj
      uv += uj * vj
      vv += vj * vj

    quad1 = wp.vec3(u0, v0, uu)
    efcid1 = contact_efc_address_in[conid, 1]
    ctx_quad_out[worldid, efcid1] = quad1

    mu2 = mu * mu
    quad2 = wp.vec3(uv, vv, efc_D / (mu2 * (1.0 + mu2)))
    efcid2 = contact_efc_address_in[conid, 2]
    ctx_quad_out[worldid, efcid2] = quad2

  ctx_quad_out[worldid, efcid] = quad


@wp.kernel
def linesearch_qacc_ma(
  # In:
  ctx_search_in: wp.array2d[float],
  ctx_mv_in: wp.array2d[float],
  ctx_alpha_in: wp.array[float],
  ctx_done_in: wp.array[bool],
  # Data out:
  qacc_out: wp.array2d[float],
  efc_Ma_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()

  if ctx_done_in[worldid]:
    return

  alpha = ctx_alpha_in[worldid]
  qacc_out[worldid, dofid] += alpha * ctx_search_in[worldid, dofid]
  efc_Ma_out[worldid, dofid] += alpha * ctx_mv_in[worldid, dofid]


@wp.kernel
def linesearch_jaref(
  # Data in:
  nefc_in: wp.array[int],
  # In:
  ctx_jv_in: wp.array2d[float],
  ctx_alpha_in: wp.array[float],
  ctx_done_in: wp.array[bool],
  # Out:
  ctx_Jaref_out: wp.array2d[float],
):
  worldid, efcid = wp.tid()

  if efcid >= nefc_in[worldid]:
    return

  if ctx_done_in[worldid]:
    return

  ctx_Jaref_out[worldid, efcid] += ctx_alpha_in[worldid] * ctx_jv_in[worldid, efcid]


@event_scope
def _linesearch(m: types.Model, d: types.Data, ctx: SolverContext, cost: wp.array2d[float]):
  """Linesearch for constraint solver.

  Args:
    m: Model
    d: Data
    ctx: SolverContext
    cost: Scratch array for storing costs per (world, alpha) - used for parallel mode
  """
  # mv = qM @ search (common to both parallel and iterative)
  support.mul_m(m, d, ctx.mv, ctx.search, skip=ctx.done)

  # Fuse jv computation in-kernel for small nv (iterative only, dense only)
  # Parallel linesearch always requires jv pre-computed
  # Sparse mode requires pre-computed jv since in-kernel uses dense indexing
  fuse_jv = m.nv <= 50 and not m.opt.ls_parallel and not m.is_sparse

  # jv = J @ search (when not fused into iterative kernel)
  if not fuse_jv:
    dofs_per_thread = 20 if m.nv > 50 else 50
    threads_per_efc = ceil(m.nv / dofs_per_thread)

    if threads_per_efc > 1:
      wp.launch(
        linesearch_zero_jv,
        dim=(d.nworld, d.njmax),
        inputs=[d.nefc, ctx.done],
        outputs=[ctx.jv],
      )

    wp.launch(
      linesearch_jv_fused(m.is_sparse, m.nv, dofs_per_thread),
      dim=(d.nworld, d.njmax, threads_per_efc),
      inputs=[d.nefc, d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind, d.efc.J, ctx.search, ctx.done],
      outputs=[ctx.jv],
    )

  if m.opt.ls_parallel:
    _linesearch_parallel(m, d, ctx, cost)
  else:
    _linesearch_iterative(m, d, ctx, fuse_jv)


@wp.kernel
def solve_init_efc(
  # Data out:
  solver_niter_out: wp.array[int],
  # Out:
  ctx_search_dot_out: wp.array[float],
  ctx_cost_out: wp.array[float],
  ctx_done_out: wp.array[bool],
):
  worldid = wp.tid()
  ctx_cost_out[worldid] = types.MJ_MAXVAL
  solver_niter_out[worldid] = 0
  ctx_done_out[worldid] = False
  ctx_search_dot_out[worldid] = 0.0


@cache_kernel
def solve_init_jaref(is_sparse: bool, nv: int, dofs_per_thread: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    qacc_in: wp.array2d[float],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_aref_in: wp.array2d[float],
    # Out:
    ctx_Jaref_out: wp.array2d[float],
  ):
    worldid, efcid, dofstart = wp.tid()

    if efcid >= nefc_in[worldid]:
      return

    jaref = float(0.0)
    if wp.static(is_sparse):
      rownnz = efc_J_rownnz_in[worldid, efcid]
      rowadr = efc_J_rowadr_in[worldid, efcid]
      for i in range(rownnz):
        sparseid = rowadr + i
        colind = efc_J_colind_in[worldid, 0, sparseid]
        jaref += efc_J_in[worldid, 0, sparseid] * qacc_in[worldid, colind]
      ctx_Jaref_out[worldid, efcid] = jaref - efc_aref_in[worldid, efcid]
    else:
      if wp.static(dofs_per_thread >= nv):
        for i in range(wp.static(min(dofs_per_thread, nv))):
          jaref += efc_J_in[worldid, efcid, i] * qacc_in[worldid, i]
        ctx_Jaref_out[worldid, efcid] = jaref - efc_aref_in[worldid, efcid]

      else:
        for i in range(wp.static(dofs_per_thread)):
          ii = dofstart * wp.static(dofs_per_thread) + i
          if ii < nv:
            jaref += efc_J_in[worldid, efcid, ii] * qacc_in[worldid, ii]

        if dofstart == 0:
          wp.atomic_add(ctx_Jaref_out, worldid, efcid, jaref - efc_aref_in[worldid, efcid])
        else:
          wp.atomic_add(ctx_Jaref_out, worldid, efcid, jaref)

  return kernel


@wp.kernel
def solve_init_search(
  # In:
  ctx_Mgrad_in: wp.array2d[float],
  # Out:
  ctx_search_out: wp.array2d[float],
  ctx_search_dot_out: wp.array[float],
):
  worldid, dofid = wp.tid()
  search = -1.0 * ctx_Mgrad_in[worldid, dofid]
  ctx_search_out[worldid, dofid] = search
  wp.atomic_add(ctx_search_dot_out, worldid, search * search)


@wp.kernel
def update_constraint_init_cost(
  # In:
  ctx_cost_in: wp.array[float],
  ctx_done_in: wp.array[bool],
  # Out:
  ctx_gauss_out: wp.array[float],
  ctx_cost_out: wp.array[float],
  ctx_prev_cost_out: wp.array[float],
):
  worldid = wp.tid()

  if ctx_done_in[worldid]:
    return

  ctx_gauss_out[worldid] = 0.0
  ctx_prev_cost_out[worldid] = ctx_cost_in[worldid]
  ctx_cost_out[worldid] = 0.0


@cache_kernel
def update_constraint_efc(track_changes: bool):
  TRACK_CHANGES = track_changes

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    opt_impratio_invsqrt: wp.array[float],
    # Data in:
    ne_in: wp.array[int],
    nf_in: wp.array[int],
    nefc_in: wp.array[int],
    contact_friction_in: wp.array[types.vec5],
    contact_dim_in: wp.array[int],
    contact_efc_address_in: wp.array2d[int],
    efc_type_in: wp.array2d[int],
    efc_id_in: wp.array2d[int],
    efc_D_in: wp.array2d[float],
    efc_frictionloss_in: wp.array2d[float],
    nacon_in: wp.array[int],
    # In:
    ctx_Jaref_in: wp.array2d[float],
    ctx_done_in: wp.array[bool],
    # Data out:
    efc_force_out: wp.array2d[float],
    efc_state_out: wp.array2d[int],
    # Out:
    ctx_cost_out: wp.array[float],
    changed_ids_out: wp.array2d[int],
    changed_count_out: wp.array[int],
  ):
    worldid, efcid = wp.tid()

    if efcid >= nefc_in[worldid]:
      return

    if ctx_done_in[worldid]:
      return

    # Read old QUADRATIC status before overwriting
    if wp.static(TRACK_CHANGES):
      old_quad = efc_state_out[worldid, efcid] == types.ConstraintState.QUADRATIC.value

    efc_D = efc_D_in[worldid, efcid]
    Jaref = ctx_Jaref_in[worldid, efcid]

    ne = ne_in[worldid]
    nf = nf_in[worldid]

    new_state = types.ConstraintState.SATISFIED.value

    if efcid < ne:
      # equality
      efc_force_out[worldid, efcid] = -efc_D * Jaref
      new_state = types.ConstraintState.QUADRATIC.value
      wp.atomic_add(ctx_cost_out, worldid, 0.5 * efc_D * Jaref * Jaref)
    elif efcid < ne + nf:
      # friction
      f = efc_frictionloss_in[worldid, efcid]
      rf = math.safe_div(f, efc_D)
      if Jaref <= -rf:
        efc_force_out[worldid, efcid] = f
        new_state = types.ConstraintState.LINEARNEG.value
        wp.atomic_add(ctx_cost_out, worldid, -f * (0.5 * rf + Jaref))
      elif Jaref >= rf:
        efc_force_out[worldid, efcid] = -f
        new_state = types.ConstraintState.LINEARPOS.value
        wp.atomic_add(ctx_cost_out, worldid, -f * (0.5 * rf - Jaref))
      else:
        efc_force_out[worldid, efcid] = -efc_D * Jaref
        new_state = types.ConstraintState.QUADRATIC.value
        wp.atomic_add(ctx_cost_out, worldid, 0.5 * efc_D * Jaref * Jaref)
    elif efc_type_in[worldid, efcid] != types.ConstraintType.CONTACT_ELLIPTIC:
      # limit, frictionless contact, pyramidal friction cone contact
      if Jaref >= 0.0:
        efc_force_out[worldid, efcid] = 0.0
        new_state = types.ConstraintState.SATISFIED.value
      else:
        efc_force_out[worldid, efcid] = -efc_D * Jaref
        new_state = types.ConstraintState.QUADRATIC.value
        wp.atomic_add(ctx_cost_out, worldid, 0.5 * efc_D * Jaref * Jaref)
    else:  # elliptic friction cone contact
      conid = efc_id_in[worldid, efcid]

      if conid >= nacon_in[0]:
        return

      dim = contact_dim_in[conid]
      friction = contact_friction_in[conid]
      mu = friction[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]

      efcid0 = contact_efc_address_in[conid, 0]
      if efcid0 < 0:
        return

      N = ctx_Jaref_in[worldid, efcid0] * mu

      ufrictionj = float(0.0)
      TT = float(0.0)
      for j in range(1, dim):
        efcidj = contact_efc_address_in[conid, j]
        if efcidj < 0:
          return
        frictionj = friction[j - 1]
        uj = ctx_Jaref_in[worldid, efcidj] * frictionj
        TT += uj * uj
        if efcid == efcidj:
          ufrictionj = uj * frictionj

      if TT <= 0.0:
        T = 0.0
      else:
        T = wp.sqrt(TT)

      # top zone
      if (N >= mu * T) or ((T <= 0.0) and (N >= 0.0)):
        efc_force_out[worldid, efcid] = 0.0
        new_state = types.ConstraintState.SATISFIED.value
      # bottom zone
      elif (mu * N + T <= 0.0) or ((T <= 0.0) and (N < 0.0)):
        efc_force_out[worldid, efcid] = -efc_D * Jaref
        new_state = types.ConstraintState.QUADRATIC.value
        wp.atomic_add(ctx_cost_out, worldid, 0.5 * efc_D * Jaref * Jaref)
      # middle zone
      else:
        dm = math.safe_div(efc_D_in[worldid, efcid0], mu * mu * (1.0 + mu * mu))
        nmt = N - mu * T

        force = -dm * nmt * mu

        if efcid == efcid0:
          efc_force_out[worldid, efcid] = force
          wp.atomic_add(ctx_cost_out, worldid, 0.5 * dm * nmt * nmt)
        else:
          efc_force_out[worldid, efcid] = -math.safe_div(force, T) * ufrictionj

        new_state = types.ConstraintState.CONE.value

    efc_state_out[worldid, efcid] = new_state

    if wp.static(TRACK_CHANGES):
      new_quad = new_state == types.ConstraintState.QUADRATIC.value
      if old_quad != new_quad:
        idx = wp.atomic_add(changed_count_out, worldid, 1)
        changed_ids_out[worldid, idx] = efcid

  return kernel


@wp.kernel
def update_constraint_init_qfrc_constraint_sparse(
  # Data in:
  nefc_in: wp.array[int],
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  efc_J_in: wp.array3d[float],
  efc_force_in: wp.array2d[float],
  # In:
  ctx_done_in: wp.array[bool],
  # Data out:
  qfrc_constraint_out: wp.array2d[float],
):
  worldid, efcid = wp.tid()

  if ctx_done_in[worldid]:
    return

  if efcid >= nefc_in[worldid]:
    return

  force = efc_force_in[worldid, efcid]

  rownnz = efc_J_rownnz_in[worldid, efcid]
  rowadr = efc_J_rowadr_in[worldid, efcid]
  for i in range(rownnz):
    sparseid = rowadr + i
    colind = efc_J_colind_in[worldid, 0, sparseid]
    efc_J = efc_J_in[worldid, 0, sparseid]
    wp.atomic_add(qfrc_constraint_out[worldid], colind, efc_J * force)


@wp.kernel
def update_constraint_init_qfrc_constraint_dense(
  # Data in:
  nefc_in: wp.array[int],
  efc_J_in: wp.array3d[float],
  efc_force_in: wp.array2d[float],
  njmax_in: int,
  # In:
  ctx_done_in: wp.array[bool],
  # Data out:
  qfrc_constraint_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()

  if ctx_done_in[worldid]:
    return

  sum_qfrc = float(0.0)
  for efcid in range(min(njmax_in, nefc_in[worldid])):
    efc_J = efc_J_in[worldid, efcid, dofid]
    force = efc_force_in[worldid, efcid]
    sum_qfrc += efc_J * force

  qfrc_constraint_out[worldid, dofid] = sum_qfrc


@cache_kernel
def update_constraint_gauss_cost(nv: int, dofs_per_thread: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    qacc_in: wp.array2d[float],
    qfrc_smooth_in: wp.array2d[float],
    qacc_smooth_in: wp.array2d[float],
    efc_Ma_in: wp.array2d[float],
    # In:
    ctx_done_in: wp.array[bool],
    # Out:
    ctx_gauss_out: wp.array[float],
    ctx_cost_out: wp.array[float],
  ):
    worldid, dofstart = wp.tid()

    if ctx_done_in[worldid]:
      return

    gauss_cost = float(0.0)

    if wp.static(dofs_per_thread >= nv):
      for i in range(wp.static(min(dofs_per_thread, nv))):
        gauss_cost += (efc_Ma_in[worldid, i] - qfrc_smooth_in[worldid, i]) * (qacc_in[worldid, i] - qacc_smooth_in[worldid, i])
      ctx_gauss_out[worldid] += 0.5 * gauss_cost
      ctx_cost_out[worldid] += 0.5 * gauss_cost

    else:
      for i in range(wp.static(dofs_per_thread)):
        ii = dofstart * wp.static(dofs_per_thread) + i
        if ii < nv:
          gauss_cost += (efc_Ma_in[worldid, ii] - qfrc_smooth_in[worldid, ii]) * (
            qacc_in[worldid, ii] - qacc_smooth_in[worldid, ii]
          )
      wp.atomic_add(ctx_gauss_out, worldid, 0.5 * gauss_cost)
      wp.atomic_add(ctx_cost_out, worldid, 0.5 * gauss_cost)

  return kernel


@wp.kernel
def update_gradient_h_incremental(
  # Data in:
  efc_J_in: wp.array3d[float],
  efc_D_in: wp.array2d[float],
  efc_state_in: wp.array2d[int],
  # In:
  changed_ids_in: wp.array2d[int],
  changed_count_in: wp.array[int],
  # Out:
  ctx_h_out: wp.array3d[float],
):
  """Incrementally update lower triangle of H for changed constraints.

  Each thread handles one (i, j) element of the lower triangle.
  For each changed constraint, adds or subtracts D * J[i] * J[j].
  """
  worldid, elementid = wp.tid()

  n_changes = changed_count_in[worldid]
  if n_changes == 0:
    return

  # Lower triangle index: elementid -> (i, j) where i >= j
  i = (int(wp.sqrt(float(1 + 8 * elementid))) - 1) // 2
  j = elementid - (i * (i + 1)) // 2

  delta = float(0.0)
  for change_idx in range(n_changes):
    efcid = changed_ids_in[worldid, change_idx]
    Ji = efc_J_in[worldid, efcid, i]
    if Ji == 0.0:
      continue
    Jj = efc_J_in[worldid, efcid, j]
    if Jj == 0.0:
      continue

    D = efc_D_in[worldid, efcid]
    if efc_state_in[worldid, efcid] == types.ConstraintState.QUADRATIC.value:
      delta += D * Ji * Jj
    else:
      delta -= D * Ji * Jj

  if delta != 0.0:
    ctx_h_out[worldid, i, j] += delta


@wp.kernel
def update_gradient_h_incremental_sparse(
  # Data in:
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  efc_J_in: wp.array3d[float],
  efc_D_in: wp.array2d[float],
  efc_state_in: wp.array2d[int],
  # In:
  changed_ids_in: wp.array2d[int],
  changed_count_in: wp.array[int],
  # Out:
  ctx_h_out: wp.array3d[float],
):
  """Incrementally update lower triangle of H for changed constraints (sparse J)."""
  worldid, change_idx = wp.tid()

  n_changes = changed_count_in[worldid]
  if change_idx >= n_changes:
    return

  efcid = changed_ids_in[worldid, change_idx]
  D = efc_D_in[worldid, efcid]
  sign = float(0.0)
  if efc_state_in[worldid, efcid] == types.ConstraintState.QUADRATIC.value:
    sign = D
  else:
    sign = -D

  rownnz = efc_J_rownnz_in[worldid, efcid]
  rowadr = efc_J_rowadr_in[worldid, efcid]

  for ii in range(rownnz):
    sparseidi = rowadr + ii
    Ji = efc_J_in[worldid, 0, sparseidi]
    if Ji == 0.0:
      continue
    colindi = efc_J_colind_in[worldid, 0, sparseidi]
    for jj in range(ii + 1):
      sparseidj = rowadr + jj
      Jj = efc_J_in[worldid, 0, sparseidj]
      if Jj == 0.0:
        continue
      colindj = efc_J_colind_in[worldid, 0, sparseidj]
      h = sign * Ji * Jj
      # Ensure lower triangle: larger index first
      if colindi >= colindj:
        wp.atomic_add(ctx_h_out[worldid, colindi], colindj, h)
      else:
        wp.atomic_add(ctx_h_out[worldid, colindj], colindi, h)


def _update_constraint(m: types.Model, d: types.Data, ctx: SolverContext | InverseContext, track_changes: bool = False):
  """Update constraint arrays after each solve iteration."""
  wp.launch(
    update_constraint_init_cost,
    dim=(d.nworld),
    inputs=[ctx.cost, ctx.done],
    outputs=[ctx.gauss, ctx.cost, ctx.prev_cost],
  )

  efc_inputs = [
    m.opt.impratio_invsqrt,
    d.ne,
    d.nf,
    d.nefc,
    d.contact.friction,
    d.contact.dim,
    d.contact.efc_address,
    d.efc.type,
    d.efc.id,
    d.efc.D,
    d.efc.frictionloss,
    d.nacon,
    ctx.Jaref,
    ctx.done,
  ]

  wp.launch(
    update_constraint_efc(track_changes),
    dim=(d.nworld, d.njmax),
    inputs=efc_inputs,
    outputs=[d.efc.force, d.efc.state, ctx.cost, ctx.changed_efc_ids, ctx.changed_efc_count],
  )

  # qfrc_constraint = efc_J.T @ efc_force
  if m.is_sparse:
    d.qfrc_constraint.zero_()
    wp.launch(
      update_constraint_init_qfrc_constraint_sparse,
      dim=(d.nworld, d.njmax),
      inputs=[d.nefc, d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind, d.efc.J, d.efc.force, ctx.done],
      outputs=[d.qfrc_constraint],
    )
  else:
    wp.launch(
      update_constraint_init_qfrc_constraint_dense,
      dim=(d.nworld, m.nv),
      inputs=[d.nefc, d.efc.J, d.efc.force, d.njmax, ctx.done],
      outputs=[d.qfrc_constraint],
    )

  # if we are only using 1 thread, it makes sense to do more dofs and skip the atomics.
  # For more than 1 thread, dofs_per_thread is lower for better load balancing.
  if m.nv > 50:
    dofs_per_thread = 20
  else:
    dofs_per_thread = 50

  threads_per_efc = ceil(m.nv / dofs_per_thread)

  # gauss = 0.5 * (Ma - qfrc_smooth).T @ (qacc - qacc_smooth)
  wp.launch(
    update_constraint_gauss_cost(m.nv, dofs_per_thread),
    dim=(d.nworld, threads_per_efc),
    inputs=[d.qacc, d.qfrc_smooth, d.qacc_smooth, d.efc.Ma, ctx.done],
    outputs=[ctx.gauss, ctx.cost],
  )


@wp.kernel
def update_gradient_zero_grad_dot(
  # In:
  ctx_done_in: wp.array[bool],
  # Out:
  ctx_grad_dot_out: wp.array[float],
):
  worldid = wp.tid()

  if ctx_done_in[worldid]:
    return

  ctx_grad_dot_out[worldid] = 0.0


@wp.kernel
def update_gradient_grad(
  # Data in:
  qfrc_smooth_in: wp.array2d[float],
  qfrc_constraint_in: wp.array2d[float],
  efc_Ma_in: wp.array2d[float],
  # In:
  ctx_done_in: wp.array[bool],
  # Out:
  ctx_grad_out: wp.array2d[float],
  ctx_grad_dot_out: wp.array[float],
):
  worldid, dofid = wp.tid()

  if ctx_done_in[worldid]:
    return

  grad = efc_Ma_in[worldid, dofid] - qfrc_smooth_in[worldid, dofid] - qfrc_constraint_in[worldid, dofid]
  ctx_grad_out[worldid, dofid] = grad
  wp.atomic_add(ctx_grad_dot_out, worldid, grad * grad)


@wp.kernel
def update_gradient_set_h_qM_lower_sparse(
  # Model:
  qM_fullm_i: wp.array[int],
  qM_fullm_j: wp.array[int],
  # Data in:
  qM_in: wp.array3d[float],
  # In:
  ctx_done_in: wp.array[bool],
  # Out:
  ctx_h_out: wp.array3d[float],
):
  worldid, elementid = wp.tid()

  if ctx_done_in[worldid]:
    return

  i = qM_fullm_i[elementid]
  j = qM_fullm_j[elementid]
  ctx_h_out[worldid, i, j] += qM_in[worldid, 0, elementid]


@wp.func
def state_check(D: float, state: int) -> float:
  if state == types.ConstraintState.QUADRATIC.value:
    return D
  else:
    return 0.0


@wp.func
def active_check(tid: int, threshold: int) -> float:
  if tid >= threshold:
    return 0.0
  else:
    return 1.0


@cache_kernel
def update_gradient_JTDAJ_sparse_tiled(tile_size: int, njmax: int):
  TILE_SIZE = tile_size

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_state_in: wp.array2d[int],
    # In:
    ctx_done_in: wp.array[bool],
    # Out:
    ctx_h_out: wp.array3d[float],
  ):
    worldid, elementid = wp.tid()

    if ctx_done_in[worldid]:
      return

    nefc = nefc_in[worldid]

    # get lower diagonal index
    i = (int(sqrt(float(1 + 8 * elementid))) - 1) // 2
    j = elementid - (i * (i + 1)) // 2

    offset_i = i * TILE_SIZE
    offset_j = j * TILE_SIZE

    sum_val = wp.tile_zeros(shape=(TILE_SIZE, TILE_SIZE), dtype=wp.float32)

    # Each tile processes looping over all constraints, producing 1 output tile
    for k in range(0, njmax, TILE_SIZE):
      if k >= nefc:
        break

      # AD: leaving bounds-check disabled here because I'm not entirely sure that
      # everything always hits the fast path. The padding takes care of any
      # potential OOB accesses.
      J_ki = wp.tile_load(efc_J_in[worldid], shape=(TILE_SIZE, TILE_SIZE), offset=(k, offset_i), bounds_check=False)

      if offset_i != offset_j:
        J_kj = wp.tile_load(efc_J_in[worldid], shape=(TILE_SIZE, TILE_SIZE), offset=(k, offset_j), bounds_check=False)
      else:
        wp.tile_assign(J_kj, J_ki, (0, 0))

      D_k = wp.tile_load(efc_D_in[worldid], shape=TILE_SIZE, offset=k, bounds_check=False)
      state = wp.tile_load(efc_state_in[worldid], shape=TILE_SIZE, offset=k, bounds_check=False)

      D_k = wp.tile_map(state_check, D_k, state)

      # force unused elements to be zero
      tid_tile = wp.tile_arange(TILE_SIZE, dtype=int)
      threshold_tile = wp.tile_ones(shape=TILE_SIZE, dtype=int) * (nefc - k)

      active_tile = wp.tile_map(active_check, tid_tile, threshold_tile)
      D_k = wp.tile_map(wp.mul, active_tile, D_k)

      J_ki = wp.tile_map(wp.mul, wp.tile_transpose(J_ki), wp.tile_broadcast(D_k, shape=(TILE_SIZE, TILE_SIZE)))

      sum_val += wp.tile_matmul(J_ki, J_kj)

    # AD: setting bounds_check to True explicitly here because for some reason it was
    # slower to disable it.
    wp.tile_store(ctx_h_out[worldid], sum_val, offset=(offset_i, offset_j), bounds_check=True)

  return kernel


@cache_kernel
def update_gradient_JTDAJ_dense_tiled(nv_pad: int, tile_size: int, njmax: int):
  if njmax < tile_size:
    tile_size = njmax

  TILE_SIZE_K = tile_size

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    qM_in: wp.array3d[float],
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_state_in: wp.array2d[int],
    # In:
    ctx_done_in: wp.array[bool],
    # Out:
    ctx_h_out: wp.array3d[float],
  ):
    worldid = wp.tid()

    if ctx_done_in[worldid]:
      return

    nefc = nefc_in[worldid]

    sum_val = wp.tile_load(qM_in[worldid], shape=(nv_pad, nv_pad), bounds_check=True)

    # Each tile processes one output tile by looping over all constraints
    for k in range(0, njmax, TILE_SIZE_K):
      if k >= nefc:
        break

      # AD: leaving bounds-check disabled here because I'm not entirely sure that
      # everything always hits the fast path. The padding takes care of any
      #  potential OOB accesses.
      J_kj = wp.tile_load(efc_J_in[worldid], shape=(TILE_SIZE_K, nv_pad), offset=(k, 0), bounds_check=False)

      # state check
      D_k = wp.tile_load(efc_D_in[worldid], shape=TILE_SIZE_K, offset=k, bounds_check=False)
      state = wp.tile_load(efc_state_in[worldid], shape=TILE_SIZE_K, offset=k, bounds_check=False)

      D_k = wp.tile_map(state_check, D_k, state)

      # force unused elements to be zero
      tid_tile = wp.tile_arange(TILE_SIZE_K, dtype=int)
      threshold_tile = wp.tile_ones(shape=TILE_SIZE_K, dtype=int) * (nefc - k)

      active_tile = wp.tile_map(active_check, tid_tile, threshold_tile)
      D_k = wp.tile_map(wp.mul, active_tile, D_k)

      J_ki = wp.tile_map(wp.mul, wp.tile_transpose(J_kj), wp.tile_broadcast(D_k, shape=(nv_pad, TILE_SIZE_K)))

      sum_val += wp.tile_matmul(J_ki, J_kj)

    wp.tile_store(ctx_h_out[worldid], sum_val, bounds_check=False)

  return kernel


# TODO(thowell): combine with JTDAJ ?
@wp.kernel
def update_gradient_JTCJ_sparse(
  # Model:
  opt_impratio_invsqrt: wp.array[float],
  dof_tri_row: wp.array[int],
  dof_tri_col: wp.array[int],
  # Data in:
  contact_dist_in: wp.array[float],
  contact_includemargin_in: wp.array[float],
  contact_friction_in: wp.array[types.vec5],
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  efc_J_in: wp.array3d[float],
  efc_D_in: wp.array2d[float],
  efc_state_in: wp.array2d[int],
  naconmax_in: int,
  nacon_in: wp.array[int],
  # In:
  ctx_Jaref_in: wp.array2d[float],
  ctx_done_in: wp.array[bool],
  nblocks_perblock: int,
  dim_block: int,
  # Out:
  ctx_h_out: wp.array3d[float],
):
  conid_start, elementid = wp.tid()

  dof1id = dof_tri_row[elementid]
  dof2id = dof_tri_col[elementid]

  for i in range(nblocks_perblock):
    conid = conid_start + i * dim_block

    if conid >= min(nacon_in[0], naconmax_in):
      return

    worldid = contact_worldid_in[conid]
    if ctx_done_in[worldid]:
      continue

    condim = contact_dim_in[conid]

    if condim == 1:
      continue

    # check contact status
    if contact_dist_in[conid] - contact_includemargin_in[conid] >= 0.0:
      continue

    efcid0 = contact_efc_address_in[conid, 0]
    if efc_state_in[worldid, efcid0] != types.ConstraintState.CONE:
      continue

    # All dims share the same sparsity pattern. Scan colind once to find
    # the sparse positions of dof1id and dof2id. Skip if either is absent.
    rownnz = efc_J_rownnz_in[worldid, efcid0]
    rowadr0 = efc_J_rowadr_in[worldid, efcid0]
    pos1 = int(-1)
    pos2 = int(-1)
    for k in range(rownnz):
      col = efc_J_colind_in[worldid, 0, rowadr0 + k]
      if col == dof1id:
        pos1 = k
      if col == dof2id:
        pos2 = k
      if pos1 >= 0 and pos2 >= 0:
        break
    if pos1 < 0 or pos2 < 0:
      continue

    fri = contact_friction_in[conid]
    mu = fri[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]

    mu2 = mu * mu
    dm = math.safe_div(efc_D_in[worldid, efcid0], mu2 * (1.0 + mu2))

    if dm == 0.0:
      continue

    n = ctx_Jaref_in[worldid, efcid0] * mu
    u = types.vec6(n, 0.0, 0.0, 0.0, 0.0, 0.0)

    tt = float(0.0)
    for j in range(1, condim):
      efcidj = contact_efc_address_in[conid, j]
      uj = ctx_Jaref_in[worldid, efcidj] * fri[j - 1]
      tt += uj * uj
      u[j] = uj

    if tt <= 0.0:
      t = 0.0
    else:
      t = wp.sqrt(tt)
    t = wp.max(t, types.MJ_MINVAL)
    ttt = wp.max(t * t * t, types.MJ_MINVAL)

    # Precompute common subexpressions.
    mu_over_t = math.safe_div(mu, t)
    mu_n_over_ttt = mu * math.safe_div(n, ttt)
    mu2_minus_mu_n_over_t = mu2 - mu * math.safe_div(n, t)

    h = float(0.0)

    for dim1id in range(condim):
      if dim1id == 0:
        rowadr1 = rowadr0
        dm_fri1 = dm * mu
      else:
        efcid1 = contact_efc_address_in[conid, dim1id]
        rowadr1 = efc_J_rowadr_in[worldid, efcid1]
        dm_fri1 = dm * fri[dim1id - 1]

      # Direct J reads using cached sparse positions.
      efc_J11 = efc_J_in[worldid, 0, rowadr1 + pos1]
      efc_J12 = efc_J_in[worldid, 0, rowadr1 + pos2]

      ui = u[dim1id]

      for dim2id in range(0, dim1id + 1):
        if dim2id == 0:
          rowadr2 = rowadr0
          dm_fri12 = dm_fri1 * mu
        else:
          efcid2 = contact_efc_address_in[conid, dim2id]
          rowadr2 = efc_J_rowadr_in[worldid, efcid2]
          dm_fri12 = dm_fri1 * fri[dim2id - 1]

        # Direct J reads using cached sparse positions.
        efc_J21 = efc_J_in[worldid, 0, rowadr2 + pos1]
        efc_J22 = efc_J_in[worldid, 0, rowadr2 + pos2]

        uj = u[dim2id]

        # set first row/column: (1, -mu/t * u)
        if dim1id == 0 and dim2id == 0:
          hcone = 1.0
        elif dim1id == 0:
          hcone = -mu_over_t * uj
        elif dim2id == 0:
          hcone = -mu_over_t * ui
        else:
          hcone = mu_n_over_ttt * ui * uj

          # add to diagonal: mu^2 - mu * n / t
          if dim1id == dim2id:
            hcone += mu2_minus_mu_n_over_t

        hcone *= dm_fri12

        if hcone != 0.0:
          h += hcone * efc_J11 * efc_J22

          if dim1id != dim2id:
            h += hcone * efc_J12 * efc_J21

    ctx_h_out[worldid, dof1id, dof2id] += h


@wp.kernel
def update_gradient_JTCJ_dense(
  # Model:
  opt_impratio_invsqrt: wp.array[float],
  dof_tri_row: wp.array[int],
  dof_tri_col: wp.array[int],
  # Data in:
  contact_dist_in: wp.array[float],
  contact_includemargin_in: wp.array[float],
  contact_friction_in: wp.array[types.vec5],
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  efc_J_in: wp.array3d[float],
  efc_D_in: wp.array2d[float],
  efc_state_in: wp.array2d[int],
  naconmax_in: int,
  nacon_in: wp.array[int],
  # In:
  ctx_Jaref_in: wp.array2d[float],
  ctx_done_in: wp.array[bool],
  nblocks_perblock: int,
  dim_block: int,
  # Out:
  ctx_h_out: wp.array3d[float],
):
  conid_start, elementid = wp.tid()

  dof1id = dof_tri_row[elementid]
  dof2id = dof_tri_col[elementid]

  for i in range(nblocks_perblock):
    conid = conid_start + i * dim_block

    if conid >= min(nacon_in[0], naconmax_in):
      return

    worldid = contact_worldid_in[conid]
    if ctx_done_in[worldid]:
      continue

    condim = contact_dim_in[conid]

    if condim == 1:
      continue

    # check contact status
    if contact_dist_in[conid] - contact_includemargin_in[conid] >= 0.0:
      continue

    efcid0 = contact_efc_address_in[conid, 0]
    if efc_state_in[worldid, efcid0] != types.ConstraintState.CONE:
      continue

    fri = contact_friction_in[conid]
    mu = fri[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]

    mu2 = mu * mu
    dm = math.safe_div(efc_D_in[worldid, efcid0], mu2 * (1.0 + mu2))

    if dm == 0.0:
      continue

    n = ctx_Jaref_in[worldid, efcid0] * mu
    u = types.vec6(n, 0.0, 0.0, 0.0, 0.0, 0.0)

    tt = float(0.0)
    for j in range(1, condim):
      efcidj = contact_efc_address_in[conid, j]
      uj = ctx_Jaref_in[worldid, efcidj] * fri[j - 1]
      tt += uj * uj
      u[j] = uj

    if tt <= 0.0:
      t = 0.0
    else:
      t = wp.sqrt(tt)
    t = wp.max(t, types.MJ_MINVAL)
    ttt = wp.max(t * t * t, types.MJ_MINVAL)

    h = float(0.0)

    for dim1id in range(condim):
      if dim1id == 0:
        efcid1 = efcid0
      else:
        efcid1 = contact_efc_address_in[conid, dim1id]

      efc_J11 = efc_J_in[worldid, efcid1, dof1id]
      efc_J12 = efc_J_in[worldid, efcid1, dof2id]

      ui = u[dim1id]

      for dim2id in range(0, dim1id + 1):
        if dim2id == 0:
          efcid2 = efcid0
        else:
          efcid2 = contact_efc_address_in[conid, dim2id]

        efc_J21 = efc_J_in[worldid, efcid2, dof1id]
        efc_J22 = efc_J_in[worldid, efcid2, dof2id]

        uj = u[dim2id]

        # set first row/column: (1, -mu/t * u)
        if dim1id == 0 and dim2id == 0:
          hcone = 1.0
        elif dim1id == 0:
          hcone = -math.safe_div(mu, t) * uj
        elif dim2id == 0:
          hcone = -math.safe_div(mu, t) * ui
        else:
          hcone = mu * math.safe_div(n, ttt) * ui * uj

          # add to diagonal: mu^2 - mu * n / t
          if dim1id == dim2id:
            hcone += mu2 - mu * math.safe_div(n, t)

        # pre and post multiply by diag(mu, friction) scale by dm
        if dim1id == 0:
          fri1 = mu
        else:
          fri1 = fri[dim1id - 1]

        if dim2id == 0:
          fri2 = mu
        else:
          fri2 = fri[dim2id - 1]

        hcone *= dm * fri1 * fri2

        if hcone != 0.0:
          h += hcone * efc_J11 * efc_J22

          if dim1id != dim2id:
            h += hcone * efc_J12 * efc_J21

    ctx_h_out[worldid, dof1id, dof2id] += h


@cache_kernel
def update_gradient_cholesky(tile_size: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # In:
    ctx_grad_in: wp.array2d[float],
    h_in: wp.array3d[float],
    ctx_done_in: wp.array[bool],
    # Out:
    ctx_Mgrad_out: wp.array2d[float],
  ):
    worldid = wp.tid()
    TILE_SIZE = wp.static(tile_size)

    if ctx_done_in[worldid]:
      return

    mat_tile = wp.tile_load(h_in[worldid], shape=(TILE_SIZE, TILE_SIZE))
    fact_tile = wp.tile_cholesky(mat_tile)
    input_tile = wp.tile_load(ctx_grad_in[worldid], shape=TILE_SIZE)
    output_tile = wp.tile_cholesky_solve(fact_tile, input_tile)
    wp.tile_store(ctx_Mgrad_out[worldid], output_tile)

  return kernel


@cache_kernel
def update_gradient_cholesky_blocked(tile_size: int, matrix_size: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # In:
    ctx_done_in: wp.array[bool],
    ctx_grad_in: wp.array3d[float],
    ctx_h_in: wp.array3d[float],
    ctx_hfactor: wp.array3d[float],
    # Out:
    ctx_Mgrad_out: wp.array3d[float],
  ):
    worldid = wp.tid()
    TILE_SIZE = wp.static(tile_size)

    if ctx_done_in[worldid]:
      return

    # We need matrix size both as a runtime input as well as a static input:
    # static input is needed to specify the tile sizes for the compiler
    # runtime input is needed for the loop bounds, otherwise warp will unroll
    # unconditionally leading to shared memory capacity issues.

    wp.static(create_blocked_cholesky_func(TILE_SIZE))(ctx_h_in[worldid], matrix_size, ctx_hfactor[worldid])
    wp.static(create_blocked_cholesky_solve_func(TILE_SIZE, matrix_size))(
      ctx_hfactor[worldid], ctx_grad_in[worldid], matrix_size, ctx_Mgrad_out[worldid]
    )

  return kernel


def update_gradient_cholesky_blocked_skip_unchanged(tile_size: int, matrix_size: int):
  """Blocked Cholesky that skips factorization when no constraints changed."""

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # In:
    ctx_done_in: wp.array[bool],
    ctx_grad_in: wp.array3d[float],
    ctx_h_in: wp.array3d[float],
    changed_count_in: wp.array[int],
    ctx_hfactor: wp.array3d[float],
    # Out:
    ctx_Mgrad_out: wp.array3d[float],
  ):
    worldid = wp.tid()
    TILE_SIZE = wp.static(tile_size)

    if ctx_done_in[worldid]:
      return

    if changed_count_in[worldid] > 0:
      wp.static(create_blocked_cholesky_func(TILE_SIZE))(ctx_h_in[worldid], matrix_size, ctx_hfactor[worldid])

    wp.static(create_blocked_cholesky_solve_func(TILE_SIZE, matrix_size))(
      ctx_hfactor[worldid], ctx_grad_in[worldid], matrix_size, ctx_Mgrad_out[worldid]
    )

  return kernel


@wp.kernel
def padding_h(nv: int, ctx_done_in: wp.array[bool], ctx_h_out: wp.array3d[float]):
  worldid, elementid = wp.tid()

  if ctx_done_in[worldid]:
    return

  dofid = nv + elementid
  ctx_h_out[worldid, dofid, dofid] = 1.0


def _cholesky_factorize_solve(m: types.Model, d: types.Data, ctx: SolverContext, skip_unchanged: bool = False):
  """Cholesky factorize ctx.h and solve for Mgrad.

  If skip_unchanged is True (blocked path only), worlds where no constraints
  changed reuse the cached factorization in hfactor instead of refactorizing.
  """
  if m.nv <= _BLOCK_CHOLESKY_DIM:
    wp.launch_tiled(
      update_gradient_cholesky(m.nv),
      dim=d.nworld,
      inputs=[ctx.grad, ctx.h, ctx.done],
      outputs=[ctx.Mgrad],
      block_dim=m.block_dim.update_gradient_cholesky,
    )
  else:
    wp.launch(
      padding_h,
      dim=(d.nworld, m.nv_pad - m.nv),
      inputs=[m.nv, ctx.done],
      outputs=[ctx.h],
    )

    if skip_unchanged:
      wp.launch_tiled(
        update_gradient_cholesky_blocked_skip_unchanged(types.TILE_SIZE_JTDAJ_DENSE, m.nv_pad),
        dim=d.nworld,
        inputs=[ctx.done, ctx.grad.reshape(shape=(d.nworld, ctx.grad.shape[1], 1)), ctx.h, ctx.changed_efc_count, ctx.hfactor],
        outputs=[ctx.Mgrad.reshape(shape=(d.nworld, ctx.Mgrad.shape[1], 1))],
        block_dim=m.block_dim.update_gradient_cholesky_blocked,
      )
    else:
      wp.launch_tiled(
        update_gradient_cholesky_blocked(types.TILE_SIZE_JTDAJ_DENSE, m.nv_pad),
        dim=d.nworld,
        inputs=[ctx.done, ctx.grad.reshape(shape=(d.nworld, ctx.grad.shape[1], 1)), ctx.h, ctx.hfactor],
        outputs=[ctx.Mgrad.reshape(shape=(d.nworld, ctx.Mgrad.shape[1], 1))],
        block_dim=m.block_dim.update_gradient_cholesky_blocked,
      )


@wp.kernel
def _JTDAJ_sparse(
  # Data in:
  nefc_in: wp.array[int],
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  efc_J_in: wp.array3d[float],
  efc_D_in: wp.array2d[float],
  efc_state_in: wp.array2d[int],
  # In:
  ctx_done_in: wp.array[bool],
  # Out:
  h_out: wp.array3d[float],
):
  worldid, efcid = wp.tid()

  if ctx_done_in[worldid]:
    return

  if efcid >= nefc_in[worldid]:
    return

  efc_D = efc_D_in[worldid, efcid]
  efc_state = efc_state_in[worldid, efcid]

  if state_check(efc_D, efc_state) == 0.0:
    return

  rownnz = efc_J_rownnz_in[worldid, efcid]
  rowadr = efc_J_rowadr_in[worldid, efcid]

  for i in range(rownnz):
    sparseidi = rowadr + i
    Ji = efc_J_in[worldid, 0, sparseidi]
    colindi = efc_J_colind_in[worldid, 0, sparseidi]
    for j in range(i, rownnz):
      if j == i:
        sparseidj = sparseidi
        Jj = Ji
        colindj = colindi
      else:
        sparseidj = rowadr + j
        Jj = efc_J_in[worldid, 0, sparseidj]
        colindj = efc_J_colind_in[worldid, 0, sparseidj]

      h = Ji * Jj * efc_D
      # Store in lower triangle only: ensure row >= col
      row = wp.max(colindi, colindj)
      col = wp.min(colindi, colindj)
      wp.atomic_add(h_out[worldid, row], col, h)


def _update_gradient(m: types.Model, d: types.Data, ctx: SolverContext):
  # grad = Ma - qfrc_smooth - qfrc_constraint
  wp.launch(update_gradient_zero_grad_dot, dim=(d.nworld), inputs=[ctx.done], outputs=[ctx.grad_dot])

  wp.launch(
    update_gradient_grad,
    dim=(d.nworld, m.nv),
    inputs=[d.qfrc_smooth, d.qfrc_constraint, d.efc.Ma, ctx.done],
    outputs=[ctx.grad, ctx.grad_dot],
  )

  if m.opt.solver == types.SolverType.CG:
    smooth.solve_m(m, d, ctx.Mgrad, ctx.grad)
  elif m.opt.solver == types.SolverType.NEWTON:
    # h = qM + (efc_J.T * efc_D * active) @ efc_J
    if m.is_sparse:
      ctx.h.zero_()
      wp.launch(
        _JTDAJ_sparse,
        dim=(d.nworld, d.njmax),
        inputs=[d.nefc, d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind, d.efc.J, d.efc.D, d.efc.state, ctx.done],
        outputs=[ctx.h],
      )

      wp.launch(
        update_gradient_set_h_qM_lower_sparse,
        dim=(d.nworld, m.qM_fullm_i.size),
        inputs=[m.qM_fullm_i, m.qM_fullm_j, d.qM, ctx.done],
        outputs=[ctx.h],
      )
    else:
      with scoped_mathdx_gemm_disabled():
        wp.launch_tiled(
          update_gradient_JTDAJ_dense_tiled(m.nv_pad, types.TILE_SIZE_JTDAJ_DENSE, d.njmax),
          dim=d.nworld,
          inputs=[
            d.nefc,
            d.qM,
            d.efc.J,
            d.efc.D,
            d.efc.state,
            ctx.done,
          ],
          outputs=[ctx.h],
          block_dim=m.block_dim.update_gradient_JTDAJ_dense,
        )

    if m.opt.cone == types.ConeType.ELLIPTIC:
      # Optimization: launching update_gradient_JTCJ with limited number of blocks on a GPU.
      # Profiling suggests that only a fraction of blocks out of the original
      # d.njmax blocks do the actual work. It aims to minimize #CTAs with no
      # effective work. It launches with #blocks that's proportional to the number
      # of SMs on the GPU. We can now query the SM count:
      # https://github.com/NVIDIA/warp/commit/f3814e7e5459e5fd13032cf0fddb3daddd510f30

      # make dim_block and nblocks_perblock static for update_gradient_JTCJ to allow
      # loop unrolling
      if wp.get_device().is_cuda:
        sm_count = wp.get_device().sm_count

        # Here we assume one block has 256 threads. We use a factor of 6, which
        # can be changed in the future to fine-tune the perf. The optimal factor will
        # depend on the kernel's occupancy, which determines how many blocks can
        # simultaneously run on the SM. TODO: This factor can be tuned further.
        dim_block = ceil((sm_count * 6 * 256) / m.dof_tri_row.size)
      else:
        # fall back for CPU
        dim_block = d.naconmax

      nblocks_perblock = int((d.naconmax + dim_block - 1) / dim_block)

      if m.is_sparse:
        wp.launch(
          update_gradient_JTCJ_sparse,
          dim=(dim_block, m.dof_tri_row.size),
          inputs=[
            m.opt.impratio_invsqrt,
            m.dof_tri_row,
            m.dof_tri_col,
            d.contact.dist,
            d.contact.includemargin,
            d.contact.friction,
            d.contact.dim,
            d.contact.efc_address,
            d.contact.worldid,
            d.efc.J_rownnz,
            d.efc.J_rowadr,
            d.efc.J_colind,
            d.efc.J,
            d.efc.D,
            d.efc.state,
            d.naconmax,
            d.nacon,
            ctx.Jaref,
            ctx.done,
            nblocks_perblock,
            dim_block,
          ],
          outputs=[ctx.h],
        )
      else:
        wp.launch(
          update_gradient_JTCJ_dense,
          dim=(dim_block, m.dof_tri_row.size),
          inputs=[
            m.opt.impratio_invsqrt,
            m.dof_tri_row,
            m.dof_tri_col,
            d.contact.dist,
            d.contact.includemargin,
            d.contact.friction,
            d.contact.dim,
            d.contact.efc_address,
            d.contact.worldid,
            d.efc.J,
            d.efc.D,
            d.efc.state,
            d.naconmax,
            d.nacon,
            ctx.Jaref,
            ctx.done,
            nblocks_perblock,
            dim_block,
          ],
          outputs=[ctx.h],
        )

    _cholesky_factorize_solve(m, d, ctx)
  else:
    raise ValueError(f"Unknown solver type: {m.opt.solver}")


def _update_gradient_incremental(m: types.Model, d: types.Data, ctx: SolverContext):
  """Incremental gradient update: update H for changed constraints + re-factorize.

  Skips the full J^T*D*J rebuild by applying only the delta from constraints
  that changed QUADRATIC state, then re-factorizes and solves.
  """
  wp.launch(update_gradient_zero_grad_dot, dim=(d.nworld), inputs=[ctx.done], outputs=[ctx.grad_dot])

  wp.launch(
    update_gradient_grad,
    dim=(d.nworld, m.nv),
    inputs=[d.qfrc_smooth, d.qfrc_constraint, d.efc.Ma, ctx.done],
    outputs=[ctx.grad, ctx.grad_dot],
  )

  # Update lower triangle of H with delta from changed constraints
  if m.is_sparse:
    wp.launch(
      update_gradient_h_incremental_sparse,
      dim=(d.nworld, ctx.changed_efc_ids.shape[1]),
      inputs=[
        d.efc.J_rownnz,
        d.efc.J_rowadr,
        d.efc.J_colind,
        d.efc.J,
        d.efc.D,
        d.efc.state,
        ctx.changed_efc_ids,
        ctx.changed_efc_count,
      ],
      outputs=[ctx.h],
    )
  else:
    lower_tri_dim = m.nv * (m.nv + 1) // 2
    wp.launch(
      update_gradient_h_incremental,
      dim=(d.nworld, lower_tri_dim),
      inputs=[
        d.efc.J,
        d.efc.D,
        d.efc.state,
        ctx.changed_efc_ids,
        ctx.changed_efc_count,
      ],
      outputs=[ctx.h],
    )

  _cholesky_factorize_solve(m, d, ctx, skip_unchanged=True)


@wp.kernel
def solve_prev_grad_Mgrad(
  # In:
  ctx_grad_in: wp.array2d[float],
  ctx_Mgrad_in: wp.array2d[float],
  ctx_done_in: wp.array[bool],
  # Out:
  ctx_prev_grad_out: wp.array2d[float],
  ctx_prev_Mgrad_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()

  if ctx_done_in[worldid]:
    return

  ctx_prev_grad_out[worldid, dofid] = ctx_grad_in[worldid, dofid]
  ctx_prev_Mgrad_out[worldid, dofid] = ctx_Mgrad_in[worldid, dofid]


@wp.kernel
def solve_beta(
  # Model:
  nv: int,
  # In:
  ctx_grad_in: wp.array2d[float],
  ctx_Mgrad_in: wp.array2d[float],
  ctx_prev_grad_in: wp.array2d[float],
  ctx_prev_Mgrad_in: wp.array2d[float],
  ctx_done_in: wp.array[bool],
  # Out:
  ctx_beta_out: wp.array[float],
):
  worldid = wp.tid()

  if ctx_done_in[worldid]:
    return

  beta_num = float(0.0)
  beta_den = float(0.0)
  for dofid in range(nv):
    prev_Mgrad = ctx_prev_Mgrad_in[worldid][dofid]
    beta_num += ctx_grad_in[worldid, dofid] * (ctx_Mgrad_in[worldid, dofid] - prev_Mgrad)
    beta_den += ctx_prev_grad_in[worldid, dofid] * prev_Mgrad

  ctx_beta_out[worldid] = wp.max(0.0, beta_num / wp.max(types.MJ_MINVAL, beta_den))


@wp.kernel
def solve_zero_search_dot(
  # In:
  ctx_done_in: wp.array[bool],
  # Out:
  ctx_search_dot_out: wp.array[float],
):
  worldid = wp.tid()

  if ctx_done_in[worldid]:
    return

  ctx_search_dot_out[worldid] = 0.0


@wp.kernel
def solve_search_update(
  # Model:
  opt_solver: int,
  # In:
  ctx_Mgrad_in: wp.array2d[float],
  ctx_search_in: wp.array2d[float],
  ctx_beta_in: wp.array[float],
  ctx_done_in: wp.array[bool],
  # Out:
  ctx_search_out: wp.array2d[float],
  ctx_search_dot_out: wp.array[float],
):
  worldid, dofid = wp.tid()

  if ctx_done_in[worldid]:
    return

  search = -1.0 * ctx_Mgrad_in[worldid, dofid]

  if opt_solver == types.SolverType.CG:
    search += ctx_beta_in[worldid] * ctx_search_in[worldid, dofid]

  ctx_search_out[worldid, dofid] = search
  wp.atomic_add(ctx_search_dot_out, worldid, search * search)


@wp.kernel
def solve_done(
  # Model:
  nv: int,
  opt_tolerance: wp.array[float],
  opt_iterations: int,
  stat_meaninertia: wp.array[float],
  # In:
  ctx_grad_dot_in: wp.array[float],
  ctx_cost_in: wp.array[float],
  ctx_prev_cost_in: wp.array[float],
  ctx_done_in: wp.array[bool],
  # Data out:
  solver_niter_out: wp.array[int],
  # Out:
  nsolving_out: wp.array[int],
  ctx_done_out: wp.array[bool],
):
  worldid = wp.tid()

  if ctx_done_in[worldid]:
    return

  solver_niter_out[worldid] += 1
  tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]
  meaninertia = stat_meaninertia[worldid % stat_meaninertia.shape[0]]

  improvement = _rescale(nv, meaninertia, ctx_prev_cost_in[worldid] - ctx_cost_in[worldid])
  gradient = _rescale(nv, meaninertia, wp.sqrt(ctx_grad_dot_in[worldid]))
  done = (improvement < tolerance) or (gradient < tolerance)
  if done or solver_niter_out[worldid] == opt_iterations:
    # if the solver has converged or the maximum number of iterations has been reached then
    # mark this world as done and remove it from the number of unconverged worlds
    ctx_done_out[worldid] = True
    wp.atomic_add(nsolving_out, 0, -1)


@event_scope
def _solver_iteration(
  m: types.Model,
  d: types.Data,
  ctx: SolverContext,
  step_size_cost: wp.array2d[float],
  nsolving: wp.array[int],
):
  _linesearch(m, d, ctx, step_size_cost)

  if m.opt.solver == types.SolverType.CG:
    wp.launch(
      solve_prev_grad_Mgrad,
      dim=(d.nworld, m.nv),
      inputs=[ctx.grad, ctx.Mgrad, ctx.done],
      outputs=[ctx.prev_grad, ctx.prev_Mgrad],
    )

  # Incremental H is only valid for non-elliptic cones. The elliptic cone
  # path in update_constraint_efc has early returns that skip state change
  # tracking, and the additional JTCJ Hessian term depends on Jaref which
  # changes every iteration.
  incremental = m.opt.solver == types.SolverType.NEWTON and m.opt.cone != types.ConeType.ELLIPTIC

  if incremental:
    # Must complete before update_constraint_efc which atomically increments.
    ctx.changed_efc_count.zero_()

  _update_constraint(m, d, ctx, track_changes=incremental)

  if incremental:
    _update_gradient_incremental(m, d, ctx)
  else:
    _update_gradient(m, d, ctx)

  # polak-ribiere
  if m.opt.solver == types.SolverType.CG:
    wp.launch(
      solve_beta,
      dim=d.nworld,
      inputs=[m.nv, ctx.grad, ctx.Mgrad, ctx.prev_grad, ctx.prev_Mgrad, ctx.done],
      outputs=[ctx.beta],
    )

  wp.launch(solve_zero_search_dot, dim=(d.nworld), inputs=[ctx.done], outputs=[ctx.search_dot])

  wp.launch(
    solve_search_update,
    dim=(d.nworld, m.nv),
    inputs=[m.opt.solver, ctx.Mgrad, ctx.search, ctx.beta, ctx.done],
    outputs=[ctx.search, ctx.search_dot],
  )

  wp.launch(
    solve_done,
    dim=d.nworld,
    inputs=[
      m.nv,
      m.opt.tolerance,
      m.opt.iterations,
      m.stat.meaninertia,
      ctx.grad_dot,
      ctx.cost,
      ctx.prev_cost,
      ctx.done,
    ],
    outputs=[d.solver_niter, nsolving, ctx.done],
  )


def init_context(m: types.Model, d: types.Data, ctx: SolverContext | InverseContext, grad: bool = True):
  # initialize some efc arrays
  wp.launch(
    solve_init_efc,
    dim=(d.nworld),
    outputs=[d.solver_niter, ctx.search_dot, ctx.cost, ctx.done],
  )

  # jaref = d.efc_J @ d.qacc - d.efc_aref

  # if we are only using 1 thread, it makes sense to do more dofs as we can also skip the
  # init kernel. For more than 1 thread, dofs_per_thread is lower for better load balancing.

  dofs_per_thread, threads_per_efc = _compute_jaref_threading(m.nv)
  # we need to clear the jaref array if we're doing atomic adds.
  if threads_per_efc > 1:
    ctx.Jaref.zero_()

  wp.launch(
    solve_init_jaref(m.is_sparse, m.nv, dofs_per_thread),
    dim=(d.nworld, d.njmax, threads_per_efc),
    inputs=[d.nefc, d.qacc, d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind, d.efc.J, d.efc.aref],
    outputs=[ctx.Jaref],
  )

  # Ma = qM @ qacc
  support.mul_m(m, d, d.efc.Ma, d.qacc, skip=ctx.done)

  _update_constraint(m, d, ctx)

  if grad:
    _update_gradient(m, d, ctx)


def _warmstart_qacc_init(m: types.Model, d: types.Data):
  """Copy qacc_warmstart or qacc_smooth into qacc based on WARMSTART flag."""
  if not (m.opt.disableflags & types.DisableBit.WARMSTART):
    wp.copy(d.qacc, d.qacc_warmstart)
  else:
    wp.copy(d.qacc, d.qacc_smooth)


def _compute_jaref_threading(nv: int) -> tuple[int, int]:
  """Compute jaref parallelization parameters.

  Returns (dofs_per_thread, threads_per_efc).
  """
  dofs_per_thread = 20 if nv > 50 else 50
  threads_per_efc = ceil(nv / dofs_per_thread)
  return dofs_per_thread, threads_per_efc


@event_scope
def solve(m: types.Model, d: types.Data):
  if d.njmax == 0 or m.nv == 0:
    wp.copy(d.qacc, d.qacc_smooth)
    d.solver_niter.fill_(0)
  elif m.opt.solver == types.SolverType.PGS:
    ctx = _create_pgs_context(m, d)
    _solve_pgs(m, d, ctx)
  else:
    ctx = create_solver_context(m, d)
    _solve(m, d, ctx)


def _solve(m: types.Model, d: types.Data, ctx: SolverContext):
  """Finds forces that satisfy constraints."""
  _warmstart_qacc_init(m, d)

  #  context
  init_context(m, d, ctx, grad=True)

  # search = -Mgrad
  wp.launch(
    solve_init_search,
    dim=(d.nworld, m.nv),
    inputs=[ctx.Mgrad],
    outputs=[ctx.search, ctx.search_dot],
  )

  step_size_cost = wp.empty((d.nworld, m.opt.ls_iterations if m.opt.ls_parallel else 0), dtype=float)

  nsolving = wp.full(shape=(1,), value=d.nworld, dtype=int)
  if m.opt.iterations != 0 and m.opt.graph_conditional:
    # Note: the iteration kernel (indicated by while_body) is repeatedly launched
    # as long as condition_iteration is not zero.
    # condition_iteration is a warp array of size 1 and type int, it counts the number
    # of worlds that are not converged, it becomes 0 when all worlds are converged.
    # When the number of iterations reaches m.opt.iterations, solver_niter
    # becomes zero and all worlds are marked as converged to avoid an infinite loop.
    # note: we only launch the iteration kernel if everything is not done
    wp.capture_while(
      nsolving, while_body=_solver_iteration, m=m, d=d, ctx=ctx, step_size_cost=step_size_cost, nsolving=nsolving
    )
  else:
    # This branch is mostly for when JAX is used as it is currently not compatible
    # with CUDA graph conditional.
    # It should be removed when JAX becomes compatible.
    for _ in range(m.opt.iterations):
      _solver_iteration(m, d, ctx, step_size_cost, nsolving)


@wp.kernel
def _pgs_build_constraint_groups(
  # Data in:
  nefc_in: wp.array[int],
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array2d[int],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  # Out:
  group_efc_start_out: wp.array2d[int],
  group_efc_count_out: wp.array2d[int],
  ngroups_out: wp.array[int],
):
  """Identify constraint groups for coloring.

  Multi-row constraints (connect, weld, contacts) become single groups.
  Single-row constraints (limits, friction, etc.) are individual groups.
  """
  worldid = wp.tid()
  nefc = nefc_in[worldid]
  ng = int(0)
  i = int(0)

  while i < nefc:
    typ = efc_type_in[worldid, i]

    if typ == types.ConstraintType.CONTACT_PYRAMIDAL:
      # Contiguous block: group all pyramidal faces for this contact
      con_id = efc_id_in[worldid, i]
      dim = contact_dim_in[con_id]
      count = int(0)
      if dim > 1:
        count = 2 * (dim - 1)
      else:
        count = 1
      # Only create group if this is the first row (idx0)
      idx0 = contact_efc_address_in[con_id, 0]
      if i == idx0:
        group_efc_start_out[worldid, ng] = i
        group_efc_count_out[worldid, ng] = count
        ng += 1
      i += 1

    elif typ == types.ConstraintType.CONTACT_ELLIPTIC:
      # Contiguous block: group all dim rows for this elliptic contact
      con_id = efc_id_in[worldid, i]
      idx0 = contact_efc_address_in[con_id, 0]
      if i == idx0:
        dim = contact_dim_in[con_id]
        group_efc_start_out[worldid, ng] = i
        group_efc_count_out[worldid, ng] = dim
        ng += 1
      i += 1

    elif typ == types.ConstraintType.EQUALITY:
      # Consecutive EQUALITY rows with same efc_id form one group
      # (connect = 3 rows, weld = 6 rows, joint/tendon = 1 row)
      eq_id = efc_id_in[worldid, i]
      count = int(1)
      done = int(0)
      while i + count < nefc and done == 0:
        if efc_type_in[worldid, i + count] != types.ConstraintType.EQUALITY:
          done = 1
        elif efc_id_in[worldid, i + count] != eq_id:
          done = 1
        else:
          count += 1

      group_efc_start_out[worldid, ng] = i
      group_efc_count_out[worldid, ng] = count
      ng += 1
      i += count

    else:
      # Single-row constraint (limit, friction, frictionless contact)
      group_efc_start_out[worldid, ng] = i
      group_efc_count_out[worldid, ng] = 1
      ng += 1
      i += 1

  ngroups_out[worldid] = ng


def _pgs_color_groups(nv: int, is_sparse: bool):
  """Factory for greedy coloring kernel using per-tree color lists.

  Combines sparse and dense Jacobian paths via compile-time branching.
  Conflict detection operates at tree granularity because the BT matrix
  (M⁻¹Jᵀ) couples all DOFs within a kinematic tree. Two constraints that
  touch any DOFs in the same tree must not be assigned the same color.

  Uses per-tree integer color lists to support any number of colors
  (up to pgs_max_colors). Groups that cannot be assigned a color (overflow)
  are marked with color = -1 for sequential processing.
  """
  NV = nv
  IS_SPARSE = is_sparse

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    dof_treeid: wp.array[int],
    # Data in:
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    # In:
    ngroups_in: wp.array[int],
    group_efc_start_in: wp.array2d[int],
    group_efc_count_in: wp.array2d[int],
    max_colors: int,
    # In/Out:
    tree_color_list_inout: wp.array3d[int],  # (nworld, ntree, max_alloc)
    tree_color_count_inout: wp.array2d[int],  # (nworld, ntree)
    color_used_inout: wp.array2d[int],  # (nworld, max_colors)
    tree_update_inout: wp.array2d[int],  # (nworld, ntree)
    # Out:
    node_color_out: wp.array2d[int],  # kernel_analyzer: ignore
    ncolors_out: wp.array[int],
  ):
    """Greedy coloring via per-tree integer color lists.

    For each group (sequentially within each world):
      1. Gather colors used by neighboring groups via tree color lists.
      2. Pick the smallest unused color.
      3. Update tree color lists for all touched trees.
      4. If no free color is found, mark the group with color = -1.

    Complexity: O(ngroups * avg_nnz_per_group * avg_colors_per_tree) per world.
    Each world runs as one thread; with thousands of worlds, GPU occupancy
    comes from inter-world parallelism.
    """
    worldid = wp.tid()
    ng = ngroups_in[worldid]
    max_color_so_far = int(-1)

    for group in range(ng):
      efc_start = group_efc_start_in[worldid, group]
      efc_count = group_efc_count_in[worldid, group]

      # Clear scratch up to max_color_so_far + 2
      for c in range(max_color_so_far + 2):
        color_used_inout[worldid, c] = 0  # kernel_analyzer: ignore

      # Gather used colors from all trees touched by this group
      for r in range(efc_count):
        efcid = efc_start + r
        if wp.static(IS_SPARSE):
          nnz = efc_J_rownnz_in[worldid, efcid]
          adr = efc_J_rowadr_in[worldid, efcid]
          for k in range(nnz):
            dof = efc_J_colind_in[worldid, 0, adr + k]
            treeid = dof_treeid[dof]
            for i in range(tree_color_count_inout[worldid, treeid]):
              c = tree_color_list_inout[worldid, treeid, i]
              color_used_inout[worldid, c] = 1  # kernel_analyzer: ignore
        else:
          for dof in range(NV):
            if efc_J_in[worldid, efcid, dof] != 0.0:
              treeid = dof_treeid[dof]
              for i in range(tree_color_count_inout[worldid, treeid]):
                c = tree_color_list_inout[worldid, treeid, i]
                color_used_inout[worldid, c] = 1  # kernel_analyzer: ignore

      # Find smallest unused color
      color = int(-1)
      for c in range(max_colors):
        if color < 0 and color_used_inout[worldid, c] == 0:
          color = c

      node_color_out[worldid, group] = color

      # Update tree color lists with this group's assigned color
      if color >= 0:
        if color > max_color_so_far:
          max_color_so_far = color
        for r in range(efc_count):
          efcid = efc_start + r
          if wp.static(IS_SPARSE):
            nnz = efc_J_rownnz_in[worldid, efcid]
            adr = efc_J_rowadr_in[worldid, efcid]
            for k in range(nnz):
              dof = efc_J_colind_in[worldid, 0, adr + k]
              treeid = dof_treeid[dof]
              if tree_update_inout[worldid, treeid] != group:
                tree_update_inout[worldid, treeid] = group  # kernel_analyzer: ignore
                idx = tree_color_count_inout[worldid, treeid]
                tree_color_list_inout[worldid, treeid, idx] = color  # kernel_analyzer: ignore
                tree_color_count_inout[worldid, treeid] = idx + 1  # kernel_analyzer: ignore
          else:
            for dof in range(NV):
              if efc_J_in[worldid, efcid, dof] != 0.0:
                treeid = dof_treeid[dof]
                if tree_update_inout[worldid, treeid] != group:
                  tree_update_inout[worldid, treeid] = group  # kernel_analyzer: ignore
                  idx = tree_color_count_inout[worldid, treeid]
                  tree_color_list_inout[worldid, treeid, idx] = color  # kernel_analyzer: ignore
                  tree_color_count_inout[worldid, treeid] = idx + 1  # kernel_analyzer: ignore

    ncolors_out[worldid] = max_color_so_far + 1

  return kernel


@wp.kernel
def _pgs_build_flat_schedule(
  # In:
  ngroups_in: wp.array[int],
  node_color_in: wp.array2d[int],
  ncolors_in: wp.array[int],
  max_ncolors: int,
  # Out:
  schedule_offsets_out: wp.array2d[int],  # (nworld, max_colors + 1)
  schedule_groups_out: wp.array2d[int],  # (nworld, njmax)
):
  """Build flat CSR schedule via counting sort by color.

  Groups for color c in world w are at:
    schedule_groups[w, offsets[w,c] : offsets[w,c+1]]

  Offsets for colors beyond this world's ncolors are filled with the
  final slot value so the sweep kernel sees empty ranges.
  """
  worldid = wp.tid()
  ng = ngroups_in[worldid]
  nc = ncolors_in[worldid]

  # Iterate colors in order, placing groups into the flat schedule
  slot = int(0)
  for c in range(nc):
    schedule_offsets_out[worldid, c] = slot
    for g in range(ng):
      if node_color_in[worldid, g] == c:
        schedule_groups_out[worldid, slot] = g
        slot += 1

  # Fill remaining offsets up to max_ncolors with final slot value
  for c in range(nc, max_ncolors + 1):
    schedule_offsets_out[worldid, c] = slot


@wp.kernel
def _pgs_build_overflow_list(
  # In:
  ngroups_in: wp.array[int],
  node_color_in: wp.array2d[int],
  # Out:
  overflow_list_out: wp.array2d[int],
  n_overflow_out: wp.array[int],
):
  """Build list of overflow (uncolored) groups.

  Groups with node_color == -1 could not be assigned a color and must be
  processed sequentially to avoid data races.
  """
  worldid, group_id = wp.tid()
  ng = ngroups_in[worldid]
  if group_id >= ng:
    return

  if node_color_in[worldid, group_id] < 0:
    slot = wp.atomic_add(n_overflow_out, worldid, 1)
    overflow_list_out[worldid, slot] = group_id


@event_scope
def _pgs_coloring(
  m: types.Model,
  d: types.Data,
  ctx,
):
  """Compute constraint coloring for parallel PGS sweep.

  Supports both sparse and dense Jacobians.

  Conflict detection operates at tree granularity: constraints that touch
  any DOFs in the same kinematic tree are assigned different colors. This
  is necessary because the BT matrix (M⁻¹Jᵀ) couples all DOFs within a tree.

  Groups that cannot be assigned a color (when ngroups > max_colors for
  single-tree models) are tracked in overflow_list for sequential processing.
  """
  max_colors = m.opt.pgs_max_colors

  # Step 1: Build constraint groups (GPU)
  wp.launch(
    _pgs_build_constraint_groups,
    dim=(d.nworld,),
    inputs=[
      d.nefc,
      d.contact.dim,
      d.contact.efc_address,
      d.efc.type,
      d.efc.id,
    ],
    outputs=[ctx.group_efc_start, ctx.group_efc_count, ctx.ngroups],
  )

  # Step 2: Greedy coloring via per-tree integer color lists (GPU)
  ctx.tree_color_count.zero_()
  ctx.tree_update_inout.fill_(-1)
  ctx.node_color.fill_(-1)
  wp.launch(
    _pgs_color_groups(m.nv, m.is_sparse),
    dim=(d.nworld,),
    inputs=[
      m.dof_treeid,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      d.efc.J,
      ctx.ngroups,
      ctx.group_efc_start,
      ctx.group_efc_count,
      max_colors,
    ],
    outputs=[
      ctx.tree_color_list,
      ctx.tree_color_count,
      ctx.color_used_inout,
      ctx.tree_update_inout,
      ctx.node_color,
      ctx.ncolors,
    ],
  )

  # Step 3: Build flat CSR schedule (GPU)
  wp.launch(
    _pgs_build_flat_schedule,
    dim=(d.nworld,),
    inputs=[ctx.ngroups, ctx.node_color, ctx.ncolors, max_colors],
    outputs=[ctx.schedule_offsets, ctx.schedule_groups],
  )

  # Step 5: Build overflow list for uncolored groups (GPU)
  ctx.n_overflow.zero_()
  wp.launch(
    _pgs_build_overflow_list,
    dim=(d.nworld, d.njmax),
    inputs=[ctx.ngroups, ctx.node_color],
    outputs=[ctx.overflow_list, ctx.n_overflow],
  )


@dataclasses.dataclass
class PGSContext:
  """Workspace arrays for the PGS (Projected Gauss-Seidel) dual solver."""

  BT: wp.array3d[float]  # M⁻¹·Jᵀ, shape (nworld, njmax, nv)
  AR_diag_inv: wp.array2d[float]  # 1/AR[i,i], shape (nworld, njmax)
  AR_elliptic: wp.array3d[float]  # upper-tri AR sub-block, shape (nworld, njmax, 21)
  done: wp.array[bool]  # convergence flag per world
  warmstart_cost: wp.array[float]  # warmstart cost, shape (nworld,)
  nsolving: wp.array[int]  # number of worlds still solving, shape (1,)
  # --- Sparse BT index (ntree > 1) ---
  BT_rownnz: wp.array2d[int]  # (nworld, njmax) nnz per BT row
  BT_rowadr: wp.array2d[int]  # (nworld, njmax) start addr in BT_colind
  BT_colind: wp.array3d[int]  # (nworld, 1, bt_nnz_max) column indices
  # --- Coloring workspace ---
  group_efc_start: wp.array2d  # (nworld, njmax) first efc row per group
  group_efc_count: wp.array2d  # (nworld, njmax) efc rows per group
  ngroups: wp.array  # (nworld,) groups per world
  tree_color_list: wp.array3d  # (nworld, ntree, max_alloc) color ints per tree
  tree_color_count: wp.array2d  # (nworld, ntree) colors assigned per tree
  color_used_inout: wp.array2d  # (nworld, max_colors) scratch for coloring
  tree_update_inout: wp.array2d  # (nworld, ntree) dedup scratch for coloring
  node_color: wp.array2d  # (nworld, njmax) color per group
  schedule_groups: wp.array2d  # (nworld, njmax) flat sorted group IDs
  schedule_offsets: wp.array2d  # (nworld, max_colors + 1) CSR offsets
  ncolors: wp.array  # (nworld,) actual colors used per world
  improvement: wp.array  # (nworld,) atomic convergence accumulator
  # --- Overflow (uncolored groups) ---
  overflow_list: wp.array2d  # (nworld, njmax) overflow group IDs
  n_overflow: wp.array  # (nworld,) count of overflow groups per world


def _create_pgs_context(m: types.Model, d: types.Data) -> PGSContext:
  """Create PGSContext with allocated workspace arrays."""
  nworld = d.nworld
  njmax = d.njmax
  max_colors = m.opt.pgs_max_colors
  max_colors_alloc = min(max_colors, njmax) if max_colors > 0 else 1
  use_sparse_bt = m.ntree > 1 and m.is_sparse
  bt_nnz_max = njmax * m.nv if use_sparse_bt else 0

  return PGSContext(
    BT=wp.zeros((nworld, njmax, m.nv), dtype=float),
    AR_diag_inv=wp.empty((nworld, njmax), dtype=float),
    AR_elliptic=wp.empty((nworld, njmax, 21), dtype=float),
    done=wp.empty((nworld,), dtype=bool),
    warmstart_cost=wp.empty((nworld,), dtype=float),
    nsolving=wp.empty((1,), dtype=int),
    BT_rownnz=wp.zeros((nworld, njmax), dtype=wp.int32) if use_sparse_bt else None,
    BT_rowadr=wp.zeros((nworld, njmax), dtype=wp.int32) if use_sparse_bt else None,
    BT_colind=wp.zeros((nworld, 1, bt_nnz_max), dtype=wp.int32) if use_sparse_bt else None,
    group_efc_start=wp.empty((nworld, njmax), dtype=wp.int32),
    group_efc_count=wp.empty((nworld, njmax), dtype=wp.int32),
    ngroups=wp.empty(nworld, dtype=wp.int32),
    tree_color_list=wp.empty((nworld, max(m.ntree, 1), max_colors_alloc), dtype=wp.int32),
    tree_color_count=wp.empty((nworld, max(m.ntree, 1)), dtype=wp.int32),
    color_used_inout=wp.empty((nworld, max(max_colors, 1)), dtype=wp.int32),
    tree_update_inout=wp.full((nworld, max(m.ntree, 1)), value=-1, dtype=wp.int32),
    node_color=wp.full((nworld, njmax), value=-1, dtype=wp.int32),
    schedule_groups=wp.empty((nworld, njmax), dtype=wp.int32),
    schedule_offsets=wp.empty((nworld, max(max_colors, 1) + 1), dtype=wp.int32),
    ncolors=wp.empty(nworld, dtype=wp.int32),
    improvement=wp.empty(nworld, dtype=float),
    overflow_list=wp.empty((nworld, njmax), dtype=wp.int32),
    n_overflow=wp.empty(nworld, dtype=wp.int32),
  )


def _pgs_precompute_AR_diag_inv(nv: int, is_sparse: bool):
  """Precompute 1/AR[i,i] for all constraints using J and BT (AR-free).

  AR[i,i] = J[i,:] · BT[i,:] + 1/D[i] is constant across iterations.
  Precomputing avoids an O(nv) dot product per 1D constraint per iteration.
  """
  IS_SPARSE = is_sparse

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    # In:
    BT_in: wp.array3d[float],
    # Out:
    AR_diag_inv_out: wp.array2d[float],
  ):
    worldid, efcid = wp.tid()
    NV = wp.static(nv)
    nefc = nefc_in[worldid]

    if efcid >= nefc:
      return

    # AR[i,i] = J[i,:] · BT[i,:] + 1/D[i]
    AR_ii = float(0.0)
    if wp.static(IS_SPARSE):
      rownnz = efc_J_rownnz_in[worldid, efcid]
      rowadr = efc_J_rowadr_in[worldid, efcid]
      for k in range(rownnz):
        sparseid = rowadr + k
        col = efc_J_colind_in[worldid, 0, sparseid]
        AR_ii += efc_J_in[worldid, 0, sparseid] * BT_in[worldid, efcid, col]
    else:
      for k in range(NV):
        AR_ii += efc_J_in[worldid, efcid, k] * BT_in[worldid, efcid, k]

    D_i = efc_D_in[worldid, efcid]
    if D_i > 0.0:
      AR_ii += 1.0 / D_i

    if AR_ii > 0.0:
      AR_diag_inv_out[worldid, efcid] = 1.0 / AR_ii
    else:
      AR_diag_inv_out[worldid, efcid] = 0.0

  return kernel


def _pgs_precompute_AR_elliptic(nv: int, is_sparse: bool):
  """Precompute AR sub-block for elliptic contacts.

  AR[di,dj] = J[idx_di,:] · BT[idx_dj,:] + δ(di,dj)/D[idx_di]
  Stores upper-triangle elements at the contact's normal-row efc index.

  Layout (21 elements max for dim=6):
    0:AR00 1:AR01 2:AR02 3:AR03 4:AR04 5:AR05
           6:AR11 7:AR12 8:AR13 9:AR14 10:AR15
                  11:AR22 12:AR23 13:AR24 14:AR25
                          15:AR33 16:AR34 17:AR35
                                  18:AR44 19:AR45
                                          20:AR55
  """
  IS_SPARSE = is_sparse

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    contact_dim_in: wp.array[int],
    contact_efc_address_in: wp.array2d[int],
    efc_type_in: wp.array2d[int],
    efc_id_in: wp.array2d[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    # In:
    BT_in: wp.array3d[float],
    # Out:
    AR_elliptic_out: wp.array3d[float],
  ):
    worldid, efcid = wp.tid()
    NV = wp.static(nv)
    nefc = nefc_in[worldid]

    if efcid >= nefc:
      return

    # Only process elliptic contact normal rows
    if efc_type_in[worldid, efcid] != types.ConstraintType.CONTACT_ELLIPTIC:
      return

    con_id = efc_id_in[worldid, efcid]
    idx0 = contact_efc_address_in[con_id, 0]

    # Only process from the normal row
    if efcid != idx0:
      return

    dim = contact_dim_in[con_id]

    # Look up indices for all dimensions
    idx1 = contact_efc_address_in[con_id, 1]
    idx2 = contact_efc_address_in[con_id, 2]
    idx3 = int(-1)
    idx4 = int(-1)
    idx5 = int(-1)
    if dim >= 4:
      idx3 = contact_efc_address_in[con_id, 3]
    if dim >= 6:
      idx4 = contact_efc_address_in[con_id, 4]
      idx5 = contact_efc_address_in[con_id, 5]

    # Initialize AR elements
    AR_00 = float(0.0)
    AR_01 = float(0.0)
    AR_02 = float(0.0)
    AR_03 = float(0.0)
    AR_04 = float(0.0)
    AR_05 = float(0.0)
    AR_11 = float(0.0)
    AR_12 = float(0.0)
    AR_13 = float(0.0)
    AR_14 = float(0.0)
    AR_15 = float(0.0)
    AR_22 = float(0.0)
    AR_23 = float(0.0)
    AR_24 = float(0.0)
    AR_25 = float(0.0)
    AR_33 = float(0.0)
    AR_34 = float(0.0)
    AR_35 = float(0.0)
    AR_44 = float(0.0)
    AR_45 = float(0.0)
    AR_55 = float(0.0)

    if wp.static(IS_SPARSE):
      # Row 0
      rnnz_0 = efc_J_rownnz_in[worldid, idx0]
      radr_0 = efc_J_rowadr_in[worldid, idx0]
      for k in range(rnnz_0):
        sid = radr_0 + k
        col = efc_J_colind_in[worldid, 0, sid]
        jv = efc_J_in[worldid, 0, sid]
        AR_00 += jv * BT_in[worldid, idx0, col]
        AR_01 += jv * BT_in[worldid, idx1, col]
        AR_02 += jv * BT_in[worldid, idx2, col]
        if dim >= 4:
          AR_03 += jv * BT_in[worldid, idx3, col]
        if dim >= 6:
          AR_04 += jv * BT_in[worldid, idx4, col]
          AR_05 += jv * BT_in[worldid, idx5, col]
      # Row 1
      rnnz_1 = efc_J_rownnz_in[worldid, idx1]
      radr_1 = efc_J_rowadr_in[worldid, idx1]
      for k in range(rnnz_1):
        sid = radr_1 + k
        col = efc_J_colind_in[worldid, 0, sid]
        jv = efc_J_in[worldid, 0, sid]
        AR_11 += jv * BT_in[worldid, idx1, col]
        AR_12 += jv * BT_in[worldid, idx2, col]
        if dim >= 4:
          AR_13 += jv * BT_in[worldid, idx3, col]
        if dim >= 6:
          AR_14 += jv * BT_in[worldid, idx4, col]
          AR_15 += jv * BT_in[worldid, idx5, col]
      # Row 2
      rnnz_2 = efc_J_rownnz_in[worldid, idx2]
      radr_2 = efc_J_rowadr_in[worldid, idx2]
      for k in range(rnnz_2):
        sid = radr_2 + k
        col = efc_J_colind_in[worldid, 0, sid]
        jv = efc_J_in[worldid, 0, sid]
        AR_22 += jv * BT_in[worldid, idx2, col]
        if dim >= 4:
          AR_23 += jv * BT_in[worldid, idx3, col]
        if dim >= 6:
          AR_24 += jv * BT_in[worldid, idx4, col]
          AR_25 += jv * BT_in[worldid, idx5, col]
      if dim >= 4:
        rnnz_3 = efc_J_rownnz_in[worldid, idx3]
        radr_3 = efc_J_rowadr_in[worldid, idx3]
        for k in range(rnnz_3):
          sid = radr_3 + k
          col = efc_J_colind_in[worldid, 0, sid]
          jv = efc_J_in[worldid, 0, sid]
          AR_33 += jv * BT_in[worldid, idx3, col]
          if dim >= 6:
            AR_34 += jv * BT_in[worldid, idx4, col]
            AR_35 += jv * BT_in[worldid, idx5, col]
      if dim >= 6:
        rnnz_4 = efc_J_rownnz_in[worldid, idx4]
        radr_4 = efc_J_rowadr_in[worldid, idx4]
        for k in range(rnnz_4):
          sid = radr_4 + k
          col = efc_J_colind_in[worldid, 0, sid]
          jv = efc_J_in[worldid, 0, sid]
          AR_44 += jv * BT_in[worldid, idx4, col]
          AR_45 += jv * BT_in[worldid, idx5, col]
        rnnz_5 = efc_J_rownnz_in[worldid, idx5]
        radr_5 = efc_J_rowadr_in[worldid, idx5]
        for k in range(rnnz_5):
          sid = radr_5 + k
          col = efc_J_colind_in[worldid, 0, sid]
          jv = efc_J_in[worldid, 0, sid]
          AR_55 += jv * BT_in[worldid, idx5, col]
    else:
      for k in range(NV):
        j0 = efc_J_in[worldid, idx0, k]
        AR_00 += j0 * BT_in[worldid, idx0, k]
        AR_01 += j0 * BT_in[worldid, idx1, k]
        AR_02 += j0 * BT_in[worldid, idx2, k]
        if dim >= 4:
          AR_03 += j0 * BT_in[worldid, idx3, k]
        if dim >= 6:
          AR_04 += j0 * BT_in[worldid, idx4, k]
          AR_05 += j0 * BT_in[worldid, idx5, k]
      for k in range(NV):
        j1 = efc_J_in[worldid, idx1, k]
        AR_11 += j1 * BT_in[worldid, idx1, k]
        AR_12 += j1 * BT_in[worldid, idx2, k]
        if dim >= 4:
          AR_13 += j1 * BT_in[worldid, idx3, k]
        if dim >= 6:
          AR_14 += j1 * BT_in[worldid, idx4, k]
          AR_15 += j1 * BT_in[worldid, idx5, k]
      for k in range(NV):
        j2 = efc_J_in[worldid, idx2, k]
        AR_22 += j2 * BT_in[worldid, idx2, k]
        if dim >= 4:
          AR_23 += j2 * BT_in[worldid, idx3, k]
        if dim >= 6:
          AR_24 += j2 * BT_in[worldid, idx4, k]
          AR_25 += j2 * BT_in[worldid, idx5, k]
      if dim >= 4:
        for k in range(NV):
          j3 = efc_J_in[worldid, idx3, k]
          AR_33 += j3 * BT_in[worldid, idx3, k]
          if dim >= 6:
            AR_34 += j3 * BT_in[worldid, idx4, k]
            AR_35 += j3 * BT_in[worldid, idx5, k]
      if dim >= 6:
        for k in range(NV):
          j4 = efc_J_in[worldid, idx4, k]
          AR_44 += j4 * BT_in[worldid, idx4, k]
          AR_45 += j4 * BT_in[worldid, idx5, k]
        for k in range(NV):
          AR_55 += efc_J_in[worldid, idx5, k] * BT_in[worldid, idx5, k]

    # Add 1/D to diagonal
    D_0 = efc_D_in[worldid, idx0]
    if D_0 > 0.0:
      AR_00 += 1.0 / D_0
    D_1 = efc_D_in[worldid, idx1]
    if D_1 > 0.0:
      AR_11 += 1.0 / D_1
    D_2 = efc_D_in[worldid, idx2]
    if D_2 > 0.0:
      AR_22 += 1.0 / D_2
    if dim >= 4:
      D_3 = efc_D_in[worldid, idx3]
      if D_3 > 0.0:
        AR_33 += 1.0 / D_3
    if dim >= 6:
      D_4 = efc_D_in[worldid, idx4]
      if D_4 > 0.0:
        AR_44 += 1.0 / D_4
      D_5 = efc_D_in[worldid, idx5]
      if D_5 > 0.0:
        AR_55 += 1.0 / D_5

    # Store upper-triangle elements
    AR_elliptic_out[worldid, idx0, 0] = AR_00
    AR_elliptic_out[worldid, idx0, 1] = AR_01
    AR_elliptic_out[worldid, idx0, 2] = AR_02
    AR_elliptic_out[worldid, idx0, 3] = AR_03
    AR_elliptic_out[worldid, idx0, 4] = AR_04
    AR_elliptic_out[worldid, idx0, 5] = AR_05
    AR_elliptic_out[worldid, idx0, 6] = AR_11
    AR_elliptic_out[worldid, idx0, 7] = AR_12
    AR_elliptic_out[worldid, idx0, 8] = AR_13
    AR_elliptic_out[worldid, idx0, 9] = AR_14
    AR_elliptic_out[worldid, idx0, 10] = AR_15
    AR_elliptic_out[worldid, idx0, 11] = AR_22
    AR_elliptic_out[worldid, idx0, 12] = AR_23
    AR_elliptic_out[worldid, idx0, 13] = AR_24
    AR_elliptic_out[worldid, idx0, 14] = AR_25
    AR_elliptic_out[worldid, idx0, 15] = AR_33
    AR_elliptic_out[worldid, idx0, 16] = AR_34
    AR_elliptic_out[worldid, idx0, 17] = AR_35
    AR_elliptic_out[worldid, idx0, 18] = AR_44
    AR_elliptic_out[worldid, idx0, 19] = AR_45
    AR_elliptic_out[worldid, idx0, 20] = AR_55

  return kernel


@wp.func
def _pgs_elliptic_normal_ray(
  # In:
  f0: float,
  f1: float,
  f2: float,
  f3: float,
  f4: float,
  f5: float,
  res0: float,
  res1: float,
  res2: float,
  res3: float,
  res4: float,
  res5: float,
  AR_00: float,
  AR_01: float,
  AR_02: float,
  AR_03: float,
  AR_04: float,
  AR_05: float,
  AR_11: float,
  AR_12: float,
  AR_13: float,
  AR_14: float,
  AR_15: float,
  AR_22: float,
  AR_23: float,
  AR_24: float,
  AR_25: float,
  AR_33: float,
  AR_34: float,
  AR_35: float,
  AR_44: float,
  AR_45: float,
  AR_55: float,
) -> tuple[float, float, float, float, float, float]:
  """Normal or ray update for elliptic cone constraint.

  If f0 < MJ_MINVAL: simple normal update, zero friction forces.
  Else: ray update using AR sub-block.
  Returns updated forces (new0..new5).
  """
  if f0 < MJ_MINVAL:
    new0 = f0 - res0 * (1.0 / AR_00 if AR_00 > 0.0 else 0.0)
    if new0 < 0.0:
      new0 = 0.0
    return new0, 0.0, 0.0, 0.0, 0.0, 0.0

  # AR * f (symmetric matvec)
  av0 = AR_00 * f0 + AR_01 * f1 + AR_02 * f2 + AR_03 * f3 + AR_04 * f4 + AR_05 * f5
  av1 = AR_01 * f0 + AR_11 * f1 + AR_12 * f2 + AR_13 * f3 + AR_14 * f4 + AR_15 * f5
  av2 = AR_02 * f0 + AR_12 * f1 + AR_22 * f2 + AR_23 * f3 + AR_24 * f4 + AR_25 * f5
  av3 = AR_03 * f0 + AR_13 * f1 + AR_23 * f2 + AR_33 * f3 + AR_34 * f4 + AR_35 * f5
  av4 = AR_04 * f0 + AR_14 * f1 + AR_24 * f2 + AR_34 * f3 + AR_44 * f4 + AR_45 * f5
  av5 = AR_05 * f0 + AR_15 * f1 + AR_25 * f2 + AR_35 * f3 + AR_45 * f4 + AR_55 * f5

  denom = f0 * av0 + f1 * av1 + f2 * av2 + f3 * av3 + f4 * av4 + f5 * av5
  vdotres = f0 * res0 + f1 * res1 + f2 * res2 + f3 * res3 + f4 * res4 + f5 * res5

  new0 = f0
  new1 = f1
  new2 = f2
  new3 = f3
  new4 = f4
  new5 = f5

  if denom >= MJ_MINVAL:
    x = -vdotres / denom
    if f0 + x * f0 < 0.0:
      x = -1.0
    new0 = f0 + x * f0
    new1 = f1 + x * f1
    new2 = f2 + x * f2
    new3 = f3 + x * f3
    new4 = f4 + x * f4
    new5 = f5 + x * f5

  return new0, new1, new2, new3, new4, new5


@wp.func
def _pgs_elliptic_friction(
  # In:
  dim: int,
  fn: float,
  old0: float,
  old1: float,
  old2: float,
  old3: float,
  old4: float,
  old5: float,
  res1: float,
  res2: float,
  res3: float,
  res4: float,
  res5: float,
  AR_01: float,
  AR_02: float,
  AR_03: float,
  AR_04: float,
  AR_05: float,
  AR_11: float,
  AR_12: float,
  AR_13: float,
  AR_14: float,
  AR_15: float,
  AR_22: float,
  AR_23: float,
  AR_24: float,
  AR_25: float,
  AR_33: float,
  AR_34: float,
  AR_35: float,
  AR_44: float,
  AR_45: float,
  AR_55: float,
  mu: types.vec5,
) -> tuple[float, float, float, float, float]:
  """QCQP friction update for elliptic cone constraint.

  Given updated normal force fn, solve for tangential forces.
  Returns (new1..new5).
  """
  if fn < MJ_MINVAL:
    return 0.0, 0.0, 0.0, 0.0, 0.0

  new1 = float(0.0)
  new2 = float(0.0)
  new3 = float(0.0)
  new4 = float(0.0)
  new5 = float(0.0)

  if dim == 3:
    bc0 = res1 - (AR_11 * old1 + AR_12 * old2) + AR_01 * (fn - old0)
    bc1 = res2 - (AR_12 * old1 + AR_22 * old2) + AR_02 * (fn - old0)
    v0, v1, active = util_solve.qcqp2(AR_11, AR_12, AR_22, bc0, bc1, mu[0], mu[1], fn)
    if active:
      s = v0 * v0 / (mu[0] * mu[0]) + v1 * v1 / (mu[1] * mu[1])
      s = wp.sqrt(fn * fn / wp.max(MJ_MINVAL, s))
      v0 *= s
      v1 *= s
    new1 = v0
    new2 = v1

  elif dim == 4:
    bc0 = res1 - (AR_11 * old1 + AR_12 * old2 + AR_13 * old3) + AR_01 * (fn - old0)
    bc1 = res2 - (AR_12 * old1 + AR_22 * old2 + AR_23 * old3) + AR_02 * (fn - old0)
    bc2 = res3 - (AR_13 * old1 + AR_23 * old2 + AR_33 * old3) + AR_03 * (fn - old0)
    v0, v1, v2, active = util_solve.qcqp3(
      AR_11,
      AR_12,
      AR_13,
      AR_22,
      AR_23,
      AR_33,
      bc0,
      bc1,
      bc2,
      mu[0],
      mu[1],
      mu[2],
      fn,
    )
    if active:
      s = v0 * v0 / (mu[0] * mu[0]) + v1 * v1 / (mu[1] * mu[1]) + v2 * v2 / (mu[2] * mu[2])
      s = wp.sqrt(fn * fn / wp.max(MJ_MINVAL, s))
      v0 *= s
      v1 *= s
      v2 *= s
    new1 = v0
    new2 = v1
    new3 = v2

  elif dim == 6:
    Ac00 = AR_11
    Ac01 = AR_12
    Ac02 = AR_13
    Ac03 = AR_14
    Ac04 = AR_15
    Ac11 = AR_22
    Ac12 = AR_23
    Ac13 = AR_24
    Ac14 = AR_25
    Ac22 = AR_33
    Ac23 = AR_34
    Ac24 = AR_35
    Ac33 = AR_44
    Ac34 = AR_45
    Ac44 = AR_55
    bc0 = res1 - (Ac00 * old1 + Ac01 * old2 + Ac02 * old3 + Ac03 * old4 + Ac04 * old5) + AR_01 * (fn - old0)
    bc1 = res2 - (Ac01 * old1 + Ac11 * old2 + Ac12 * old3 + Ac13 * old4 + Ac14 * old5) + AR_02 * (fn - old0)
    bc2 = res3 - (Ac02 * old1 + Ac12 * old2 + Ac22 * old3 + Ac23 * old4 + Ac24 * old5) + AR_03 * (fn - old0)
    bc3 = res4 - (Ac03 * old1 + Ac13 * old2 + Ac23 * old3 + Ac33 * old4 + Ac34 * old5) + AR_04 * (fn - old0)
    bc4 = res5 - (Ac04 * old1 + Ac14 * old2 + Ac24 * old3 + Ac34 * old4 + Ac44 * old5) + AR_05 * (fn - old0)
    v0, v1, v2, v3, v4, active = util_solve.qcqp5(
      Ac00,
      Ac01,
      Ac02,
      Ac03,
      Ac04,
      Ac11,
      Ac12,
      Ac13,
      Ac14,
      Ac22,
      Ac23,
      Ac24,
      Ac33,
      Ac34,
      Ac44,
      bc0,
      bc1,
      bc2,
      bc3,
      bc4,
      mu[0],
      mu[1],
      mu[2],
      mu[3],
      mu[4],
      fn,
    )
    if active:
      s = (
        v0 * v0 / (mu[0] * mu[0])
        + v1 * v1 / (mu[1] * mu[1])
        + v2 * v2 / (mu[2] * mu[2])
        + v3 * v3 / (mu[3] * mu[3])
        + v4 * v4 / (mu[4] * mu[4])
      )
      s = wp.sqrt(fn * fn / wp.max(MJ_MINVAL, s))
      v0 *= s
      v1 *= s
      v2 *= s
      v3 *= s
      v4 *= s
    new1 = v0
    new2 = v1
    new3 = v2
    new4 = v3
    new5 = v4

  return new1, new2, new3, new4, new5


@wp.func
def _pgs_elliptic_cost_change(
  # In:
  d0: float,
  d1: float,
  d2: float,
  d3: float,
  d4: float,
  d5: float,
  res0: float,
  res1: float,
  res2: float,
  res3: float,
  res4: float,
  res5: float,
  AR_00: float,
  AR_01: float,
  AR_02: float,
  AR_03: float,
  AR_04: float,
  AR_05: float,
  AR_11: float,
  AR_12: float,
  AR_13: float,
  AR_14: float,
  AR_15: float,
  AR_22: float,
  AR_23: float,
  AR_24: float,
  AR_25: float,
  AR_33: float,
  AR_34: float,
  AR_35: float,
  AR_44: float,
  AR_45: float,
  AR_55: float,
) -> float:
  """Compute elliptic cost change = 0.5 * d' * AR * d + d' * res."""
  ad0 = AR_00 * d0 + AR_01 * d1 + AR_02 * d2 + AR_03 * d3 + AR_04 * d4 + AR_05 * d5
  ad1 = AR_01 * d0 + AR_11 * d1 + AR_12 * d2 + AR_13 * d3 + AR_14 * d4 + AR_15 * d5
  ad2 = AR_02 * d0 + AR_12 * d1 + AR_22 * d2 + AR_23 * d3 + AR_24 * d4 + AR_25 * d5
  ad3 = AR_03 * d0 + AR_13 * d1 + AR_23 * d2 + AR_33 * d3 + AR_34 * d4 + AR_35 * d5
  ad4 = AR_04 * d0 + AR_14 * d1 + AR_24 * d2 + AR_34 * d3 + AR_44 * d4 + AR_45 * d5
  ad5 = AR_05 * d0 + AR_15 * d1 + AR_25 * d2 + AR_35 * d3 + AR_45 * d4 + AR_55 * d5
  return (
    0.5 * (d0 * ad0 + d1 * ad1 + d2 * ad2 + d3 * ad3 + d4 * ad4 + d5 * ad5)
    + d0 * res0
    + d1 * res1
    + d2 * res2
    + d3 * res3
    + d4 * res4
    + d5 * res5
  )


# kernel_analyzer: off
@wp.func
def _pgs_solve_elliptic_cone(
  # Constraint info:
  dim: int,
  # Old forces:
  old0: float,
  old1: float,
  old2: float,
  old3: float,
  old4: float,
  old5: float,
  # Residuals:
  res0: float,
  res1: float,
  res2: float,
  res3: float,
  res4: float,
  res5: float,
  # Precomputed AR sub-block array:
  AR_elliptic_in: wp.array3d[float],
  worldid: int,
  idx0: int,
  # Friction coefficients:
  mu: types.vec5,
  # SOR omega (1.0 for GS modes):
  sor_omega: float,
) -> tuple[float, float, float, float, float, float, float]:
  """Shared elliptic QCQP solve: load AR, normal/ray update, friction, SOR, cost change.

  Returns (new0, new1, new2, new3, new4, new5, cost_change).
  """
  # ---- Load precomputed AR sub-block (21 elements, upper triangle) ----
  AR_00 = AR_elliptic_in[worldid, idx0, 0]
  AR_01 = AR_elliptic_in[worldid, idx0, 1]
  AR_02 = AR_elliptic_in[worldid, idx0, 2]
  AR_03 = AR_elliptic_in[worldid, idx0, 3]
  AR_04 = AR_elliptic_in[worldid, idx0, 4]
  AR_05 = AR_elliptic_in[worldid, idx0, 5]
  AR_11 = AR_elliptic_in[worldid, idx0, 6]
  AR_12 = AR_elliptic_in[worldid, idx0, 7]
  AR_13 = AR_elliptic_in[worldid, idx0, 8]
  AR_14 = AR_elliptic_in[worldid, idx0, 9]
  AR_15 = AR_elliptic_in[worldid, idx0, 10]
  AR_22 = AR_elliptic_in[worldid, idx0, 11]
  AR_23 = AR_elliptic_in[worldid, idx0, 12]
  AR_24 = AR_elliptic_in[worldid, idx0, 13]
  AR_25 = AR_elliptic_in[worldid, idx0, 14]
  AR_33 = AR_elliptic_in[worldid, idx0, 15]
  AR_34 = AR_elliptic_in[worldid, idx0, 16]
  AR_35 = AR_elliptic_in[worldid, idx0, 17]
  AR_44 = AR_elliptic_in[worldid, idx0, 18]
  AR_45 = AR_elliptic_in[worldid, idx0, 19]
  AR_55 = AR_elliptic_in[worldid, idx0, 20]

  # ---- Normal/ray update ----
  new0, new1, new2, new3, new4, new5 = _pgs_elliptic_normal_ray(
    old0,
    old1,
    old2,
    old3,
    old4,
    old5,
    res0,
    res1,
    res2,
    res3,
    res4,
    res5,
    AR_00,
    AR_01,
    AR_02,
    AR_03,
    AR_04,
    AR_05,
    AR_11,
    AR_12,
    AR_13,
    AR_14,
    AR_15,
    AR_22,
    AR_23,
    AR_24,
    AR_25,
    AR_33,
    AR_34,
    AR_35,
    AR_44,
    AR_45,
    AR_55,
  )

  # ---- Friction update (QCQP) ----
  if old0 >= MJ_MINVAL:
    new1, new2, new3, new4, new5 = _pgs_elliptic_friction(
      dim,
      new0,
      old0,
      old1,
      old2,
      old3,
      old4,
      old5,
      res1,
      res2,
      res3,
      res4,
      res5,
      AR_01,
      AR_02,
      AR_03,
      AR_04,
      AR_05,
      AR_11,
      AR_12,
      AR_13,
      AR_14,
      AR_15,
      AR_22,
      AR_23,
      AR_24,
      AR_25,
      AR_33,
      AR_34,
      AR_35,
      AR_44,
      AR_45,
      AR_55,
      mu,
    )

  # ---- SOR blend (sor_omega=1.0 is identity for GS modes) ----
  new0 = (1.0 - sor_omega) * old0 + sor_omega * new0
  new1 = (1.0 - sor_omega) * old1 + sor_omega * new1
  new2 = (1.0 - sor_omega) * old2 + sor_omega * new2
  new3 = (1.0 - sor_omega) * old3 + sor_omega * new3
  new4 = (1.0 - sor_omega) * old4 + sor_omega * new4
  new5 = (1.0 - sor_omega) * old5 + sor_omega * new5

  # ---- Cost change ----
  d0 = new0 - old0
  d1 = new1 - old1
  d2 = new2 - old2
  d3 = new3 - old3
  d4 = new4 - old4
  d5 = new5 - old5
  change = _pgs_elliptic_cost_change(
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    res0,
    res1,
    res2,
    res3,
    res4,
    res5,
    AR_00,
    AR_01,
    AR_02,
    AR_03,
    AR_04,
    AR_05,
    AR_11,
    AR_12,
    AR_13,
    AR_14,
    AR_15,
    AR_22,
    AR_23,
    AR_24,
    AR_25,
    AR_33,
    AR_34,
    AR_35,
    AR_44,
    AR_45,
    AR_55,
  )

  return new0, new1, new2, new3, new4, new5, change


# kernel_analyzer: on


def _pgs_update_qacc_delta_func(nv: int, use_sparse_bt: bool):
  """Factory returning a @wp.func that applies a single-row qacc delta.

  qacc[worldid, :] += delta * BT[worldid, efcid, :]

  Uses sparse or dense BT based on use_sparse_bt (compile-time constant).
  """
  NV = nv
  USE_SPARSE_BT = use_sparse_bt

  # kernel_analyzer: off
  @wp.func
  def update_qacc_delta(
    worldid: int,
    efcid: int,
    delta: float,
    BT_in: wp.array3d[float],
    BT_rownnz_in: wp.array2d[int],
    BT_rowadr_in: wp.array2d[int],
    BT_colind_in: wp.array3d[int],
    qacc_inout: wp.array2d[float],
  ):
    if wp.static(USE_SPARSE_BT):
      bt_nnz = BT_rownnz_in[worldid, efcid]
      bt_adr = BT_rowadr_in[worldid, efcid]
      for k in range(bt_nnz):
        col = BT_colind_in[worldid, 0, bt_adr + k]
        qacc_inout[worldid, col] += delta * BT_in[worldid, efcid, col]  # kernel_analyzer: ignore
    else:
      for k in range(wp.static(NV)):
        qacc_inout[worldid, k] += delta * BT_in[worldid, efcid, k]  # kernel_analyzer: ignore

  return update_qacc_delta
  # kernel_analyzer: on


def _pgs_sweep_kernel(nv: int, is_sparse: bool, use_sparse_bt: bool):
  """One PGS Gauss-Seidel sweep over all constraints (AR-free).

  Sequential per world — processes constraints one at a time,
  following MuJoCo C's mj_solPGS implementation.

  Uses on-the-fly J·BT products instead of materialized AR matrix.
  Maintains qacc incrementally: qacc += delta · BT[i,:] after each update.

  For 1D constraints, uses precomputed 1/AR[i,i] to avoid redundant O(nv)
  dot products each iteration (AR diagonal is constant across iterations).

  Handles:
  - Equality constraints (no projection)
  - Friction constraints (interval [-floss, floss])
  - Limit and contact constraints (non-negativity [0, inf))
  - Elliptic cone constraints (QCQP sub-problems for dim 3, 4, 6)
  """
  IS_SPARSE = is_sparse
  USE_SPARSE_BT = use_sparse_bt
  update_qacc = _pgs_update_qacc_delta_func(nv, use_sparse_bt)

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(  # kernel_analyzer: ignore
    # Data in:
    ne_in: wp.array[int],
    nf_in: wp.array[int],
    nefc_in: wp.array[int],
    efc_type_in: wp.array2d[int],
    efc_frictionloss_in: wp.array2d[float],
    efc_id_in: wp.array2d[int],
    contact_dim_in: wp.array[int],
    contact_friction_in: wp.array[types.vec5],
    contact_efc_address_in: wp.array2d[int],
    # J (dense or sparse format):
    efc_J_in: wp.array3d[float],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    # In:
    BT_in: wp.array3d[float],
    # Sparse BT index (ntree > 1):
    BT_rownnz_in: wp.array2d[int],
    BT_rowadr_in: wp.array2d[int],
    BT_colind_in: wp.array3d[int],
    # Precomputed 1/AR[i,i] for 1D constraints:
    AR_diag_inv_in: wp.array2d[float],
    # Precomputed AR sub-block for elliptic contacts:
    AR_elliptic_in: wp.array3d[float],
    # Constraint params:
    efc_D_in: wp.array2d[float],
    efc_aref_in: wp.array2d[float],
    # Model:
    opt_tolerance: wp.array[float],
    stat_meaninertia: wp.array[float],
    maxiter: int,
    # Data in/out:
    efc_force_inout: wp.array2d[float],
    qacc_inout: wp.array2d[float],
    done_inout: wp.array[bool],
    solver_niter_inout: wp.array[int],
    nsolving_inout: wp.array[int],
    # Data out:
    efc_state_out: wp.array2d[int],
  ):
    worldid = wp.tid()

    if done_inout[worldid]:
      return

    nefc = nefc_in[worldid]
    ne = ne_in[worldid]
    nf = nf_in[worldid]
    NV = wp.static(nv)
    tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]

    improvement = float(0.0)

    i = int(0)
    while i < nefc:
      # Get constraint dimensionality
      dim = int(1)
      if efc_type_in[worldid, i] == types.ConstraintType.CONTACT_ELLIPTIC:
        con_id = efc_id_in[worldid, i]
        dim = contact_dim_in[con_id]

      # ---- Simple (1D) constraint ----
      if dim == 1:
        # Compute residual: res = J[i,:] · qacc - aref[i] + f[i]/D[i]
        res = float(0.0)
        if wp.static(IS_SPARSE):
          rownnz_i = efc_J_rownnz_in[worldid, i]
          rowadr_i = efc_J_rowadr_in[worldid, i]
          for k in range(rownnz_i):
            sparseid = rowadr_i + k
            col = efc_J_colind_in[worldid, 0, sparseid]
            res += efc_J_in[worldid, 0, sparseid] * qacc_inout[worldid, col]
        else:
          for k in range(NV):
            res += efc_J_in[worldid, i, k] * qacc_inout[worldid, k]
        res -= efc_aref_in[worldid, i]
        D_i = efc_D_in[worldid, i]
        if D_i > 0.0:
          res += efc_force_inout[worldid, i] / D_i

        # Use precomputed 1/AR[i,i] — avoids O(nv) dot product each iteration
        old_force = efc_force_inout[worldid, i]
        AR_diag_inv = AR_diag_inv_in[worldid, i]
        new_force = old_force - res * AR_diag_inv

        # Project onto constraint bounds
        if i >= ne and i < ne + nf:
          floss = efc_frictionloss_in[worldid, i]
          if new_force < -floss:
            new_force = -floss
          elif new_force > floss:
            new_force = floss
        elif i >= ne + nf:
          if new_force < 0.0:
            new_force = 0.0

        # Cost change: 0.5 * delta * AR[i,i] * delta + delta * res
        delta = new_force - old_force
        AR_ii = 1.0 / AR_diag_inv if AR_diag_inv > 0.0 else 0.0
        change = 0.5 * delta * delta * AR_ii + delta * res

        if change > tolerance:
          new_force = old_force
        else:
          improvement -= change
          # Update qacc incrementally
          actual_delta = new_force - old_force
          update_qacc(worldid, i, actual_delta, BT_in, BT_rownnz_in, BT_rowadr_in, BT_colind_in, qacc_inout)

        efc_force_inout[worldid, i] = new_force  # kernel_analyzer: ignore
        i += 1

      # ---- Elliptic cone constraint (multi-dim) ----
      else:
        con_id = efc_id_in[worldid, i]
        mu = contact_friction_in[con_id]

        # Indirect indexing: look up actual efcid for each dimension
        # Rows are contiguous (base_efcid + dim) due to block allocation
        idx0 = contact_efc_address_in[con_id, 0]
        idx1 = int(-1)
        idx2 = int(-1)
        idx3 = int(-1)
        idx4 = int(-1)
        idx5 = int(-1)
        if dim >= 3:
          idx1 = contact_efc_address_in[con_id, 1]
          idx2 = contact_efc_address_in[con_id, 2]
        if dim >= 4:
          idx3 = contact_efc_address_in[con_id, 3]
        if dim >= 6:
          idx4 = contact_efc_address_in[con_id, 4]
          idx5 = contact_efc_address_in[con_id, 5]

        # Skip non-normal rows — only process from the first (normal) row
        if i != idx0:
          i += dim  # skip entire block
          continue

        # ---- Compute residuals for all dim rows ----
        # res_d = J[idx_d,:] · qacc - aref[idx_d] + f[idx_d]/D[idx_d]
        res0 = float(0.0)
        res1 = float(0.0)
        res2 = float(0.0)
        res3 = float(0.0)
        res4 = float(0.0)
        res5 = float(0.0)

        # Row 0 (normal)
        if wp.static(IS_SPARSE):
          rnnz0 = efc_J_rownnz_in[worldid, idx0]
          radr0 = efc_J_rowadr_in[worldid, idx0]
          for k in range(rnnz0):
            sid = radr0 + k
            col = efc_J_colind_in[worldid, 0, sid]
            res0 += efc_J_in[worldid, 0, sid] * qacc_inout[worldid, col]
        else:
          for k in range(NV):
            res0 += efc_J_in[worldid, idx0, k] * qacc_inout[worldid, k]
        res0 -= efc_aref_in[worldid, idx0]
        D_0 = efc_D_in[worldid, idx0]
        if D_0 > 0.0:
          res0 += efc_force_inout[worldid, idx0] / D_0

        if dim >= 3:
          if wp.static(IS_SPARSE):
            rnnz1 = efc_J_rownnz_in[worldid, idx1]
            radr1 = efc_J_rowadr_in[worldid, idx1]
            for k in range(rnnz1):
              sid = radr1 + k
              col = efc_J_colind_in[worldid, 0, sid]
              res1 += efc_J_in[worldid, 0, sid] * qacc_inout[worldid, col]
          else:
            for k in range(NV):
              res1 += efc_J_in[worldid, idx1, k] * qacc_inout[worldid, k]
          res1 -= efc_aref_in[worldid, idx1]
          D_1 = efc_D_in[worldid, idx1]
          if D_1 > 0.0:
            res1 += efc_force_inout[worldid, idx1] / D_1

          if wp.static(IS_SPARSE):
            rnnz2 = efc_J_rownnz_in[worldid, idx2]
            radr2 = efc_J_rowadr_in[worldid, idx2]
            for k in range(rnnz2):
              sid = radr2 + k
              col = efc_J_colind_in[worldid, 0, sid]
              res2 += efc_J_in[worldid, 0, sid] * qacc_inout[worldid, col]
          else:
            for k in range(NV):
              res2 += efc_J_in[worldid, idx2, k] * qacc_inout[worldid, k]
          res2 -= efc_aref_in[worldid, idx2]
          D_2 = efc_D_in[worldid, idx2]
          if D_2 > 0.0:
            res2 += efc_force_inout[worldid, idx2] / D_2

        if dim >= 4:
          if wp.static(IS_SPARSE):
            rnnz3 = efc_J_rownnz_in[worldid, idx3]
            radr3 = efc_J_rowadr_in[worldid, idx3]
            for k in range(rnnz3):
              sid = radr3 + k
              col = efc_J_colind_in[worldid, 0, sid]
              res3 += efc_J_in[worldid, 0, sid] * qacc_inout[worldid, col]
          else:
            for k in range(NV):
              res3 += efc_J_in[worldid, idx3, k] * qacc_inout[worldid, k]
          res3 -= efc_aref_in[worldid, idx3]
          D_3 = efc_D_in[worldid, idx3]
          if D_3 > 0.0:
            res3 += efc_force_inout[worldid, idx3] / D_3

        if dim >= 6:
          if wp.static(IS_SPARSE):
            rnnz4 = efc_J_rownnz_in[worldid, idx4]
            radr4 = efc_J_rowadr_in[worldid, idx4]
            for k in range(rnnz4):
              sid = radr4 + k
              col = efc_J_colind_in[worldid, 0, sid]
              res4 += efc_J_in[worldid, 0, sid] * qacc_inout[worldid, col]
          else:
            for k in range(NV):
              res4 += efc_J_in[worldid, idx4, k] * qacc_inout[worldid, k]
          res4 -= efc_aref_in[worldid, idx4]
          D_4 = efc_D_in[worldid, idx4]
          if D_4 > 0.0:
            res4 += efc_force_inout[worldid, idx4] / D_4

          if wp.static(IS_SPARSE):
            rnnz5 = efc_J_rownnz_in[worldid, idx5]
            radr5 = efc_J_rowadr_in[worldid, idx5]
            for k in range(rnnz5):
              sid = radr5 + k
              col = efc_J_colind_in[worldid, 0, sid]
              res5 += efc_J_in[worldid, 0, sid] * qacc_inout[worldid, col]
          else:
            for k in range(NV):
              res5 += efc_J_in[worldid, idx5, k] * qacc_inout[worldid, k]
          res5 -= efc_aref_in[worldid, idx5]
          D_5 = efc_D_in[worldid, idx5]
          if D_5 > 0.0:
            res5 += efc_force_inout[worldid, idx5] / D_5

        # ---- Save old forces ----
        old0 = efc_force_inout[worldid, idx0]
        old1 = float(0.0)
        old2 = float(0.0)
        old3 = float(0.0)
        old4 = float(0.0)
        old5 = float(0.0)
        if dim >= 3:
          old1 = efc_force_inout[worldid, idx1]
          old2 = efc_force_inout[worldid, idx2]
        if dim >= 4:
          old3 = efc_force_inout[worldid, idx3]
        if dim >= 6:
          old4 = efc_force_inout[worldid, idx4]
          old5 = efc_force_inout[worldid, idx5]

        # ---- Solve elliptic QCQP (shared) ----
        new0, new1, new2, new3, new4, new5, change = _pgs_solve_elliptic_cone(
          dim,
          old0,
          old1,
          old2,
          old3,
          old4,
          old5,
          res0,
          res1,
          res2,
          res3,
          res4,
          res5,
          AR_elliptic_in,
          worldid,
          idx0,
          mu,
          1.0,
        )

        # ---- Accept or reject ----
        if change <= tolerance:
          improvement -= change
          d0 = new0 - old0
          d1 = new1 - old1
          d2 = new2 - old2
          d3 = new3 - old3
          d4 = new4 - old4
          d5 = new5 - old5
          efc_force_inout[worldid, idx0] = new0  # kernel_analyzer: ignore
          if dim >= 3:
            efc_force_inout[worldid, idx1] = new1  # kernel_analyzer: ignore
            efc_force_inout[worldid, idx2] = new2  # kernel_analyzer: ignore
          if dim >= 4:
            efc_force_inout[worldid, idx3] = new3  # kernel_analyzer: ignore
          if dim >= 6:
            efc_force_inout[worldid, idx4] = new4  # kernel_analyzer: ignore
            efc_force_inout[worldid, idx5] = new5  # kernel_analyzer: ignore
          # Update qacc
          if d0 != 0.0:
            update_qacc(worldid, idx0, d0, BT_in, BT_rownnz_in, BT_rowadr_in, BT_colind_in, qacc_inout)
          if dim >= 3:
            if d1 != 0.0:
              update_qacc(worldid, idx1, d1, BT_in, BT_rownnz_in, BT_rowadr_in, BT_colind_in, qacc_inout)
            if d2 != 0.0:
              update_qacc(worldid, idx2, d2, BT_in, BT_rownnz_in, BT_rowadr_in, BT_colind_in, qacc_inout)
          if dim >= 4:
            if d3 != 0.0:
              update_qacc(worldid, idx3, d3, BT_in, BT_rownnz_in, BT_rowadr_in, BT_colind_in, qacc_inout)
          if dim >= 6:
            if d4 != 0.0:
              update_qacc(worldid, idx4, d4, BT_in, BT_rownnz_in, BT_rowadr_in, BT_colind_in, qacc_inout)
            if d5 != 0.0:
              update_qacc(worldid, idx5, d5, BT_in, BT_rownnz_in, BT_rowadr_in, BT_colind_in, qacc_inout)

        i += dim

    # ---- Inline dual state ----
    for efcid in range(nefc):
      force = efc_force_inout[worldid, efcid]
      if efcid < ne:
        efc_state_out[worldid, efcid] = types.ConstraintState.QUADRATIC.value
      elif efcid < ne + nf:
        floss = efc_frictionloss_in[worldid, efcid]
        if force <= -floss:
          efc_state_out[worldid, efcid] = types.ConstraintState.LINEARPOS.value
        elif force >= floss:
          efc_state_out[worldid, efcid] = types.ConstraintState.LINEARNEG.value
        else:
          efc_state_out[worldid, efcid] = types.ConstraintState.QUADRATIC.value
      elif efc_type_in[worldid, efcid] != types.ConstraintType.CONTACT_ELLIPTIC:
        if force <= 0.0:
          efc_state_out[worldid, efcid] = types.ConstraintState.SATISFIED.value
        else:
          efc_state_out[worldid, efcid] = types.ConstraintState.QUADRATIC.value
      else:
        efc_state_out[worldid, efcid] = types.ConstraintState.SATISFIED.value

    # ---- Inline convergence check ----
    NV_ = wp.static(nv)
    tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]
    meaninertia = stat_meaninertia[worldid % stat_meaninertia.shape[0]]
    scale = 1.0 / (meaninertia * float(wp.max(1, NV_)))
    scaled_improvement = improvement * scale

    solver_niter_inout[worldid] += 1  # kernel_analyzer: ignore

    if scaled_improvement < tolerance or solver_niter_inout[worldid] >= maxiter:
      done_inout[worldid] = True  # kernel_analyzer: ignore
      wp.atomic_add(nsolving_inout, 0, -1)

  return kernel


def _pgs_process_group_func(nv: int, is_sparse: bool, use_sparse_bt: bool):
  """Factory returning a shared @wp.func for processing one constraint group.

  Handles both 1D constraints (pyramidal contacts, limits, friction, equality)
  and elliptic QCQP contacts (dim 3, 4, 6). Used by both the color-parallel
  sweep and the sequential overflow sweep.

  Returns improvement via atomic accumulation.
  """
  IS_SPARSE = is_sparse
  update_qacc = _pgs_update_qacc_delta_func(nv, use_sparse_bt)

  # kernel_analyzer: off
  @wp.func
  def process_group(
    worldid: int,
    efc_start: int,
    efc_count: int,
    ne: int,
    nf: int,
    tolerance: float,
    # Data arrays:
    efc_type_in: wp.array2d[int],
    efc_frictionloss_in: wp.array2d[float],
    efc_id_in: wp.array2d[int],
    contact_dim_in: wp.array[int],
    contact_friction_in: wp.array[types.vec5],
    contact_efc_address_in: wp.array2d[int],
    efc_J_in: wp.array3d[float],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    BT_in: wp.array3d[float],
    BT_rownnz_in: wp.array2d[int],
    BT_rowadr_in: wp.array2d[int],
    BT_colind_in: wp.array3d[int],
    AR_diag_inv_in: wp.array2d[float],
    AR_elliptic_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_aref_in: wp.array2d[float],
    # In/Out:
    efc_force_inout: wp.array2d[float],
    qacc_inout: wp.array2d[float],
    improvement_out: wp.array[float],
  ):
    NV = wp.static(nv)

    # Check if this group is an elliptic contact (multi-dim QCQP)
    is_elliptic = int(0)
    dim = int(1)
    if efc_type_in[worldid, efc_start] == types.ConstraintType.CONTACT_ELLIPTIC:
      con_id = efc_id_in[worldid, efc_start]
      dim = contact_dim_in[con_id]
      if dim > 1:
        is_elliptic = 1

    if is_elliptic == 0:
      # ---- 1D or multi-row sequential (pyramidal/connect/weld) ----
      for row_offset in range(efc_count):
        i = efc_start + row_offset

        # Compute residual
        res = float(0.0)
        if wp.static(IS_SPARSE):
          rownnz_i = efc_J_rownnz_in[worldid, i]
          rowadr_i = efc_J_rowadr_in[worldid, i]
          for k in range(rownnz_i):
            sparseid = rowadr_i + k
            col = efc_J_colind_in[worldid, 0, sparseid]
            res += efc_J_in[worldid, 0, sparseid] * qacc_inout[worldid, col]
        else:
          for k in range(NV):
            res += efc_J_in[worldid, i, k] * qacc_inout[worldid, k]
        res -= efc_aref_in[worldid, i]
        D_i = efc_D_in[worldid, i]
        if D_i > 0.0:
          res += efc_force_inout[worldid, i] / D_i

        old_force = efc_force_inout[worldid, i]
        AR_diag_inv = AR_diag_inv_in[worldid, i]
        new_force = old_force - res * AR_diag_inv

        # Project
        if i >= ne and i < ne + nf:
          floss = efc_frictionloss_in[worldid, i]
          if new_force < -floss:
            new_force = -floss
          elif new_force > floss:
            new_force = floss
        elif i >= ne + nf:
          if new_force < 0.0:
            new_force = 0.0

        # Cost change
        delta = new_force - old_force
        AR_ii = 1.0 / AR_diag_inv if AR_diag_inv > 0.0 else 0.0
        change = 0.5 * delta * delta * AR_ii + delta * res

        if change > tolerance:
          new_force = old_force
        else:
          wp.atomic_add(improvement_out, worldid, -change)
          actual_delta = new_force - old_force
          update_qacc(worldid, i, actual_delta, BT_in, BT_rownnz_in, BT_rowadr_in, BT_colind_in, qacc_inout)

        efc_force_inout[worldid, i] = new_force  # kernel_analyzer: ignore

    else:
      # ---- Elliptic cone constraint (multi-dim QCQP) ----
      # Contiguous rows: idx_d = efc_start + d
      con_id = efc_id_in[worldid, efc_start]
      mu = contact_friction_in[con_id]

      idx0 = efc_start
      idx1 = efc_start + 1
      idx2 = efc_start + 2
      idx3 = int(-1)
      idx4 = int(-1)
      idx5 = int(-1)
      if dim >= 4:
        idx3 = efc_start + 3
      if dim >= 6:
        idx4 = efc_start + 4
        idx5 = efc_start + 5

      # ---- Compute residuals for all dim rows ----
      res0 = float(0.0)
      res1 = float(0.0)
      res2 = float(0.0)
      res3 = float(0.0)
      res4 = float(0.0)
      res5 = float(0.0)

      # Row 0 (normal)
      if wp.static(IS_SPARSE):
        rnnz0 = efc_J_rownnz_in[worldid, idx0]
        radr0 = efc_J_rowadr_in[worldid, idx0]
        for k in range(rnnz0):
          sid = radr0 + k
          col = efc_J_colind_in[worldid, 0, sid]
          res0 += efc_J_in[worldid, 0, sid] * qacc_inout[worldid, col]
      else:
        for k in range(NV):
          res0 += efc_J_in[worldid, idx0, k] * qacc_inout[worldid, k]
      res0 -= efc_aref_in[worldid, idx0]
      D_0 = efc_D_in[worldid, idx0]
      if D_0 > 0.0:
        res0 += efc_force_inout[worldid, idx0] / D_0

      # Rows 1,2
      if wp.static(IS_SPARSE):
        rnnz1 = efc_J_rownnz_in[worldid, idx1]
        radr1 = efc_J_rowadr_in[worldid, idx1]
        for k in range(rnnz1):
          sid = radr1 + k
          col = efc_J_colind_in[worldid, 0, sid]
          res1 += efc_J_in[worldid, 0, sid] * qacc_inout[worldid, col]
      else:
        for k in range(NV):
          res1 += efc_J_in[worldid, idx1, k] * qacc_inout[worldid, k]
      res1 -= efc_aref_in[worldid, idx1]
      D_1 = efc_D_in[worldid, idx1]
      if D_1 > 0.0:
        res1 += efc_force_inout[worldid, idx1] / D_1

      if wp.static(IS_SPARSE):
        rnnz2 = efc_J_rownnz_in[worldid, idx2]
        radr2 = efc_J_rowadr_in[worldid, idx2]
        for k in range(rnnz2):
          sid = radr2 + k
          col = efc_J_colind_in[worldid, 0, sid]
          res2 += efc_J_in[worldid, 0, sid] * qacc_inout[worldid, col]
      else:
        for k in range(NV):
          res2 += efc_J_in[worldid, idx2, k] * qacc_inout[worldid, k]
      res2 -= efc_aref_in[worldid, idx2]
      D_2 = efc_D_in[worldid, idx2]
      if D_2 > 0.0:
        res2 += efc_force_inout[worldid, idx2] / D_2

      if dim >= 4:
        if wp.static(IS_SPARSE):
          rnnz3 = efc_J_rownnz_in[worldid, idx3]
          radr3 = efc_J_rowadr_in[worldid, idx3]
          for k in range(rnnz3):
            sid = radr3 + k
            col = efc_J_colind_in[worldid, 0, sid]
            res3 += efc_J_in[worldid, 0, sid] * qacc_inout[worldid, col]
        else:
          for k in range(NV):
            res3 += efc_J_in[worldid, idx3, k] * qacc_inout[worldid, k]
        res3 -= efc_aref_in[worldid, idx3]
        D_3 = efc_D_in[worldid, idx3]
        if D_3 > 0.0:
          res3 += efc_force_inout[worldid, idx3] / D_3

      if dim >= 6:
        if wp.static(IS_SPARSE):
          rnnz4 = efc_J_rownnz_in[worldid, idx4]
          radr4 = efc_J_rowadr_in[worldid, idx4]
          for k in range(rnnz4):
            sid = radr4 + k
            col = efc_J_colind_in[worldid, 0, sid]
            res4 += efc_J_in[worldid, 0, sid] * qacc_inout[worldid, col]
        else:
          for k in range(NV):
            res4 += efc_J_in[worldid, idx4, k] * qacc_inout[worldid, k]
        res4 -= efc_aref_in[worldid, idx4]
        D_4 = efc_D_in[worldid, idx4]
        if D_4 > 0.0:
          res4 += efc_force_inout[worldid, idx4] / D_4

        if wp.static(IS_SPARSE):
          rnnz5 = efc_J_rownnz_in[worldid, idx5]
          radr5 = efc_J_rowadr_in[worldid, idx5]
          for k in range(rnnz5):
            sid = radr5 + k
            col = efc_J_colind_in[worldid, 0, sid]
            res5 += efc_J_in[worldid, 0, sid] * qacc_inout[worldid, col]
        else:
          for k in range(NV):
            res5 += efc_J_in[worldid, idx5, k] * qacc_inout[worldid, k]
        res5 -= efc_aref_in[worldid, idx5]
        D_5 = efc_D_in[worldid, idx5]
        if D_5 > 0.0:
          res5 += efc_force_inout[worldid, idx5] / D_5

      # ---- Save old forces ----
      old0 = efc_force_inout[worldid, idx0]
      old1 = efc_force_inout[worldid, idx1]
      old2 = efc_force_inout[worldid, idx2]
      old3 = float(0.0)
      old4 = float(0.0)
      old5 = float(0.0)
      if dim >= 4:
        old3 = efc_force_inout[worldid, idx3]
      if dim >= 6:
        old4 = efc_force_inout[worldid, idx4]
        old5 = efc_force_inout[worldid, idx5]

      # ---- Solve elliptic QCQP (shared) ----
      new0, new1, new2, new3, new4, new5, change = _pgs_solve_elliptic_cone(
        dim,
        old0,
        old1,
        old2,
        old3,
        old4,
        old5,
        res0,
        res1,
        res2,
        res3,
        res4,
        res5,
        AR_elliptic_in,
        worldid,
        idx0,
        mu,
        1.0,
      )

      # ---- Accept or reject ----
      if change <= tolerance:
        wp.atomic_add(improvement_out, worldid, -change)
        d0 = new0 - old0
        d1 = new1 - old1
        d2 = new2 - old2
        d3 = new3 - old3
        d4 = new4 - old4
        d5 = new5 - old5
        efc_force_inout[worldid, idx0] = new0  # kernel_analyzer: ignore
        efc_force_inout[worldid, idx1] = new1  # kernel_analyzer: ignore
        efc_force_inout[worldid, idx2] = new2  # kernel_analyzer: ignore
        if dim >= 4:
          efc_force_inout[worldid, idx3] = new3  # kernel_analyzer: ignore
        if dim >= 6:
          efc_force_inout[worldid, idx4] = new4  # kernel_analyzer: ignore
          efc_force_inout[worldid, idx5] = new5  # kernel_analyzer: ignore
        # Update qacc
        if d0 != 0.0:
          update_qacc(worldid, idx0, d0, BT_in, BT_rownnz_in, BT_rowadr_in, BT_colind_in, qacc_inout)
        if d1 != 0.0:
          update_qacc(worldid, idx1, d1, BT_in, BT_rownnz_in, BT_rowadr_in, BT_colind_in, qacc_inout)
        if d2 != 0.0:
          update_qacc(worldid, idx2, d2, BT_in, BT_rownnz_in, BT_rowadr_in, BT_colind_in, qacc_inout)
        if dim >= 4:
          if d3 != 0.0:
            update_qacc(worldid, idx3, d3, BT_in, BT_rownnz_in, BT_rowadr_in, BT_colind_in, qacc_inout)
        if dim >= 6:
          if d4 != 0.0:
            update_qacc(worldid, idx4, d4, BT_in, BT_rownnz_in, BT_rowadr_in, BT_colind_in, qacc_inout)
          if d5 != 0.0:
            update_qacc(worldid, idx5, d5, BT_in, BT_rownnz_in, BT_rowadr_in, BT_colind_in, qacc_inout)

  return process_group
  # kernel_analyzer: on


def _pgs_color_sweep_kernel(nv: int, is_sparse: bool, use_sparse_bt: bool):
  """Color-parallel PGS sweep: each thread processes one constraint group.

  Handles 1D constraints (pyramidal contacts, limits, friction, equality),
  and elliptic QCQP contacts (dim 3, 4, 6). qacc updates need no atomics
  since constraints within a color don't share DOFs.

  Contact efc rows are contiguous (efc_start + dimid).
  """
  process_group = _pgs_process_group_func(nv, is_sparse, use_sparse_bt)

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(  # kernel_analyzer: ignore
    # In:
    schedule_groups_in: wp.array2d[int],
    schedule_offsets_in: wp.array2d[int],
    color_idx: int,
    group_efc_start_in: wp.array2d[int],
    group_efc_count_in: wp.array2d[int],
    # Data in:
    ne_in: wp.array[int],
    nf_in: wp.array[int],
    nefc_in: wp.array[int],
    efc_type_in: wp.array2d[int],
    efc_frictionloss_in: wp.array2d[float],
    efc_id_in: wp.array2d[int],
    contact_dim_in: wp.array[int],
    contact_friction_in: wp.array[types.vec5],
    contact_efc_address_in: wp.array2d[int],
    efc_J_in: wp.array3d[float],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    BT_in: wp.array3d[float],
    # Sparse BT index (ntree > 1):
    BT_rownnz_in: wp.array2d[int],
    BT_rowadr_in: wp.array2d[int],
    BT_colind_in: wp.array3d[int],
    AR_diag_inv_in: wp.array2d[float],
    AR_elliptic_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_aref_in: wp.array2d[float],
    # Model:
    opt_tolerance: wp.array[float],
    done_in: wp.array[bool],
    # Out:
    efc_force_inout: wp.array2d[float],
    qacc_inout: wp.array2d[float],
    improvement_out: wp.array[float],  # kernel_analyzer: ignore
  ):
    worldid, batch_idx = wp.tid()

    if done_in[worldid]:
      return

    sched_start = schedule_offsets_in[worldid, color_idx]
    sched_end = schedule_offsets_in[worldid, color_idx + 1]
    if batch_idx >= (sched_end - sched_start):
      return

    group_id = schedule_groups_in[worldid, sched_start + batch_idx]
    if group_id < 0:
      return

    efc_start = group_efc_start_in[worldid, group_id]
    efc_count = group_efc_count_in[worldid, group_id]
    ne = ne_in[worldid]
    nf = nf_in[worldid]
    tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]

    process_group(
      worldid,
      efc_start,
      efc_count,
      ne,
      nf,
      tolerance,
      efc_type_in,
      efc_frictionloss_in,
      efc_id_in,
      contact_dim_in,
      contact_friction_in,
      contact_efc_address_in,
      efc_J_in,
      efc_J_rownnz_in,
      efc_J_rowadr_in,
      efc_J_colind_in,
      BT_in,
      BT_rownnz_in,
      BT_rowadr_in,
      BT_colind_in,
      AR_diag_inv_in,
      AR_elliptic_in,
      efc_D_in,
      efc_aref_in,
      efc_force_inout,
      qacc_inout,
      improvement_out,
    )

  return kernel


def _pgs_overflow_sweep_kernel(nv: int, is_sparse: bool, use_sparse_bt: bool):
  """Sequential sweep over overflow (uncolored) constraint groups.

  Processes groups that could not be assigned a color during graph coloring
  (typically when ngroups > max_colors on single-tree models). Runs with
  dim=(nworld,) — one thread per world, iterating over the overflow list
  sequentially to avoid data races.
  """
  process_group = _pgs_process_group_func(nv, is_sparse, use_sparse_bt)

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(  # kernel_analyzer: ignore
    # In:
    overflow_list_in: wp.array2d[int],
    n_overflow_in: wp.array[int],
    # Group info:
    group_efc_start_in: wp.array2d[int],
    group_efc_count_in: wp.array2d[int],
    # Data in:
    ne_in: wp.array[int],
    nf_in: wp.array[int],
    nefc_in: wp.array[int],
    efc_type_in: wp.array2d[int],
    efc_frictionloss_in: wp.array2d[float],
    efc_id_in: wp.array2d[int],
    contact_dim_in: wp.array[int],
    contact_friction_in: wp.array[types.vec5],
    contact_efc_address_in: wp.array2d[int],
    efc_J_in: wp.array3d[float],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    BT_in: wp.array3d[float],
    # Sparse BT index (ntree > 1):
    BT_rownnz_in: wp.array2d[int],
    BT_rowadr_in: wp.array2d[int],
    BT_colind_in: wp.array3d[int],
    AR_diag_inv_in: wp.array2d[float],
    AR_elliptic_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_aref_in: wp.array2d[float],
    # Model:
    opt_tolerance: wp.array[float],
    done_in: wp.array[bool],
    # Out:
    efc_force_inout: wp.array2d[float],
    qacc_inout: wp.array2d[float],
    improvement_out: wp.array[float],  # kernel_analyzer: ignore
  ):
    worldid = wp.tid()

    if done_in[worldid]:
      return

    n_ov = n_overflow_in[worldid]
    ne = ne_in[worldid]
    nf = nf_in[worldid]
    tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]

    for ov_idx in range(n_ov):
      group_id = overflow_list_in[worldid, ov_idx]
      efc_start = group_efc_start_in[worldid, group_id]
      efc_count = group_efc_count_in[worldid, group_id]

      process_group(
        worldid,
        efc_start,
        efc_count,
        ne,
        nf,
        tolerance,
        efc_type_in,
        efc_frictionloss_in,
        efc_id_in,
        contact_dim_in,
        contact_friction_in,
        contact_efc_address_in,
        efc_J_in,
        efc_J_rownnz_in,
        efc_J_rowadr_in,
        efc_J_colind_in,
        BT_in,
        BT_rownnz_in,
        BT_rowadr_in,
        BT_colind_in,
        AR_diag_inv_in,
        AR_elliptic_in,
        efc_D_in,
        efc_aref_in,
        efc_force_inout,
        qacc_inout,
        improvement_out,
      )

  return kernel


@wp.kernel
def _pgs_dual_state(
  # Data in:
  ne_in: wp.array[int],
  nf_in: wp.array[int],
  nefc_in: wp.array[int],
  efc_type_in: wp.array2d[int],
  efc_frictionloss_in: wp.array2d[float],
  efc_force_in: wp.array2d[float],
  # Data out:
  efc_state_out: wp.array2d[int],
):
  """Compute dual state for all constraint rows."""
  worldid, efcid = wp.tid()
  nefc = nefc_in[worldid]
  if efcid >= nefc:
    return

  ne = ne_in[worldid]
  nf = nf_in[worldid]
  force = efc_force_in[worldid, efcid]

  if efcid < ne:
    efc_state_out[worldid, efcid] = types.ConstraintState.QUADRATIC.value
  elif efcid < ne + nf:
    floss = efc_frictionloss_in[worldid, efcid]
    if force <= -floss:
      efc_state_out[worldid, efcid] = types.ConstraintState.LINEARPOS.value
    elif force >= floss:
      efc_state_out[worldid, efcid] = types.ConstraintState.LINEARNEG.value
    else:
      efc_state_out[worldid, efcid] = types.ConstraintState.QUADRATIC.value
  elif efc_type_in[worldid, efcid] != types.ConstraintType.CONTACT_ELLIPTIC:
    if force <= 0.0:
      efc_state_out[worldid, efcid] = types.ConstraintState.SATISFIED.value
    else:
      efc_state_out[worldid, efcid] = types.ConstraintState.QUADRATIC.value
  else:
    efc_state_out[worldid, efcid] = types.ConstraintState.SATISFIED.value


@wp.kernel
def _pgs_convergence_check(
  # Model:
  opt_tolerance: wp.array[float],
  stat_meaninertia: wp.array[float],
  # In:
  improvement_in: wp.array[float],
  nv_val: int,
  maxiter: int,
  # Out:
  done_inout: wp.array[bool],
  solver_niter_inout: wp.array[int],
  nsolving_inout: wp.array[int],
):
  """Check convergence after one colored sweep."""
  worldid = wp.tid()
  if done_inout[worldid]:
    return

  tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]
  meaninertia = stat_meaninertia[worldid % stat_meaninertia.shape[0]]
  scale = 1.0 / (meaninertia * float(wp.max(1, nv_val)))
  scaled_improvement = improvement_in[worldid] * scale

  solver_niter_inout[worldid] += 1  # kernel_analyzer: ignore

  if scaled_improvement < tolerance or solver_niter_inout[worldid] >= maxiter:
    done_inout[worldid] = True  # kernel_analyzer: ignore
    wp.atomic_add(nsolving_inout, 0, -1)


@wp.kernel
def _pgs_warmstart_cost_dense(
  # Data in:
  nefc_in: wp.array[int],
  qacc_in: wp.array2d[float],
  qacc_smooth_in: wp.array2d[float],
  efc_J_in: wp.array3d[float],
  efc_D_in: wp.array2d[float],
  efc_aref_in: wp.array2d[float],
  efc_force_in: wp.array2d[float],
  # Out:
  warmstart_cost_out: wp.array[float],
):
  """Compute warmstart cost = f·b + 0.5·f·AR·f (AR-free, dense J).

  cost = 0.5 · Σ_i f[i] · (J[i,:]·(qacc + qacc_smooth) - 2·aref[i] + f[i]/D[i])
  """
  worldid = wp.tid()
  nefc = nefc_in[worldid]
  nv_dim = qacc_in.shape[1]

  cost = float(0.0)
  for i in range(nefc):
    fi = efc_force_in[worldid, i]
    if fi == 0.0:
      continue

    # J[i,:] · (qacc + qacc_smooth)
    j_sum_qacc = float(0.0)
    for k in range(nv_dim):
      j_sum_qacc += efc_J_in[worldid, i, k] * (qacc_in[worldid, k] + qacc_smooth_in[worldid, k])

    # cost_i = f[i] * (j_sum_qacc - 2*aref[i] + f[i]/D[i])
    D_i = efc_D_in[worldid, i]
    R_fi = fi / D_i if D_i > 0.0 else 0.0
    cost += fi * (j_sum_qacc - 2.0 * efc_aref_in[worldid, i] + R_fi)

  warmstart_cost_out[worldid] = 0.5 * cost


@wp.kernel
def _pgs_warmstart_cost_sparse(
  # Data in:
  nefc_in: wp.array[int],
  qacc_in: wp.array2d[float],
  qacc_smooth_in: wp.array2d[float],
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  efc_J_in: wp.array3d[float],
  efc_D_in: wp.array2d[float],
  efc_aref_in: wp.array2d[float],
  efc_force_in: wp.array2d[float],
  # Out:
  warmstart_cost_out: wp.array[float],
):
  """Compute warmstart cost = f·b + 0.5·f·AR·f (AR-free, sparse J)."""
  worldid = wp.tid()
  nefc = nefc_in[worldid]

  cost = float(0.0)
  for i in range(nefc):
    fi = efc_force_in[worldid, i]
    if fi == 0.0:
      continue

    # J[i,:] · (qacc + qacc_smooth) — sparse dot
    j_sum_qacc = float(0.0)
    rownnz = efc_J_rownnz_in[worldid, i]
    rowadr = efc_J_rowadr_in[worldid, i]
    for k in range(rownnz):
      sparseid = rowadr + k
      col = efc_J_colind_in[worldid, 0, sparseid]
      j_sum_qacc += efc_J_in[worldid, 0, sparseid] * (qacc_in[worldid, col] + qacc_smooth_in[worldid, col])

    D_i = efc_D_in[worldid, i]
    R_fi = fi / D_i if D_i > 0.0 else 0.0
    cost += fi * (j_sum_qacc - 2.0 * efc_aref_in[worldid, i] + R_fi)

  warmstart_cost_out[worldid] = 0.5 * cost


@wp.kernel
def _pgs_warmstart_forces(
  # Data in:
  ne_in: wp.array[int],
  nf_in: wp.array[int],
  nefc_in: wp.array[int],
  contact_friction_in: wp.array[types.vec5],
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array2d[int],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  efc_D_in: wp.array2d[float],
  efc_frictionloss_in: wp.array2d[float],
  # In:
  impratio_invsqrt_in: wp.array[float],
  jar_in: wp.array2d[float],  # J @ qacc_warmstart - aref
  # Data out:
  efc_force_out: wp.array2d[float],
):
  """Compute warmstart forces: force = -D * jar with projection.

  Follows MuJoCo C's mj_constraintUpdate logic:
  - Equality: force = -D * jar (unconstrained)
  - Friction: clamp to [-floss, floss]
  - Limit/contact (non-negative): clamp to [0, inf)
  - Elliptic: three-zone dual cone projection
  """
  worldid, efcid = wp.tid()
  nefc = nefc_in[worldid]

  if efcid >= nefc:
    return

  ne = ne_in[worldid]
  nf = nf_in[worldid]

  # Skip friction rows of elliptic contacts (handled by normal row)
  if efcid >= ne + nf and efc_type_in[worldid, efcid] == types.ConstraintType.CONTACT_ELLIPTIC:
    con_id = efc_id_in[worldid, efcid]
    dim = contact_dim_in[con_id]
    friction = contact_friction_in[con_id]

    # Use indirect indexing: only process from the normal row (dimid=0)
    idx0 = contact_efc_address_in[con_id, 0]
    if efcid != idx0:
      return

    # Look up indirect indices for all dimensions
    idx1 = int(-1)
    idx2 = int(-1)
    idx3 = int(-1)
    idx4 = int(-1)
    idx5 = int(-1)
    if dim >= 3:
      idx1 = contact_efc_address_in[con_id, 1]
      idx2 = contact_efc_address_in[con_id, 2]
    if dim >= 4:
      idx3 = contact_efc_address_in[con_id, 3]
    if dim >= 6:
      idx4 = contact_efc_address_in[con_id, 4]
      idx5 = contact_efc_address_in[con_id, 5]

    # Compute mu = friction[0] / sqrt(impratio)
    mu_cone = friction[0] * impratio_invsqrt_in[0]

    # Map jar to regular dual cone space using indirect indices
    U0 = jar_in[worldid, idx0] * mu_cone
    N = U0
    T_sq = float(0.0)
    if dim >= 3:
      U1 = jar_in[worldid, idx1] * friction[0]
      U2 = jar_in[worldid, idx2] * friction[1]
      T_sq += U1 * U1 + U2 * U2
    if dim >= 4:
      U3 = jar_in[worldid, idx3] * friction[2]
      T_sq += U3 * U3
    if dim >= 6:
      U4 = jar_in[worldid, idx4] * friction[3]
      U5 = jar_in[worldid, idx5] * friction[4]
      T_sq += U4 * U4 + U5 * U5
    T = wp.sqrt(T_sq)

    # Three zones
    # Top zone: N >= mu*T || (T <= 0 && N >= 0) -> force = 0
    if N >= mu_cone * T or (T <= 0.0 and N >= 0.0):
      efc_force_out[worldid, idx0] = 0.0
      if dim >= 3:
        efc_force_out[worldid, idx1] = 0.0
        efc_force_out[worldid, idx2] = 0.0
      if dim >= 4:
        efc_force_out[worldid, idx3] = 0.0
      if dim >= 6:
        efc_force_out[worldid, idx4] = 0.0
        efc_force_out[worldid, idx5] = 0.0

    # Bottom zone: mu*N + T <= 0 || (T <= 0 && N < 0) -> force = -D*jar (quadratic)
    elif mu_cone * N + T <= 0.0 or (T <= 0.0 and N < 0.0):
      efc_force_out[worldid, idx0] = -efc_D_in[worldid, idx0] * jar_in[worldid, idx0]
      if dim >= 3:
        efc_force_out[worldid, idx1] = -efc_D_in[worldid, idx1] * jar_in[worldid, idx1]
        efc_force_out[worldid, idx2] = -efc_D_in[worldid, idx2] * jar_in[worldid, idx2]
      if dim >= 4:
        efc_force_out[worldid, idx3] = -efc_D_in[worldid, idx3] * jar_in[worldid, idx3]
      if dim >= 6:
        efc_force_out[worldid, idx4] = -efc_D_in[worldid, idx4] * jar_in[worldid, idx4]
        efc_force_out[worldid, idx5] = -efc_D_in[worldid, idx5] * jar_in[worldid, idx5]

    # Middle zone: cone projection
    else:
      Dm = efc_D_in[worldid, idx0] / (mu_cone * mu_cone * (1.0 + mu_cone * mu_cone))
      NmT = N - mu_cone * T
      f0 = -Dm * NmT * mu_cone
      efc_force_out[worldid, idx0] = f0

      if T > MJ_MINVAL:
        if dim >= 3:
          U1_ = jar_in[worldid, idx1] * friction[0]
          efc_force_out[worldid, idx1] = -f0 / T * U1_ * friction[0]
          U2_ = jar_in[worldid, idx2] * friction[1]
          efc_force_out[worldid, idx2] = -f0 / T * U2_ * friction[1]
        if dim >= 4:
          U3_ = jar_in[worldid, idx3] * friction[2]
          efc_force_out[worldid, idx3] = -f0 / T * U3_ * friction[2]
        if dim >= 6:
          U4_ = jar_in[worldid, idx4] * friction[3]
          efc_force_out[worldid, idx4] = -f0 / T * U4_ * friction[3]
          U5_ = jar_in[worldid, idx5] * friction[4]
          efc_force_out[worldid, idx5] = -f0 / T * U5_ * friction[4]
      else:
        if dim >= 3:
          efc_force_out[worldid, idx1] = 0.0
          efc_force_out[worldid, idx2] = 0.0
        if dim >= 4:
          efc_force_out[worldid, idx3] = 0.0
        if dim >= 6:
          efc_force_out[worldid, idx4] = 0.0
          efc_force_out[worldid, idx5] = 0.0

    return

  jar = jar_in[worldid, efcid]
  D = efc_D_in[worldid, efcid]

  force = -D * jar

  # Friction: clamp to [-floss, floss]
  if efcid >= ne and efcid < ne + nf:
    floss = efc_frictionloss_in[worldid, efcid]
    if force < -floss:
      force = -floss
    elif force > floss:
      force = floss

  # Limit/contact (non-elliptic): force >= 0
  elif efcid >= ne + nf:
    if jar >= 0.0:
      force = 0.0

  efc_force_out[worldid, efcid] = force


@wp.kernel
def _pgs_warmstart_select(
  # Data in:
  nefc_in: wp.array[int],
  qacc_smooth_in: wp.array2d[float],
  # In:
  warmstart_cost_in: wp.array[float],
  nv_val: int,
  # Data in/out:
  efc_force_inout: wp.array2d[float],
  qfrc_constraint_inout: wp.array2d[float],
  qacc_inout: wp.array2d[float],
):
  """If warmstart cost > 0 or NaN, use zero forces and reset qacc."""
  worldid = wp.tid()

  cost = warmstart_cost_in[worldid]
  # NaN check: cost != cost is true for NaN
  if cost > 0.0 or cost != cost:
    nefc = nefc_in[worldid]
    for i in range(nefc):
      efc_force_inout[worldid, i] = 0.0  # kernel_analyzer: ignore
    for i in range(nv_val):
      qfrc_constraint_inout[worldid, i] = 0.0  # kernel_analyzer: ignore
      qacc_inout[worldid, i] = qacc_smooth_in[worldid, i]  # kernel_analyzer: ignore


@wp.kernel
def _pgs_update_qacc(
  # Data in:
  nefc_in: wp.array[int],
  efc_force_in: wp.array2d[float],
  # In:
  BT_in: wp.array3d[float],
  # Data in/out:
  qacc_inout: wp.array2d[float],
):
  """Update qacc += BT^T · f, i.e., qacc[k] += Σ_j f[j] · BT[j, k]."""
  worldid, dofid = wp.tid()
  nefc = nefc_in[worldid]

  val = float(0.0)
  for j in range(nefc):
    val += efc_force_in[worldid, j] * BT_in[worldid, j, dofid]
  qacc_inout[worldid, dofid] += val  # kernel_analyzer: ignore


@wp.kernel
def _pgs_copy_j_rows_sparse(
  # Data in:
  nefc_in: wp.array[int],
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  efc_J_in: wp.array3d[float],
  # Out:
  BT_out: wp.array3d[float],
):
  """Copy all sparse J rows into BT buffer."""
  worldid, efcid = wp.tid()
  nefc = nefc_in[worldid]

  if efcid >= nefc:
    return

  rownnz = efc_J_rownnz_in[worldid, efcid]
  rowadr = efc_J_rowadr_in[worldid, efcid]
  for k in range(rownnz):
    colind = efc_J_colind_in[worldid, 0, rowadr + k]
    BT_out[worldid, efcid, colind] = efc_J_in[worldid, 0, rowadr + k]


@wp.kernel
def _pgs_copy_j_rows_dense(
  # Data in:
  nefc_in: wp.array[int],
  efc_J_in: wp.array3d[float],
  # Out:
  BT_out: wp.array3d[float],
):
  """Copy all dense J rows into BT buffer."""
  worldid, efcid, dofid = wp.tid()
  nefc = nefc_in[worldid]

  if efcid >= nefc:
    return

  BT_out[worldid, efcid, dofid] = efc_J_in[worldid, efcid, dofid]


def _pgs_solve_m_batched_sparse(nv: int, nlevels: int):
  """Batched sparse backsubstitution: BT[w,e,:] = M⁻¹ @ BT[w,e,:].

  Each (worldid, efcid) thread does a sequential LDL backsubstitution
  using the pre-computed factorization. All efcid are independent and
  run in parallel.
  """

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    # In:
    L: wp.array3d[float],
    D: wp.array2d[float],
    all_updates: wp.array[wp.vec3i],
    level_offsets: wp.array[int],
    # In/Out:
    BT_inout: wp.array3d[float],
  ):
    worldid, efcid = wp.tid()
    NV = wp.static(nv)
    NLEVELS = wp.static(nlevels)

    nefc = nefc_in[worldid]
    if efcid >= nefc:
      return

    # Forward substitution: x <- L^{-T} x
    for level in range(NLEVELS):
      level_idx = NLEVELS - 1 - level
      level_offset = level_offsets[level_idx]
      level_size = level_offsets[level_idx + 1] - level_offset

      for u in range(level_size):
        update = all_updates[level_offset + u]
        i, k, Madr_ki = update[0], update[1], update[2]
        BT_inout[worldid, efcid, i] -= L[worldid, 0, Madr_ki] * BT_inout[worldid, efcid, k]  # kernel_analyzer: ignore

    # Diagonal: x <- D^{-1} x
    for dofid in range(NV):
      BT_inout[worldid, efcid, dofid] *= D[worldid, dofid]  # kernel_analyzer: ignore

    # Backward substitution: x <- L^{-1} x
    for level in range(NLEVELS):
      level_idx = level
      level_offset = level_offsets[level_idx]
      level_size = level_offsets[level_idx + 1] - level_offset

      for u in range(level_size):
        update = all_updates[level_offset + u]
        i, k, Madr_ki = update[0], update[1], update[2]
        BT_inout[worldid, efcid, k] -= L[worldid, 0, Madr_ki] * BT_inout[worldid, efcid, i]  # kernel_analyzer: ignore

  return kernel


def _pgs_solve_m_batched_dense(tile):
  """Batched dense Cholesky backsubstitution for one tile."""

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    # In:
    L: wp.array3d[float],
    adr: wp.array[int],
    # In/Out:
    BT_inout: wp.array3d[float],
  ):
    worldid, efcid, nodeid = wp.tid()
    TILE_SIZE = wp.static(tile.size)

    nefc = nefc_in[worldid]
    if efcid >= nefc:
      return

    dofid = adr[nodeid]

    # Load L tile and y slice
    L_tile = wp.tile_load(L[worldid], shape=(TILE_SIZE, TILE_SIZE), offset=(dofid, dofid))
    y_slice = wp.tile_load(BT_inout[worldid, efcid], shape=(TILE_SIZE,), offset=(dofid,))
    x_slice = wp.tile_cholesky_solve(L_tile, y_slice)
    wp.tile_store(BT_inout[worldid, efcid], x_slice, offset=(dofid,))

  return kernel


@wp.kernel
def _pgs_bt_rownnz(
  # Model:
  dof_treeid: wp.array[int],
  tree_dofnum: wp.array[int],
  # Data in:
  nefc_in: wp.array[int],
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  # Out:
  BT_rownnz_out: wp.array2d[int],
):
  """Compute BT row nnz from J's sparsity and tree structure.

  BT has non-zeros at all DOFs of every tree that J touches.
  Unique trees are detected by scanning previously visited J columns,
  which works for any number of trees without overflow.
  """
  worldid, efcid = wp.tid()
  nefc = nefc_in[worldid]
  if efcid >= nefc:
    return

  nnz_j = efc_J_rownnz_in[worldid, efcid]
  rowadr = efc_J_rowadr_in[worldid, efcid]
  nnz = int(0)
  for k in range(nnz_j):
    col = efc_J_colind_in[worldid, 0, rowadr + k]
    tid = dof_treeid[col]
    # Check if this tree was already seen by scanning previous columns
    n_prev_matches = int(0)
    for prev in range(k):
      prev_col = efc_J_colind_in[worldid, 0, rowadr + prev]
      if dof_treeid[prev_col] == tid:
        n_prev_matches += 1
    if n_prev_matches == 0:
      nnz += tree_dofnum[tid]
  BT_rownnz_out[worldid, efcid] = nnz


@wp.kernel
def _pgs_bt_rowadr(
  # Data in:
  nefc_in: wp.array[int],
  # In:
  BT_rownnz_in: wp.array2d[int],
  # Out:
  BT_rowadr_out: wp.array2d[int],
):
  """Compute BT row start addresses via sequential prefix sum per world."""
  worldid = wp.tid()
  nefc = nefc_in[worldid]
  adr = int(0)
  for i in range(nefc):
    BT_rowadr_out[worldid, i] = adr
    adr += BT_rownnz_in[worldid, i]


@wp.kernel
def _pgs_bt_colind(
  # Model:
  dof_treeid: wp.array[int],
  tree_dofadr: wp.array[int],
  tree_dofnum: wp.array[int],
  # Data in:
  nefc_in: wp.array[int],
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  # In:
  BT_rowadr_in: wp.array2d[int],
  # Out:
  BT_colind_out: wp.array3d[int],
):
  """Fill BT column indices: all DOFs of every tree that J touches."""
  worldid, efcid = wp.tid()
  nefc = nefc_in[worldid]
  if efcid >= nefc:
    return

  nnz_j = efc_J_rownnz_in[worldid, efcid]
  rowadr_j = efc_J_rowadr_in[worldid, efcid]
  rowadr_bt = BT_rowadr_in[worldid, efcid]
  offset = int(0)
  for k in range(nnz_j):
    col = efc_J_colind_in[worldid, 0, rowadr_j + k]
    tid = dof_treeid[col]
    # Check if this tree was already seen by scanning previous columns
    n_prev_matches = int(0)
    for prev in range(k):
      prev_col = efc_J_colind_in[worldid, 0, rowadr_j + prev]
      if dof_treeid[prev_col] == tid:
        n_prev_matches += 1
    if n_prev_matches == 0:
      dofadr = tree_dofadr[tid]
      dofnum = tree_dofnum[tid]
      for d in range(dofnum):
        BT_colind_out[worldid, 0, rowadr_bt + offset] = dofadr + d
        offset += 1


@event_scope
def _pgs_compute_bt_sparsity(m: types.Model, d: types.Data, ctx: PGSContext):
  """Compute sparse BT index from J's sparsity and tree structure."""
  # Pass 1: compute nnz per BT row
  wp.launch(
    _pgs_bt_rownnz,
    dim=(d.nworld, d.njmax),
    inputs=[
      m.dof_treeid,
      m.tree_dofnum,
      d.nefc,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
    ],
    outputs=[ctx.BT_rownnz],
  )

  # Pass 2: prefix sum for row addresses
  wp.launch(
    _pgs_bt_rowadr,
    dim=(d.nworld,),
    inputs=[d.nefc, ctx.BT_rownnz],
    outputs=[ctx.BT_rowadr],
  )

  # Pass 3: fill column indices
  wp.launch(
    _pgs_bt_colind,
    dim=(d.nworld, d.njmax),
    inputs=[
      m.dof_treeid,
      m.tree_dofadr,
      m.tree_dofnum,
      d.nefc,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      ctx.BT_rowadr,
    ],
    outputs=[ctx.BT_colind],
  )


def _pgs_jacobi_update_forces_kernel(nv: int, is_sparse: bool):
  """Jacobi force update: compute residuals and update all forces in parallel.

  For each constraint i (in parallel):
    1. Compute residual: res = J[i,:] · qacc - aref[i] + f[i]/D[i]
    2. Compute new force: f_new = f_old - res / AR[i,i]
    3. Project onto constraint bounds
    4. SOR blend: f_final = (1-ω) * f_old + ω * f_new
    5. Accept/reject based on cost change vs tolerance
    6. Atomic-add improvement for convergence check

  For elliptic contacts (dim > 1): only the normal row (idx0) processes
  the full QCQP sub-problem; sub-rows skip (handled by the normal row).
  """
  IS_SPARSE = is_sparse

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    opt_tolerance: wp.array[float],
    # Data in:
    ne_in: wp.array[int],
    nf_in: wp.array[int],
    nefc_in: wp.array[int],
    qacc_in: wp.array2d[float],
    contact_friction_in: wp.array[types.vec5],
    contact_dim_in: wp.array[int],
    contact_efc_address_in: wp.array2d[int],
    efc_type_in: wp.array2d[int],
    efc_id_in: wp.array2d[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_aref_in: wp.array2d[float],
    efc_frictionloss_in: wp.array2d[float],
    # In:
    AR_diag_inv_in: wp.array2d[float],
    AR_elliptic_in: wp.array3d[float],
    sor_omega: float,
    done_in: wp.array[bool],
    # Data in/out:
    efc_force_inout: wp.array2d[float],
    # Out:
    improvement_out: wp.array[float],
  ):
    worldid, efcid = wp.tid()

    if done_in[worldid]:
      return

    nefc = nefc_in[worldid]
    if efcid >= nefc:
      return

    ne = ne_in[worldid]
    nf = nf_in[worldid]
    NV = wp.static(nv)
    tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]

    # Check if this is an elliptic contact sub-row (not the normal row)
    efc_type = efc_type_in[worldid, efcid]
    if efc_type == types.ConstraintType.CONTACT_ELLIPTIC:
      con_id = efc_id_in[worldid, efcid]
      dim = contact_dim_in[con_id]
      idx0 = contact_efc_address_in[con_id, 0]
      if efcid != idx0:
        # Sub-row: handled by normal row processing
        return

      if dim > 1:
        # ---- Elliptic contact: QCQP sub-problem ----
        mu = contact_friction_in[con_id]

        idx1 = contact_efc_address_in[con_id, 1]
        idx2 = contact_efc_address_in[con_id, 2]
        idx3 = int(-1)
        idx4 = int(-1)
        idx5 = int(-1)
        if dim >= 4:
          idx3 = contact_efc_address_in[con_id, 3]
        if dim >= 6:
          idx4 = contact_efc_address_in[con_id, 4]
          idx5 = contact_efc_address_in[con_id, 5]

        # Compute residuals for all dim rows
        res0 = float(0.0)
        res1 = float(0.0)
        res2 = float(0.0)
        res3 = float(0.0)
        res4 = float(0.0)
        res5 = float(0.0)

        if wp.static(IS_SPARSE):
          rnnz0 = efc_J_rownnz_in[worldid, idx0]
          radr0 = efc_J_rowadr_in[worldid, idx0]
          for k in range(rnnz0):
            sid = radr0 + k
            col = efc_J_colind_in[worldid, 0, sid]
            res0 += efc_J_in[worldid, 0, sid] * qacc_in[worldid, col]
        else:
          for k in range(NV):
            res0 += efc_J_in[worldid, idx0, k] * qacc_in[worldid, k]
        res0 -= efc_aref_in[worldid, idx0]
        D_0 = efc_D_in[worldid, idx0]
        if D_0 > 0.0:
          res0 += efc_force_inout[worldid, idx0] / D_0

        # Rows 1,2
        if wp.static(IS_SPARSE):
          rnnz1 = efc_J_rownnz_in[worldid, idx1]
          radr1 = efc_J_rowadr_in[worldid, idx1]
          for k in range(rnnz1):
            sid = radr1 + k
            col = efc_J_colind_in[worldid, 0, sid]
            res1 += efc_J_in[worldid, 0, sid] * qacc_in[worldid, col]
        else:
          for k in range(NV):
            res1 += efc_J_in[worldid, idx1, k] * qacc_in[worldid, k]
        res1 -= efc_aref_in[worldid, idx1]
        D_1 = efc_D_in[worldid, idx1]
        if D_1 > 0.0:
          res1 += efc_force_inout[worldid, idx1] / D_1

        if wp.static(IS_SPARSE):
          rnnz2 = efc_J_rownnz_in[worldid, idx2]
          radr2 = efc_J_rowadr_in[worldid, idx2]
          for k in range(rnnz2):
            sid = radr2 + k
            col = efc_J_colind_in[worldid, 0, sid]
            res2 += efc_J_in[worldid, 0, sid] * qacc_in[worldid, col]
        else:
          for k in range(NV):
            res2 += efc_J_in[worldid, idx2, k] * qacc_in[worldid, k]
        res2 -= efc_aref_in[worldid, idx2]
        D_2 = efc_D_in[worldid, idx2]
        if D_2 > 0.0:
          res2 += efc_force_inout[worldid, idx2] / D_2

        if dim >= 4:
          if wp.static(IS_SPARSE):
            rnnz3 = efc_J_rownnz_in[worldid, idx3]
            radr3 = efc_J_rowadr_in[worldid, idx3]
            for k in range(rnnz3):
              sid = radr3 + k
              col = efc_J_colind_in[worldid, 0, sid]
              res3 += efc_J_in[worldid, 0, sid] * qacc_in[worldid, col]
          else:
            for k in range(NV):
              res3 += efc_J_in[worldid, idx3, k] * qacc_in[worldid, k]
          res3 -= efc_aref_in[worldid, idx3]
          D_3 = efc_D_in[worldid, idx3]
          if D_3 > 0.0:
            res3 += efc_force_inout[worldid, idx3] / D_3

        if dim >= 6:
          if wp.static(IS_SPARSE):
            rnnz4 = efc_J_rownnz_in[worldid, idx4]
            radr4 = efc_J_rowadr_in[worldid, idx4]
            for k in range(rnnz4):
              sid = radr4 + k
              col = efc_J_colind_in[worldid, 0, sid]
              res4 += efc_J_in[worldid, 0, sid] * qacc_in[worldid, col]
          else:
            for k in range(NV):
              res4 += efc_J_in[worldid, idx4, k] * qacc_in[worldid, k]
          res4 -= efc_aref_in[worldid, idx4]
          D_4 = efc_D_in[worldid, idx4]
          if D_4 > 0.0:
            res4 += efc_force_inout[worldid, idx4] / D_4

          if wp.static(IS_SPARSE):
            rnnz5 = efc_J_rownnz_in[worldid, idx5]
            radr5 = efc_J_rowadr_in[worldid, idx5]
            for k in range(rnnz5):
              sid = radr5 + k
              col = efc_J_colind_in[worldid, 0, sid]
              res5 += efc_J_in[worldid, 0, sid] * qacc_in[worldid, col]
          else:
            for k in range(NV):
              res5 += efc_J_in[worldid, idx5, k] * qacc_in[worldid, k]
          res5 -= efc_aref_in[worldid, idx5]
          D_5 = efc_D_in[worldid, idx5]
          if D_5 > 0.0:
            res5 += efc_force_inout[worldid, idx5] / D_5

        # Save old forces
        old0 = efc_force_inout[worldid, idx0]
        old1 = efc_force_inout[worldid, idx1]
        old2 = efc_force_inout[worldid, idx2]
        old3 = float(0.0)
        old4 = float(0.0)
        old5 = float(0.0)
        if dim >= 4:
          old3 = efc_force_inout[worldid, idx3]
        if dim >= 6:
          old4 = efc_force_inout[worldid, idx4]
          old5 = efc_force_inout[worldid, idx5]

        # ---- Solve elliptic QCQP (shared, with SOR blend) ----
        new0, new1, new2, new3, new4, new5, change = _pgs_solve_elliptic_cone(
          dim,
          old0,
          old1,
          old2,
          old3,
          old4,
          old5,
          res0,
          res1,
          res2,
          res3,
          res4,
          res5,
          AR_elliptic_in,
          worldid,
          idx0,
          mu,
          sor_omega,
        )

        # In Jacobi mode, always accept projected forces (no per-constraint
        # cost change rejection — the estimate is invalid when all forces
        # update simultaneously).
        wp.atomic_add(improvement_out, worldid, -change)

        efc_force_inout[worldid, idx0] = new0  # kernel_analyzer: ignore
        efc_force_inout[worldid, idx1] = new1  # kernel_analyzer: ignore
        efc_force_inout[worldid, idx2] = new2  # kernel_analyzer: ignore
        if dim >= 4:
          efc_force_inout[worldid, idx3] = new3  # kernel_analyzer: ignore
        if dim >= 6:
          efc_force_inout[worldid, idx4] = new4  # kernel_analyzer: ignore
          efc_force_inout[worldid, idx5] = new5  # kernel_analyzer: ignore

        return  # Done with elliptic contact

    # ---- 1D constraint (equality, friction, limit, frictionless contact) ----
    # Compute residual: res = J[i,:] · qacc - aref[i] + f[i]/D[i]
    res = float(0.0)
    if wp.static(IS_SPARSE):
      rownnz_i = efc_J_rownnz_in[worldid, efcid]
      rowadr_i = efc_J_rowadr_in[worldid, efcid]
      for k in range(rownnz_i):
        sparseid = rowadr_i + k
        col = efc_J_colind_in[worldid, 0, sparseid]
        res += efc_J_in[worldid, 0, sparseid] * qacc_in[worldid, col]
    else:
      for k in range(NV):
        res += efc_J_in[worldid, efcid, k] * qacc_in[worldid, k]
    res -= efc_aref_in[worldid, efcid]
    D_i = efc_D_in[worldid, efcid]
    if D_i > 0.0:
      res += efc_force_inout[worldid, efcid] / D_i

    # Use precomputed 1/AR[i,i]
    old_force = efc_force_inout[worldid, efcid]
    AR_diag_inv = AR_diag_inv_in[worldid, efcid]
    new_force = old_force - res * AR_diag_inv

    # Project onto constraint bounds
    if efcid >= ne and efcid < ne + nf:
      floss = efc_frictionloss_in[worldid, efcid]
      if new_force < -floss:
        new_force = -floss
      elif new_force > floss:
        new_force = floss
    elif efcid >= ne + nf:
      if new_force < 0.0:
        new_force = 0.0

    # SOR blend
    new_force = (1.0 - sor_omega) * old_force + sor_omega * new_force

    # In Jacobi mode, always accept projected forces. The per-constraint
    # cost change is invalid when all forces update simultaneously.
    delta = new_force - old_force
    AR_ii = 1.0 / AR_diag_inv if AR_diag_inv > 0.0 else 0.0
    change = 0.5 * delta * delta * AR_ii + delta * res
    wp.atomic_add(improvement_out, worldid, -change)

    efc_force_inout[worldid, efcid] = new_force  # kernel_analyzer: ignore

  return kernel


def _pgs_jacobi_recompute_qacc_kernel():
  """Factory for qacc recomputation kernel (deferred compilation)."""

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    qacc_smooth_in: wp.array2d[float],
    efc_force_in: wp.array2d[float],
    # In:
    BT_in: wp.array3d[float],
    done_in: wp.array[bool],
    # Data out:
    qacc_out: wp.array2d[float],
  ):
    """Recompute qacc = qacc_smooth + BT^T · f (parallel over DOFs).

    In Jacobi iteration, qacc is recomputed from scratch each iteration
    instead of being updated incrementally. This removes inter-constraint
    dependencies, enabling fully parallel force updates.
    """
    worldid, dofid = wp.tid()

    if done_in[worldid]:
      return

    nefc = nefc_in[worldid]

    val = qacc_smooth_in[worldid, dofid]
    for j in range(nefc):
      val += efc_force_in[worldid, j] * BT_in[worldid, j, dofid]
    qacc_out[worldid, dofid] = val

  return kernel


@event_scope
def _pgs_iteration_jacobi(
  m: types.Model,
  d: types.Data,
  ctx: PGSContext,
  nsolving: wp.array,
):
  """One Projected Jacobi PGS iteration: parallel force update + qacc recompute + convergence.

  Unlike GS which updates forces and qacc sequentially (constraint by constraint),
  Jacobi updates all forces in parallel using the current qacc, then recomputes
  qacc from scratch. This enables dim=(nworld, nefc) parallelism instead of
  dim=(nworld,).
  """
  ctx.improvement.zero_()

  # Step 1: Update all forces in parallel
  wp.launch(
    _pgs_jacobi_update_forces_kernel(m.nv, m.is_sparse),
    dim=(d.nworld, d.njmax),
    inputs=[
      m.opt.tolerance,
      d.ne,
      d.nf,
      d.nefc,
      d.qacc,
      d.contact.friction,
      d.contact.dim,
      d.contact.efc_address,
      d.efc.type,
      d.efc.id,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      d.efc.J,
      d.efc.D,
      d.efc.aref,
      d.efc.frictionloss,
      ctx.AR_diag_inv,
      ctx.AR_elliptic,
      m.opt.pgs_sor_omega,
      ctx.done,
    ],
    outputs=[d.efc.force, ctx.improvement],
  )

  # Step 2: Recompute qacc = qacc_smooth + BT^T · f
  wp.launch(
    _pgs_jacobi_recompute_qacc_kernel(),
    dim=(d.nworld, m.nv),
    inputs=[d.nefc, d.qacc_smooth, d.efc.force, ctx.BT, ctx.done],
    outputs=[d.qacc],
  )

  # Step 3: Convergence check
  wp.launch(
    _pgs_convergence_check,
    dim=(d.nworld,),
    inputs=[
      m.opt.tolerance,
      m.stat.meaninertia,
      ctx.improvement,
      m.nv,
      m.opt.iterations,
    ],
    outputs=[ctx.done, d.solver_niter, nsolving],
  )


@event_scope
def _pgs_iteration_sequential(
  m: types.Model,
  d: types.Data,
  ctx: PGSContext,
  nsolving: wp.array,
):
  """One PGS iteration: fused sweep + dual state + convergence check."""
  use_sparse_bt = m.ntree > 1 and m.is_sparse
  wp.launch(
    _pgs_sweep_kernel(m.nv, m.is_sparse, use_sparse_bt),
    dim=(d.nworld,),
    inputs=[
      d.ne,
      d.nf,
      d.nefc,
      d.efc.type,
      d.efc.frictionloss,
      d.efc.id,
      d.contact.dim,
      d.contact.friction,
      d.contact.efc_address,
      d.efc.J,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      ctx.BT,
      ctx.BT_rownnz,
      ctx.BT_rowadr,
      ctx.BT_colind,
      ctx.AR_diag_inv,
      ctx.AR_elliptic,
      d.efc.D,
      d.efc.aref,
      m.opt.tolerance,
      m.stat.meaninertia,
      m.opt.iterations,
    ],
    outputs=[d.efc.force, d.qacc, ctx.done, d.solver_niter, nsolving, d.efc.state],
  )


@event_scope
def _pgs_iteration_color(
  m: types.Model,
  d: types.Data,
  ctx: PGSContext,
  nsolving: wp.array,
):
  """One colored PGS iteration: parallel sweep per color + convergence."""
  use_sparse_bt = m.ntree > 1 and m.is_sparse
  ctx.improvement.zero_()

  # Parallel color passes
  for c in range(m.opt.pgs_max_colors):
    wp.launch(
      _pgs_color_sweep_kernel(m.nv, m.is_sparse, use_sparse_bt),
      dim=(d.nworld, d.njmax),
      inputs=[
        ctx.schedule_groups,
        ctx.schedule_offsets,
        c,
        ctx.group_efc_start,
        ctx.group_efc_count,
        d.ne,
        d.nf,
        d.nefc,
        d.efc.type,
        d.efc.frictionloss,
        d.efc.id,
        d.contact.dim,
        d.contact.friction,
        d.contact.efc_address,
        d.efc.J,
        d.efc.J_rownnz,
        d.efc.J_rowadr,
        d.efc.J_colind,
        ctx.BT,
        ctx.BT_rownnz,
        ctx.BT_rowadr,
        ctx.BT_colind,
        ctx.AR_diag_inv,
        ctx.AR_elliptic,
        d.efc.D,
        d.efc.aref,
        m.opt.tolerance,
        ctx.done,
      ],
      outputs=[d.efc.force, d.qacc, ctx.improvement],
    )

  # Sequential overflow pass for uncolored groups
  wp.launch(
    _pgs_overflow_sweep_kernel(m.nv, m.is_sparse, use_sparse_bt),
    dim=(d.nworld,),
    inputs=[
      ctx.overflow_list,
      ctx.n_overflow,
      ctx.group_efc_start,
      ctx.group_efc_count,
      d.ne,
      d.nf,
      d.nefc,
      d.efc.type,
      d.efc.frictionloss,
      d.efc.id,
      d.contact.dim,
      d.contact.friction,
      d.contact.efc_address,
      d.efc.J,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      ctx.BT,
      ctx.BT_rownnz,
      ctx.BT_rowadr,
      ctx.BT_colind,
      ctx.AR_diag_inv,
      ctx.AR_elliptic,
      d.efc.D,
      d.efc.aref,
      m.opt.tolerance,
      ctx.done,
    ],
    outputs=[d.efc.force, d.qacc, ctx.improvement],
  )

  # Convergence check
  wp.launch(
    _pgs_convergence_check,
    dim=(d.nworld,),
    inputs=[
      m.opt.tolerance,
      m.stat.meaninertia,
      ctx.improvement,
      m.nv,
      m.opt.iterations,
    ],
    outputs=[ctx.done, d.solver_niter, nsolving],
  )


@event_scope
def _pgs_compute_BT(m: types.Model, d: types.Data, ctx: PGSContext):
  """Phase 1: Compute BT[i,:] = M⁻¹ @ J[i,:] for all constraint rows."""
  if m.is_sparse:
    # Copy all sparse J rows into BT in one launch
    wp.launch(
      _pgs_copy_j_rows_sparse,
      dim=(d.nworld, d.njmax),
      inputs=[d.nefc, d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind, d.efc.J],
      outputs=[ctx.BT],
    )

    # Batched backsubstitution: BT[w,e,:] = M⁻¹ @ BT[w,e,:] for all (w,e)
    nlevels = len(m.qLD_updates)
    wp.launch(
      _pgs_solve_m_batched_sparse(m.nv, nlevels),
      dim=(d.nworld, d.njmax),
      inputs=[
        d.nefc,
        d.qLD,
        d.qLDiagInv,
        m.qLD_all_updates,
        m.qLD_level_offsets,
      ],
      outputs=[ctx.BT],
    )
  else:
    # Copy all dense J rows into BT in one launch
    wp.launch(
      _pgs_copy_j_rows_dense,
      dim=(d.nworld, d.njmax, m.nv),
      inputs=[d.nefc, d.efc.J],
      outputs=[ctx.BT],
    )

    # Batched backsubstitution using tile Cholesky
    for tile in m.qM_tiles:
      wp.launch_tiled(
        _pgs_solve_m_batched_dense(tile),
        dim=(d.nworld, d.njmax, tile.adr.size),
        inputs=[d.nefc, d.qLD, tile.adr],
        outputs=[ctx.BT],
        block_dim=m.block_dim.cholesky_solve,
      )

  # Compute sparse BT index for multi-tree models
  if m.ntree > 1 and m.is_sparse:
    _pgs_compute_bt_sparsity(m, d, ctx)


@event_scope
def _pgs_precompute_AR(m: types.Model, d: types.Data, ctx: PGSContext):
  """Precompute 1/AR[i,i] for 1D constraints and AR sub-blocks for elliptic."""
  wp.launch(
    _pgs_precompute_AR_diag_inv(m.nv, m.is_sparse),
    dim=(d.nworld, d.njmax),
    inputs=[d.nefc, d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind, d.efc.J, d.efc.D, ctx.BT],
    outputs=[ctx.AR_diag_inv],
  )
  wp.launch(
    _pgs_precompute_AR_elliptic(m.nv, m.is_sparse),
    dim=(d.nworld, d.njmax),
    inputs=[
      d.nefc,
      d.contact.dim,
      d.contact.efc_address,
      d.efc.type,
      d.efc.id,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      d.efc.J,
      d.efc.D,
      ctx.BT,
    ],
    outputs=[ctx.AR_elliptic],
  )


@event_scope
def _pgs_warmstart(m: types.Model, d: types.Data, ctx: PGSContext):
  """Warmstart: compute initial forces from qacc_warmstart."""
  wp.copy(d.qacc, d.qacc_smooth)

  if not (m.opt.disableflags & types.DisableBit.WARMSTART):
    # Compute jar = J @ qacc_warmstart - aref (reuse shared Jaref kernel)
    jar = wp.zeros((d.nworld, d.njmax), dtype=float)
    dofs_per_thread, threads_per_efc = _compute_jaref_threading(m.nv)
    wp.launch(
      solve_init_jaref(m.is_sparse, m.nv, dofs_per_thread),
      dim=(d.nworld, d.njmax, threads_per_efc),
      inputs=[d.nefc, d.qacc_warmstart, d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind, d.efc.J, d.efc.aref],
      outputs=[jar],
    )

    # Compute initial forces: force = -D * jar with projection
    wp.launch(
      _pgs_warmstart_forces,
      dim=(d.nworld, d.njmax),
      inputs=[
        d.ne,
        d.nf,
        d.nefc,
        d.contact.friction,
        d.contact.dim,
        d.contact.efc_address,
        d.efc.type,
        d.efc.id,
        d.efc.D,
        d.efc.frictionloss,
        m.opt.impratio_invsqrt,
        jar,
      ],
      outputs=[d.efc.force],
    )

    # Update qacc for warmstart forces: qacc += BT^T · f
    wp.launch(
      _pgs_update_qacc,
      dim=(d.nworld, m.nv),
      inputs=[d.nefc, d.efc.force, ctx.BT],
      outputs=[d.qacc],
    )

    # Evaluate warmstart cost (AR-free)
    # warmstart_cost is pre-allocated in PGSContext
    if m.is_sparse:
      wp.launch(
        _pgs_warmstart_cost_sparse,
        dim=(d.nworld,),
        inputs=[
          d.nefc,
          d.qacc,
          d.qacc_smooth,
          d.efc.J_rownnz,
          d.efc.J_rowadr,
          d.efc.J_colind,
          d.efc.J,
          d.efc.D,
          d.efc.aref,
          d.efc.force,
        ],
        outputs=[ctx.warmstart_cost],
      )
    else:
      wp.launch(
        _pgs_warmstart_cost_dense,
        dim=(d.nworld,),
        inputs=[d.nefc, d.qacc, d.qacc_smooth, d.efc.J, d.efc.D, d.efc.aref, d.efc.force],
        outputs=[ctx.warmstart_cost],
      )

    # If cost > 0 or NaN, zero out forces and reset qacc
    wp.launch(
      _pgs_warmstart_select,
      dim=(d.nworld,),
      inputs=[d.nefc, d.qacc_smooth, ctx.warmstart_cost, m.nv],
      outputs=[d.efc.force, d.qfrc_constraint, d.qacc],
    )
  else:
    # Coldstart: zero forces
    d.efc.force.zero_()


@event_scope
def _pgs_iterations(m: types.Model, d: types.Data, ctx: PGSContext, use_coloring: bool):
  """Main PGS iteration loop."""
  d.solver_niter.zero_()
  ctx.done.zero_()

  nsolving = wp.full(shape=(1,), value=d.nworld, dtype=int)

  # Jacobi mode: fully parallel, no coloring needed
  if m.opt.pgs_mode == 1:
    if m.opt.iterations != 0 and m.opt.graph_conditional:
      wp.capture_while(
        nsolving,
        while_body=_pgs_iteration_jacobi,
        m=m,
        d=d,
        ctx=ctx,
        nsolving=nsolving,
      )
    else:
      for _ in range(m.opt.iterations):
        _pgs_iteration_jacobi(m, d, ctx, nsolving)
  # GS mode: sequential or colored
  elif use_coloring:
    if m.opt.iterations != 0 and m.opt.graph_conditional:
      wp.capture_while(
        nsolving,
        while_body=_pgs_iteration_color,
        m=m,
        d=d,
        ctx=ctx,
        nsolving=nsolving,
      )
    else:
      for _ in range(m.opt.iterations):
        _pgs_iteration_color(m, d, ctx, nsolving)
  elif m.opt.iterations != 0 and m.opt.graph_conditional:
    wp.capture_while(
      nsolving,
      while_body=_pgs_iteration_sequential,
      m=m,
      d=d,
      ctx=ctx,
      nsolving=nsolving,
    )
  else:
    for _ in range(m.opt.iterations):
      _pgs_iteration_sequential(m, d, ctx, nsolving)


@event_scope
def _pgs_dual_finish(m: types.Model, d: types.Data, ctx: PGSContext, use_coloring: bool):
  """Compute dual state and map to joint space (qfrc_constraint)."""
  # Compute dual state (split out from sweep for colored and Jacobi paths)
  if use_coloring or m.opt.pgs_mode == 1:
    wp.launch(
      _pgs_dual_state,
      dim=(d.nworld, d.njmax),
      inputs=[d.ne, d.nf, d.nefc, d.efc.type, d.efc.frictionloss, d.efc.force],
      outputs=[d.efc.state],
    )

  # Map to joint space (dualFinish)
  # qfrc_constraint = J^T @ efc_force
  # Must process ALL worlds (including converged ones)
  ctx.done.zero_()
  d.qfrc_constraint.zero_()
  if m.is_sparse:
    wp.launch(
      update_constraint_init_qfrc_constraint_sparse,
      dim=(d.nworld, d.njmax),
      inputs=[d.nefc, d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind, d.efc.J, d.efc.force, ctx.done],
      outputs=[d.qfrc_constraint],
    )
  else:
    wp.launch(
      update_constraint_init_qfrc_constraint_dense,
      dim=(d.nworld, m.nv),
      inputs=[d.nefc, d.efc.J, d.efc.force, d.njmax, ctx.done],
      outputs=[d.qfrc_constraint],
    )


@event_scope
def _solve_pgs(m: types.Model, d: types.Data, ctx: PGSContext):
  """PGS (Projected Gauss-Seidel) constraint solver.

  Operates in dual (constraint force) space following MuJoCo C's mj_solPGS.
  Uses AR-free formulation: maintains qacc incrementally and computes
  residuals on-the-fly via J and BT = M⁻¹·Jᵀ.

  When sparse Jacobians are available, uses constraint coloring for
  parallel execution of independent constraints within each color.

  TODO(team): Add island support.
  TODO(team): Add noslip post-processing solver.
  """
  _pgs_compute_BT(m, d, ctx)
  _pgs_precompute_AR(m, d, ctx)
  _pgs_warmstart(m, d, ctx)

  use_coloring = m.opt.pgs_max_colors > 0
  if use_coloring:
    _pgs_coloring(m, d, ctx)

  _pgs_iterations(m, d, ctx, use_coloring)
  _pgs_dual_finish(m, d, ctx, use_coloring)

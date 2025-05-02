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
from . import smooth
from . import support
from . import types
from .warp_util import event_scope
from .warp_util import kernel


@wp.func
def _rescale(m: types.Model, value: float) -> float:
  return value / (m.stat.meaninertia * float(wp.max(1, m.nv)))


@wp.func
def _in_bracket(x: wp.vec3, y: wp.vec3) -> bool:
  return (x[1] < y[1] and y[1] < 0.0) or (x[1] > y[1] and y[1] > 0.0)


@wp.func
def _eval_pt(quad: wp.vec3, alpha: wp.float32) -> wp.vec3:
  return wp.vec3(
    alpha * alpha * quad[2] + alpha * quad[1] + quad[0],
    2.0 * alpha * quad[2] + quad[1],
    2.0 * quad[2],
  )


@wp.func
def _eval_pt_elliptic(
  # Model:
  opt_impratio: float,
  # In:
  friction: wp.vec5,
  u0: float,
  uu: float,
  uv: float,
  vv: float,
  jv: float,
  D: float,
  quad: wp.vec3,
  alpha: float,
) -> wp.vec3:
  mu = friction[0] / opt_impratio
  v0 = jv * mu
  n = u0 + alpha * v0
  tsqr = uu + alpha * (2.0 * uv + alpha * vv)
  t = wp.sqrt(tsqr)  # tangential force

  bottom_zone = ((tsqr <= 0.0) and (n < 0)) or ((tsqr > 0.0) and ((mu * n + t) <= 0.0))
  middle_zone = (tsqr > 0) and (n < (mu * t)) and ((mu * n + t) > 0.0)

  # elliptic bottom zone: quadratic cose
  if bottom_zone:
    pt = _eval_pt(quad, alpha)
  else:
    pt = wp.vec3(0.0)

  # elliptic middle zone
  if t == 0.0:
    t += types.MJ_MINVAL

  if tsqr == 0.0:
    tsqr += types.MJ_MINVAL

  n1 = v0
  t1 = (uv + alpha * vv) / t
  t2 = vv / t - (uv + alpha * vv) * t1 / tsqr

  if middle_zone:
    mu2 = mu * mu
    dm = D / wp.max(mu2 * (1.0 + mu2), types.MJ_MINVAL)
    nmt = n - mu * t
    n1mut1 = n1 - mu * t1

    pt += wp.vec3(
      0.5 * dm * nmt * nmt,
      dm * nmt * n1mut1,
      dm * (n1mut1 * n1mut1 - nmt * mu * t2),
    )

  return pt


@wp.kernel
def linesearch_iterative_gtol(
  # Model:
  nv: int,
  stat_meaninertia_in: wp.array(dtype=float),
  opt_tolerance_in: float,
  opt_ls_tolerance_in: float,
  # Data in:
  efc_search_dot_in: wp.array(dtype=float),
  efc_done_in: wp.array(dtype=bool),
  # Data out:
  efc_gtol_out: wp.array(dtype=float),
):
  worldid = wp.tid()

  if efc_done_in[worldid]:
    return

  snorm = wp.math.sqrt(efc_search_dot_in[worldid])
  scale = stat_meaninertia_in * wp.float(wp.max(1, nv))
  efc_gtol_out[worldid] = opt_tolerance_in * opt_ls_tolerance_in * snorm * scale


@wp.kernel
def linesearch_iterative_init_p0_gauss(
  # Data in:
  efc_quad_gauss_in: wp.array(dtype=wp.vec3),
  efc_done_in: wp.array(dtype=bool),
  # Data out:
  efc_p0_out: wp.array(dtype=wp.vec3),
):
  worldid = wp.tid()

  if efc_done_in[worldid]:
    return

  quad = efc_quad_gauss_in[worldid]
  efc_p0_out[worldid] = wp.vec3(quad[0], quad[1], 2.0 * quad[2])


@wp.kernel
def linesearch_iterative_init_p0_elliptic0(
  # Data in:
  nefc_in: wp.array(dtype=int),
  ne_in: wp.array(dtype=int),
  nf_in: wp.array(dtype=int),
  nl_in: wp.array(dtype=int),
  efc_worldid_in: wp.array(dtype=int),
  efc_Jaref_in: wp.array(dtype=float),
  efc_condim_in: wp.array(dtype=int),
  efc_quad_in: wp.array(dtype=wp.vec3),
  efc_done_in: wp.array(dtype=bool),
  # Data out:
  efc_p0_out: wp.array(dtype=wp.vec3),
):
  efcid = wp.tid()

  if efcid >= nefc_in:
    return

  worldid = efc_worldid_in[efcid]

  if efc_done_in[worldid]:
    return

  active = efc_Jaref_in[efcid] < 0.0

  nef = ne_in[0] + nf_in[0]
  nefl = nef + nl_in[0]
  if efcid < nef:
    active = True
  elif efcid >= nefl and efc_condim_in[efcid] > 1:
    active = False

  if active:
    quad = efc_quad_in[efcid]
    wp.atomic_add(efc_p0_out, worldid, wp.vec3(quad[0], quad[1], 2.0 * quad[2]))


@wp.kernel
def linesearch_iterative_init_p0_elliptic1(
  # Model:
  opt_impratio: float,
  # Data in:
  ncon_in: wp.array(dtype=int),
  contact_friction_in: wp.array(dtype=wp.vec5),
  contact_dim_in: wp.array(dtype=int),
  contact_efc_address_in: wp.array2d(dtype=int),
  contact_worldid_in: wp.array(dtype=int),
  efc_D_in: wp.array(dtype=float),
  efc_jv_in: wp.array(dtype=float),
  efc_quad_in: wp.array(dtype=wp.vec3),
  efc_u_in: wp.array2d(dtype=float),
  efc_uu_in: wp.array(dtype=float),
  efc_uv_in: wp.array(dtype=float),
  efc_vv_in: wp.array(dtype=float),
  efc_done_in: wp.array(dtype=bool),
  # Data out:
  efc_p0_out: wp.array(dtype=wp.vec3),
):
  conid = wp.tid()

  if conid >= ncon_in:
    return

  if efc_done_in[contact_worldid_in[conid]]:
    return

  if contact_dim_in[conid] < 2:
    return

  efcid = contact_efc_address_in[conid, 0]
  worldid = contact_worldid_in[conid]

  pt = _eval_pt_elliptic(
    opt_impratio,
    contact_friction_in[conid],
    efc_u_in[conid, 0],
    efc_uu_in[conid],
    efc_uv_in[conid],
    efc_vv_in[conid],
    efc_jv_in[efcid],
    efc_D_in[efcid],
    efc_quad_in[efcid],
    0.0,
  )

  wp.atomic_add(efc_p0_out, worldid, pt)


@wp.kernel
def linesearch_iterative_init_p0_pyramidal(
  # Data in:
  nefc_in: wp.array(dtype=int),
  ne_in: wp.array(dtype=int),
  nf_in: wp.array(dtype=int),
  efc_worldid_in: wp.array(dtype=int),
  efc_Jaref_in: wp.array(dtype=float),
  efc_quad_in: wp.array(dtype=wp.vec3),
  efc_done_in: wp.array(dtype=bool),
  # Data out:
  efc_p0_out: wp.array(dtype=wp.vec3),
):
  efcid = wp.tid()

  if efcid >= nefc_in:
    return

  worldid = efc_worldid_in[efcid]

  if efc_done_in[worldid]:
    return

  if efc_Jaref_in[efcid] >= 0.0 and efcid >= ne_in[0] + nf_in[0]:
    return

  quad = efc_quad_in[efcid]

  wp.atomic_add(efc_p0_out, worldid, wp.vec3(quad[0], quad[1], 2.0 * quad[2]))


@wp.kernel
def linesearch_iterative_init_lo_gauss(
  # Data in:
  efc_quad_gauss_in: wp.array(dtype=wp.vec3),
  efc_done_in: wp.array(dtype=bool),
  efc_p0_in: wp.array(dtype=wp.vec3),
  # Data out:
  efc_lo_out: wp.array(dtype=wp.vec3),
  efc_lo_alpha_out: wp.array(dtype=wp.float32),
):
  worldid = wp.tid()

  if efc_done_in[worldid]:
    return

  p0 = efc_p0_in[worldid]
  alpha = -math._safe_div(p0[1], p0[2])
  efc_lo_out[worldid] = _eval_pt(efc_quad_gauss_in[worldid], alpha)
  efc_lo_alpha_out[worldid] = alpha


@wp.kernel
def linesearch_iterative_init_lo_elliptic0(
  # Data in:
  nefc_in: wp.array(dtype=int),
  ne_in: wp.array(dtype=int),
  nf_in: wp.array(dtype=int),
  nl_in: wp.array(dtype=int),
  efc_worldid_in: wp.array(dtype=int),
  efc_Jaref_in: wp.array(dtype=float),
  efc_jv_in: wp.array(dtype=float),
  efc_condim_in: wp.array(dtype=int),
  efc_lo_alpha_in: wp.array(dtype=wp.float32),
  efc_quad_in: wp.array(dtype=wp.vec3),
  efc_done_in: wp.array(dtype=bool),
  # Data out:
  efc_lo_out: wp.array(dtype=wp.vec3),
):
  efcid = wp.tid()

  if efcid >= nefc_in:
    return

  worldid = efc_worldid_in[efcid]

  if efc_done_in[worldid]:
    return

  alpha = efc_lo_alpha_in[worldid]

  active = efc_Jaref_in[efcid] + alpha * efc_jv_in[efcid] < 0.0

  nef = ne_in[0] + nf_in[0]
  nefl = nef + nl_in[0]
  if efcid < nef:
    active = True
  elif efcid >= nefl and efc_condim_in[efcid] > 1:
    active = False

  if active:
    wp.atomic_add(efc_lo_out, worldid, _eval_pt(efc_quad_in[efcid], alpha))


@wp.kernel
def linesearch_iterative_init_lo_elliptic1(
  # Model:
  opt_impratio: float,
  # Data in:
  ncon_in: wp.array(dtype=int),
  contact_worldid_in: wp.array(dtype=int),
  contact_dim_in: wp.array(dtype=int),
  contact_efc_address_in: wp.array2d(dtype=int),
  contact_friction_in: wp.array(dtype=wp.vec5),
  efc_D_in: wp.array(dtype=float),
  efc_lo_alpha_in: wp.array(dtype=wp.float32),
  efc_quad_in: wp.array(dtype=wp.vec3),
  efc_u_in: wp.array2d(dtype=float),
  efc_uu_in: wp.array(dtype=float),
  efc_uv_in: wp.array(dtype=float),
  efc_vv_in: wp.array(dtype=float),
  efc_jv_in: wp.array(dtype=float),
  efc_done_in: wp.array(dtype=bool),
  # Data out:
  efc_lo_out: wp.array(dtype=wp.vec3),
):
  conid = wp.tid()

  if conid >= ncon_in:
    return

  if efc_done_in[contact_worldid_in[conid]]:
    return

  if contact_dim_in[conid] < 2:
    return

  efcid = contact_efc_address_in[conid, 0]
  worldid = contact_worldid_in[conid]
  alpha = efc_lo_alpha_in[worldid]
  pt = _eval_pt_elliptic(
    opt_impratio,
    contact_friction_in[conid],
    efc_u_in[conid, 0],
    efc_uu_in[conid],
    efc_uv_in[conid],
    efc_vv_in[conid],
    efc_jv_in[efcid],
    efc_D_in[efcid],
    efc_quad_in[efcid],
    alpha,
  )
  wp.atomic_add(efc_lo_out, worldid, pt)


@wp.kernel
def linesearch_iterative_init_lo_pyramidal(
  # Data in:
  nefc_in: wp.array(dtype=int),
  ne_in: wp.array(dtype=int),
  nf_in: wp.array(dtype=int),
  efc_Jaref_in: wp.array(dtype=float),
  efc_jv_in: wp.array(dtype=float),
  efc_lo_alpha_in: wp.array(dtype=wp.float32),
  efc_quad_in: wp.array(dtype=wp.vec3),
  efc_done_in: wp.array(dtype=bool),
  efc_worldid_in: wp.array(dtype=int),
  # Data out:
  efc_lo_out: wp.array(dtype=wp.vec3),
):
  efcid = wp.tid()

  if efcid >= nefc_in:
    return

  worldid = efc_worldid_in[efcid]

  if efc_done_in[worldid]:
    return

  alpha = efc_lo_alpha_in[worldid]

  if efc_Jaref_in[efcid] + alpha * efc_jv_in[efcid] < 0.0 or (
    efcid < ne_in[0] + nf_in[0]
  ):
    wp.atomic_add(efc_lo_out, worldid, _eval_pt(efc_quad_in[efcid], alpha))


@wp.kernel
def linesearch_iterative_init_bounds(
  # Data in:
  efc_done_in: wp.array(dtype=bool),
  efc_p0_in: wp.array(dtype=wp.vec3),
  efc_lo_in: wp.array(dtype=wp.vec3),
  efc_lo_alpha_in: wp.array(dtype=wp.float32),
  # Data out:
  efc_lo_out: wp.array(dtype=wp.vec3),
  efc_lo_alpha_out: wp.array(dtype=wp.float32),
  efc_hi_out: wp.array(dtype=wp.vec3),
  efc_hi_alpha_out: wp.array(dtype=wp.float32),
):
  worldid = wp.tid()

  if efc_done_in[worldid]:
    return

  p0 = efc_p0_in[worldid]
  lo = efc_lo_in[worldid]
  lo_alpha = efc_lo_alpha_in[worldid]
  lo_less = lo[1] < p0[1]

  efc_lo_out[worldid] = wp.where(lo_less, lo, p0)
  efc_lo_alpha_out[worldid] = wp.where(lo_less, lo_alpha, 0.0)
  efc_hi_out[worldid] = wp.where(lo_less, p0, lo)
  efc_hi_alpha_out[worldid] = wp.where(lo_less, 0.0, lo_alpha)


@wp.kernel
def linesearch_iterative_next_alpha_gauss(
  # Data in:
  efc_ls_done_in: wp.array(dtype=bool),
  efc_done_in: wp.array(dtype=bool),
  efc_lo_in: wp.array(dtype=wp.vec3),
  efc_lo_alpha_in: wp.array(dtype=wp.float32),
  efc_hi_in: wp.array(dtype=wp.vec3),
  efc_hi_alpha_in: wp.array(dtype=wp.float32),
  efc_quad_gauss_in: wp.array(dtype=wp.vec3),
  # Data out:
  efc_lo_next_out: wp.array(dtype=wp.vec3),
  efc_lo_next_alpha_out: wp.array(dtype=wp.float32),
  efc_hi_next_out: wp.array(dtype=wp.vec3),
  efc_hi_next_alpha_out: wp.array(dtype=wp.float32),
  efc_mid_out: wp.array(dtype=wp.vec3),
  efc_mid_alpha_out: wp.array(dtype=wp.float32),
):
  worldid = wp.tid()

  if efc_ls_done_in[worldid]:
    return

  if efc_done_in[worldid]:
    return

  quad = efc_quad_gauss_in[worldid]

  lo = efc_lo_in[worldid]
  lo_alpha = efc_lo_alpha_in[worldid]
  lo_next_alpha = lo_alpha - math._safe_div(lo[1], lo[2])
  efc_lo_next_out[worldid] = _eval_pt(quad, lo_next_alpha)
  efc_lo_next_alpha_out[worldid] = lo_next_alpha

  hi = efc_hi_in[worldid]
  hi_alpha = efc_hi_alpha_in[worldid]
  hi_next_alpha = hi_alpha - math._safe_div(hi[1], hi[2])
  efc_hi_next_out[worldid] = _eval_pt(quad, hi_next_alpha)
  efc_hi_next_alpha_out[worldid] = hi_next_alpha

  mid_alpha = 0.5 * (lo_alpha + hi_alpha)
  efc_mid_out[worldid] = _eval_pt(quad, mid_alpha)
  efc_mid_alpha_out[worldid] = mid_alpha


@wp.kernel
def linesearch_iterative_next_quad_elliptic0(
  # Data in:
  nefc_in: wp.array(dtype=int),
  ne_in: wp.array(dtype=int),
  nf_in: wp.array(dtype=int),
  nl_in: wp.array(dtype=int),
  efc_ls_done_in: wp.array(dtype=bool),
  efc_done_in: wp.array(dtype=bool),
  efc_Jaref_in: wp.array(dtype=float),
  efc_jv_in: wp.array(dtype=float),
  efc_quad_in: wp.array(dtype=wp.vec3),
  efc_lo_next_alpha_in: wp.array(dtype=wp.float32),
  efc_hi_next_alpha_in: wp.array(dtype=wp.float32),
  efc_mid_alpha_in: wp.array(dtype=wp.float32),
  efc_condim_in: wp.array(dtype=int),
  efc_worldid_in: wp.array(dtype=int),
  # Data out:
  efc_lo_next_out: wp.array(dtype=wp.vec3),
  efc_hi_next_out: wp.array(dtype=wp.vec3),
  efc_mid_out: wp.array(dtype=wp.vec3),
):
  efcid = wp.tid()

  if efcid >= nefc_in[0]:
    return

  worldid = efc_worldid_in[efcid]

  if efc_done_in[worldid]:
    return

  if efc_ls_done_in[worldid]:
    return

  nef = ne_in[0] + nf_in[0]
  nefl = nef + nl_in[0]

  quad = efc_quad_in[efcid]
  jaref = efc_Jaref_in[efcid]
  jv = efc_jv_in[efcid]

  alpha = efc_lo_next_alpha_in[worldid]

  active = jaref + alpha * jv < 0.0
  if efcid < nef:
    active = True
  elif efcid >= nefl and efc_condim_in[efcid] > 1:
    active = False

  if active:
    wp.atomic_add(efc_lo_next_out, worldid, _eval_pt(quad, alpha))

  alpha = efc_hi_next_alpha_in[worldid]

  active = jaref + alpha * jv < 0.0
  if efcid < nef:
    active = True
  elif efcid >= nefl and efc_condim_in[efcid] > 1:
    active = False

  if active:
    wp.atomic_add(efc_hi_next_out, worldid, _eval_pt(quad, alpha))

  alpha = efc_mid_alpha_in[worldid]

  active = jaref + alpha * jv < 0.0
  if efcid < nef:
    active = True
  elif efcid >= nefl and efc_condim_in[efcid] > 1:
    active = False

  if active:
    wp.atomic_add(efc_mid_out, worldid, _eval_pt(quad, alpha))


@wp.kernel
def linesearch_iterative_next_quad_elliptic1(
  # Model:
  opt_impratio: float,
  # Data in:
  ncon_in: wp.array(dtype=int),
  contact_worldid_in: wp.array(dtype=int),
  contact_dim_in: wp.array(dtype=int),
  contact_efc_address_in: wp.array2d(dtype=int),
  contact_friction_in: wp.array(dtype=wp.vec5),
  efc_ls_done_in: wp.array(dtype=bool),
  efc_done_in: wp.array(dtype=bool),
  efc_Jaref_in: wp.array(dtype=float),
  efc_jv_in: wp.array(dtype=float),
  efc_D_in: wp.array(dtype=float),
  efc_u_in: wp.array2d(dtype=float),
  efc_uu_in: wp.array(dtype=float),
  efc_uv_in: wp.array(dtype=float),
  efc_vv_in: wp.array(dtype=float),
  efc_quad_in: wp.array(dtype=wp.vec3),
  efc_lo_next_alpha_in: wp.array(dtype=wp.float32),
  efc_hi_next_alpha_in: wp.array(dtype=wp.float32),
  efc_mid_alpha_in: wp.array(dtype=wp.float32),
  # Data out:
  efc_lo_next_out: wp.array(dtype=wp.vec3),
  efc_hi_next_out: wp.array(dtype=wp.vec3),
  efc_mid_out: wp.array(dtype=wp.vec3),
):
  conid = wp.tid()

  if conid >= ncon_in[0]:
    return

  worldid = contact_worldid_in[conid]

  if efc_done_in[worldid]:
    return

  if contact_dim_in[conid] < 2:
    return

  efcid = contact_efc_address_in[conid, 0]

  friction = contact_friction_in[conid]
  u = efc_u_in[conid, 0]
  uu = efc_uu_in[conid]
  uv = efc_uv_in[conid]
  vv = efc_vv_in[conid]
  jv = efc_jv_in[efcid]
  d = efc_D_in[efcid]
  quad = efc_quad_in[efcid]

  alpha = efc_lo_next_alpha_in[worldid]
  pt = _eval_pt_elliptic(opt_impratio, friction, u, uu, uv, vv, jv, d, quad, alpha)
  wp.atomic_add(efc_lo_next_out, worldid, pt)

  alpha = efc_hi_next_alpha_in[worldid]
  pt = _eval_pt_elliptic(opt_impratio, friction, u, uu, uv, vv, jv, d, quad, alpha)
  wp.atomic_add(efc_hi_next_out, worldid, pt)

  alpha = efc_mid_alpha_in[worldid]
  pt = _eval_pt_elliptic(opt_impratio, friction, u, uu, uv, vv, jv, d, quad, alpha)
  wp.atomic_add(efc_mid_out, worldid, pt)


@wp.kernel
def linesearch_iterative_next_quad_pyramidal(
  # Data in:
  nefc_in: wp.array(dtype=int),
  ne_in: wp.array(dtype=int),
  nf_in: wp.array(dtype=int),
  efc_ls_done_in: wp.array(dtype=bool),
  efc_done_in: wp.array(dtype=bool),
  efc_Jaref_in: wp.array(dtype=float),
  efc_jv_in: wp.array(dtype=float),
  efc_quad_in: wp.array(dtype=wp.vec3),
  efc_lo_next_alpha_in: wp.array(dtype=wp.float32),
  efc_hi_next_alpha_in: wp.array(dtype=wp.float32),
  efc_mid_alpha_in: wp.array(dtype=wp.float32),
  efc_worldid_in: wp.array(dtype=int),
  # Data out:
  efc_lo_next_out: wp.array(dtype=wp.vec3),
  efc_hi_next_out: wp.array(dtype=wp.vec3),
  efc_mid_out: wp.array(dtype=wp.vec3),
):
  efcid = wp.tid()

  if efcid >= nefc_in[0]:
    return

  worldid = efc_worldid_in[efcid]

  if efc_done_in[worldid]:
    return

  if efc_ls_done_in[worldid]:
    return

  nef_active = efcid < ne_in[0] + nf_in[0]

  quad = efc_quad_in[efcid]
  jaref = efc_Jaref_in[efcid]
  jv = efc_jv_in[efcid]

  alpha = efc_lo_next_alpha_in[worldid]
  if jaref + alpha * jv < 0.0 or nef_active:
    wp.atomic_add(efc_lo_next_out, worldid, _eval_pt(quad, alpha))

  alpha = efc_hi_next_alpha_in[worldid]
  if jaref + alpha * jv < 0.0 or nef_active:
    wp.atomic_add(efc_hi_next_out, worldid, _eval_pt(quad, alpha))

  alpha = efc_mid_alpha_in[worldid]
  if jaref + alpha * jv < 0.0 or nef_active:
    wp.atomic_add(efc_mid_out, worldid, _eval_pt(quad, alpha))


@wp.kernel
def linesearch_iterative_swap(
  # Data in:
  efc_done_in: wp.array(dtype=bool),
  efc_ls_done_in: wp.array(dtype=bool),
  efc_gtol_in: wp.array(dtype=float),
  efc_lo_in: wp.array(dtype=wp.vec3),
  efc_lo_alpha_in: wp.array(dtype=wp.float32),
  efc_hi_in: wp.array(dtype=wp.vec3),
  efc_hi_alpha_in: wp.array(dtype=wp.float32),
  efc_lo_next_in: wp.array(dtype=wp.vec3),
  efc_lo_next_alpha_in: wp.array(dtype=wp.float32),
  efc_hi_next_in: wp.array(dtype=wp.vec3),
  efc_hi_next_alpha_in: wp.array(dtype=wp.float32),
  efc_mid_in: wp.array(dtype=wp.vec3),
  efc_mid_alpha_in: wp.array(dtype=wp.float32),
  efc_p0_in: wp.array(dtype=wp.vec3),
  # Data out:
  efc_ls_done_out: wp.array(dtype=bool),
  efc_alpha_out: wp.array(dtype=wp.float32),
  efc_lo_out: wp.array(dtype=wp.vec3),
  efc_lo_alpha_out: wp.array(dtype=wp.float32),
  efc_hi_out: wp.array(dtype=wp.vec3),
  efc_hi_alpha_out: wp.array(dtype=wp.float32),
):
  worldid = wp.tid()

  if efc_done_in[worldid]:
    return

  if efc_ls_done_in[worldid]:
    return

  lo = efc_lo_in[worldid]
  lo_alpha = efc_lo_alpha_in[worldid]
  hi = efc_hi_in[worldid]
  hi_alpha = efc_hi_alpha_in[worldid]
  lo_next = efc_lo_next_in[worldid]
  lo_next_alpha = efc_lo_next_alpha_in[worldid]
  hi_next = efc_hi_next_in[worldid]
  hi_next_alpha = efc_hi_next_alpha_in[worldid]
  mid = efc_mid_in[worldid]
  mid_alpha = efc_mid_alpha_in[worldid]

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
  efc_lo_out[worldid] = lo
  efc_lo_alpha_out[worldid] = lo_alpha
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
  efc_hi_out[worldid] = hi
  efc_hi_alpha_out[worldid] = hi_alpha
  swap_hi = swap_hi_hi_next or swap_hi_mid or swap_hi_lo_next

  # if we did not adjust the interval, we are done
  # also done if either low or hi slope is nearly flat
  gtol = efc_gtol_in[worldid]
  efc_ls_done_out[worldid] = (
    (not swap_lo and not swap_hi)
    or (lo[1] < 0 and lo[1] > -gtol)
    or (hi[1] > 0 and hi[1] < gtol)
  )

  # update alpha if we have an improvement
  p0 = efc_p0_in[worldid]
  alpha = 0.0
  improved = lo[0] < p0[0] or hi[0] < p0[0]
  lo_better = lo[0] < hi[0]
  alpha = wp.where(improved and lo_better, lo_alpha, alpha)
  alpha = wp.where(improved and not lo_better, hi_alpha, alpha)
  efc_alpha_out[worldid] = alpha


def _linesearch_iterative(m: types.Model, d: types.Data):
  """Iterative linesearch."""

  wp.launch(
    linesearch_iterative_gtol,
    dim=(d.nworld,),
    inputs=[
      m.nv, m.stat.meaninertia, m.opt.tolerance, m.opt.ls_tolerance, d.efc.search_dot,
      d.efc.done
    ],
    outputs=[d.efc.gtol])  # fmt: skip

  wp.launch(
    linesearch_iterative_init_p0_gauss,
    dim=(d.nworld,),
    inputs=[
      d.efc.quad_gauss,
      d.efc.done
    ],
    outputs=[d.efc.p0])  # fmt: skip

  if m.opt.cone == types.ConeType.ELLIPTIC:
    wp.launch(
      linesearch_iterative_init_p0_elliptic0,
      dim=(d.njmax,),
      inputs=[
        d.nefc, d.ne, d.nf, d.nl, d.efc.worldid, d.efc.Jaref, d.efc.condim, d.efc.quad,
        d.efc.done
      ],
      outputs=[d.efc.p0])  # fmt: skip
    wp.launch(
      linesearch_iterative_init_p0_elliptic1,
      dim=(d.nconmax),
      inputs=[
        m.opt.impratio, d.ncon, d.contact.friction, d.contact.dim,
        d.contact.efc_address, d.contact.worldid, d.efc.D, d.efc.jv, d.efc.quad,
        d.efc.u, d.efc.uu, d.efc.uv, d.efc.vv, d.efc.done
      ],
      outputs=[d.efc.p0])  # fmt: skip
  else:
    wp.launch(
      linesearch_iterative_init_p0_pyramidal,
      dim=(d.njmax,),
      inputs=[
        d.nefc, d.ne, d.nf, d.efc.worldid, d.efc.Jaref, d.efc.quad, d.efc.done
      ], outputs=[d.efc.p0])  # fmt: skip

  wp.launch(
    linesearch_iterative_init_lo_gauss,
    dim=(d.nworld,),
    inputs=[
      d.efc.quad_gauss, d.efc.done, d.efc.p0
    ],
    outputs=[d.efc.lo, d.efc.lo_alpha])  # fmt: skip

  if m.opt.cone == types.ConeType.ELLIPTIC:
    wp.launch(
      linesearch_iterative_init_lo_elliptic0,
      dim=(d.njmax,),
      inputs=[
        d.nefc, d.ne, d.nf, d.nl, d.efc.worldid, d.efc.Jaref, d.efc.jv, d.efc.condim,
        d.efc.lo_alpha, d.efc.quad, d.efc.done
      ],
      outputs=[d.efc.lo, d.efc.lo_alpha])  # fmt: skip
    wp.launch(
      linesearch_iterative_init_lo_elliptic1,
      dim=(d.nconmax),
      inputs=[
        m.opt.impratio, d.ncon, d.contact.worldid, d.contact.dim, d.contact.efc_address,
        d.contact.friction, d.efc.D, d.efc.lo_alpha, d.efc.quad, d.efc.u, d.efc.uu,
        d.efc.uv, d.efc.vv, d.efc.jv, d.efc.done
      ],
      outputs=[d.efc.lo])  # fmt: skip
  else:
    wp.launch(
      linesearch_iterative_init_lo_pyramidal,
      dim=(d.njmax,),
      inputs=[
        d.nefc, d.ne, d.nf, d.efc.Jaref, d.efc.jv, d.efc.lo_alpha, d.efc.quad,
        d.efc.done, d.efc.worldid
      ],
      outputs=[d.efc.lo])  # fmt: skip

  # set the lo/hi interval bounds

  wp.launch(
    linesearch_iterative_init_bounds,
    dim=(d.nworld,),
    inputs=[
      d.efc.done, d.efc.p0, d.efc.lo, d.efc.lo_alpha
    ],
    outputs=[
      d.efc.lo, d.efc.lo_alpha, d.efc.hi, d.efc.hi_alpha
    ])  # fmt: skip

  for _ in range(m.opt.ls_iterations):
    # note: we always launch ls_iterations kernels, but the kernels may early exit if done is true
    # this allows us to preserve cudagraph requirements (no dynamic kernel launching) at the expense
    # of extra launches
    wp.launch(
      linesearch_iterative_next_alpha_gauss,
      dim=(d.nworld,),
      inputs=[
        d.efc.ls_done, d.efc.done, d.efc.lo, d.efc.lo_alpha, d.efc.hi, d.efc.hi_alpha,
        d.efc.quad_gauss, d.efc.hi_alpha, d.efc.quad_gauss
      ],
      outputs=[
        d.efc.lo_next, d.efc.lo_next_alpha, d.efc.hi_next, d.efc.hi_next_alpha,
        d.efc.mid, d.efc.mid_alpha
      ])  # fmt: skip


    if m.opt.cone == types.ConeType.ELLIPTIC:
      wp.launch(
        linesearch_iterative_next_quad_elliptic0,
        dim=(d.njmax,),
        inputs=[
          d.nefc, d.ne, d.nf, d.nl, d.efc.ls_done, d.efc.done, d.efc.Jaref, d.efc.jv,
          d.efc.quad, d.efc.lo_next_alpha, d.efc.hi_next_alpha, d.efc.mid_alpha,
          d.efc.condim, d.efc.worldid
        ],
        outputs=[
          d.efc.lo_next, d.efc.hi_next, d.efc.mid
        ])  # fmt: skip
      wp.launch(
        linesearch_iterative_next_quad_elliptic1,
        dim=(d.nconmax),
        inputs=[
          m.opt.impratio, d.ncon, d.contact.worldid, d.contact.dim, d.contact.efc_address,
          d.contact.friction, d.efc.ls_done, d.efc.done, d.efc.Jaref, d.efc.jv, d.efc.D,
          d.efc.u, d.efc.uu, d.efc.uv, d.efc.vv, d.efc.quad, d.efc.lo_next_alpha,
          d.efc.hi_next_alpha, d.efc.mid_alpha
        ],
        outputs=[
          d.efc.lo_next, d.efc.hi_next, d.efc.mid
        ])  # fmt: skip
  else:
    wp.launch(
      linesearch_iterative_next_quad_pyramidal,
      dim=(d.njmax,),
      inputs=[
        d.nefc, d.ne, d.nf, d.efc.ls_done, d.efc.done, d.efc.Jaref, d.efc.jv, d.efc.quad,
        d.efc.lo_next_alpha, d.efc.hi_next_alpha, d.efc.mid_alpha, d.efc.worldid
      ],
      outputs=[
        d.efc.lo_next, d.efc.hi_next, d.efc.mid
      ])  # fmt: skip

  wp.launch(
    linesearch_iterative_swap,
    dim=(d.nworld,),
    inputs=[
      d.efc.done, d.efc.ls_done, d.efc.gtol, d.efc.lo, d.efc.lo_alpha, d.efc.hi,
      d.efc.hi_alpha, d.efc.lo_next, d.efc.lo_next_alpha, d.efc.hi_next,
      d.efc.hi_next_alpha, d.efc.mid, d.efc.mid_alpha, d.efc.p0
    ],
    outputs=[
      d.efc.ls_done, d.efc.alpha, d.efc.lo, d.efc.lo_alpha, d.efc.hi, d.efc.hi_alpha
    ])  # fmt: skip


def _linesearch_parallel(m: types.Model, d: types.Data):
  ITERATIONS = m.opt.iterations

  @wp.kernel
  def _quad_total(m: types.Model, d: types.Data):
    # TODO(team): static m?
    worldid, alphaid = wp.tid()

    if ITERATIONS > 1:
      if d.efc.done[worldid]:
        return

    d.efc.quad_total_candidate[worldid, alphaid] = d.efc.quad_gauss[worldid]

  @kernel
  def _quad_total_candidate(m: types.Model, d: types.Data):
    # TODO(team): static m?
    efcid, alphaid = wp.tid()

    if efcid >= d.nefc[0]:
      return

    worldid = d.efc.worldid[efcid]

    if ITERATIONS > 1:
      if d.efc.done[worldid]:
        return

    Jaref = d.efc.Jaref[efcid]
    jv = d.efc.jv[efcid]
    quad = d.efc.quad[efcid]

    alpha = alphaid / (m.nlsp - 1)

    if (Jaref + alpha * jv) < 0.0 or (efcid < d.ne[0] + d.nf[0]):
      wp.atomic_add(d.efc.quad_total_candidate[worldid], alphaid, quad)

  @kernel
  def _cost_alpha(m: types.Model, d: types.Data):
    # TODO(team): static m?
    worldid, alphaid = wp.tid()

    if ITERATIONS > 1:
      if d.efc.done[worldid]:
        return

    alpha = alphaid / (m.nlsp - 1)
    alpha_sq = alpha * alpha
    quad_total0 = d.efc.quad_total_candidate[worldid, alphaid][0]
    quad_total1 = d.efc.quad_total_candidate[worldid, alphaid][1]
    quad_total2 = d.efc.quad_total_candidate[worldid, alphaid][2]

    d.efc.cost_candidate[worldid, alphaid] = (
      alpha_sq * quad_total2 + alpha * quad_total1 + quad_total0
    )

  @kernel
  def _best_alpha(d: types.Data):
    worldid = wp.tid()

    if wp.static(m.opt.iterations) > 1:
      if d.efc.done[worldid]:
        return

    # TODO(team): investigate alternatives to wp.argmin
    bestid = wp.argmin(d.efc.cost_candidate[worldid])
    d.efc.alpha[worldid] = bestid / (m.nlsp - 1)

  wp.launch(_quad_total, dim=(d.nworld, m.nlsp), inputs=[m, d])

  wp.launch(_quad_total_candidate, dim=(d.njmax, m.nlsp), inputs=[m, d])

  wp.launch(_cost_alpha, dim=(d.nworld, m.nlsp), inputs=[m, d])
  wp.launch(_best_alpha, dim=(d.nworld), inputs=[d])


@event_scope
def _linesearch(m: types.Model, d: types.Data):
  ITERATIONS = m.opt.iterations
  DSBL_FLOSS = m.opt.disableflags & types.DisableBit.FRICTIONLOSS

  @kernel
  def _zero_jv(d: types.Data):
    efcid = wp.tid()

    if efcid >= d.nefc[0]:
      return

    if wp.static(m.opt.iterations) > 1:
      if d.efc.done[d.efc.worldid[efcid]]:
        return

    d.efc.jv[efcid] = 0.0

  @kernel
  def _jv(d: types.Data):
    efcid, dofid = wp.tid()

    if efcid >= d.nefc[0]:
      return

    worldid = d.efc.worldid[efcid]

    if wp.static(m.opt.iterations) > 1:
      if d.efc.done[worldid]:
        return

    j = d.efc.J[efcid, dofid]
    search = d.efc.search[worldid, dofid]
    wp.atomic_add(d.efc.jv, efcid, j * search)

  @kernel
  def _zero_quad_gauss(d: types.Data):
    worldid = wp.tid()

    if wp.static(m.opt.iterations) > 1:
      if d.efc.done[worldid]:
        return

    d.efc.quad_gauss[worldid] = wp.vec3(0.0)

  @kernel
  def _init_quad_gauss(m: types.Model, d: types.Data):
    # TODO(team): static m?
    worldid, dofid = wp.tid()

    if ITERATIONS > 1:
      if d.efc.done[worldid]:
        return

    search = d.efc.search[worldid, dofid]
    quad_gauss = wp.vec3()
    quad_gauss[0] = d.efc.gauss[worldid] / float(m.nv)
    quad_gauss[1] = search * (d.efc.Ma[worldid, dofid] - d.qfrc_smooth[worldid, dofid])
    quad_gauss[2] = 0.5 * search * d.efc.mv[worldid, dofid]
    wp.atomic_add(d.efc.quad_gauss, worldid, quad_gauss)

  @kernel
  def _init_quad(d: types.Data):
    efcid = wp.tid()

    if efcid >= d.nefc[0]:
      return

    worldid = d.efc.worldid[efcid]

    if wp.static(m.opt.iterations) > 1:
      if d.efc.done[worldid]:
        return

    Jaref = d.efc.Jaref[efcid]
    jv = d.efc.jv[efcid]
    efc_D = d.efc.D[efcid]
    floss = d.efc.frictionloss[efcid]

    if floss > 0.0 and wp.static(not DSBL_FLOSS):
      rf = _safe_div(floss, efc_D)
      if Jaref <= -rf:
        d.efc.quad[efcid] = wp.vec3(floss * (-0.5 * rf - Jaref), -floss * jv, 0.0)
        return
      elif Jaref >= rf:
        d.efc.quad[efcid] = wp.vec3(floss * (-0.5 * rf + Jaref), floss * jv, 0.0)
        return

    d.efc.quad[efcid] = wp.vec3(
      0.5 * Jaref * Jaref * efc_D, jv * Jaref * efc_D, 0.5 * jv * jv * efc_D
    )

  if m.opt.cone == types.ConeType.ELLIPTIC:

    @kernel
    def _quad_elliptic(d: types.Data):
      conid, dimid = wp.tid()
      dimid += 1

      if conid >= d.ncon[0]:
        return

      if d.efc.done[d.contact.worldid[conid]]:
        return

      condim = d.contact.dim[conid]

      if condim == 1 or (dimid >= condim):
        return

      efcid0 = d.contact.efc_address[conid, 0]
      efcid = d.contact.efc_address[conid, dimid]

      # complete vector quadratic (for bottom zone)
      wp.atomic_add(d.efc.quad, efcid0, d.efc.quad[efcid])

      # rescale to make primal cone circular
      u = d.efc.u[conid, dimid]
      v = d.efc.jv[efcid] * d.contact.friction[conid][dimid - 1]
      wp.atomic_add(d.efc.uv, conid, u * v)
      wp.atomic_add(d.efc.vv, conid, v * v)

  @kernel
  def _qacc_ma(d: types.Data):
    worldid, dofid = wp.tid()

    if wp.static(m.opt.iterations) > 1:
      if d.efc.done[worldid]:
        return

    alpha = d.efc.alpha[worldid]
    d.qacc[worldid, dofid] += alpha * d.efc.search[worldid, dofid]
    d.efc.Ma[worldid, dofid] += alpha * d.efc.mv[worldid, dofid]

  @kernel
  def _jaref(d: types.Data):
    efcid = wp.tid()

    if efcid >= d.nefc[0]:
      return

    worldid = d.efc.worldid[efcid]

    if wp.static(m.opt.iterations) > 1:
      if d.efc.done[worldid]:
        return

    d.efc.Jaref[efcid] += d.efc.alpha[worldid] * d.efc.jv[efcid]

  # mv = qM @ search
  support.mul_m(m, d, d.efc.mv, d.efc.search, d.efc.done)

  # jv = efc_J @ search
  # TODO(team): is there a better way of doing batched matmuls with dynamic array sizes?
  wp.launch(_zero_jv, dim=(d.njmax), inputs=[d])

  wp.launch(_jv, dim=(d.njmax, m.nv), inputs=[d])

  # prepare quadratics
  # quad_gauss = [gauss, search.T @ Ma - search.T @ qfrc_smooth, 0.5 * search.T @ mv]
  wp.launch(_zero_quad_gauss, dim=(d.nworld), inputs=[d])

  wp.launch(_init_quad_gauss, dim=(d.nworld, m.nv), inputs=[m, d])

  # quad = [0.5 * Jaref * Jaref * efc_D, jv * Jaref * efc_D, 0.5 * jv * jv * efc_D]

  wp.launch(_init_quad, dim=(d.njmax), inputs=[d])

  if m.opt.cone == types.ConeType.ELLIPTIC:
    d.efc.uv.zero_()
    d.efc.vv.zero_()
    wp.launch(_quad_elliptic, dim=(d.nconmax, m.condim_max - 1), inputs=[d])
  if m.opt.ls_parallel:
    _linesearch_parallel(m, d)
  else:
    _linesearch_iterative(m, d)

  wp.launch(_qacc_ma, dim=(d.nworld, m.nv), inputs=[d])

  wp.launch(_jaref, dim=(d.njmax,), inputs=[d])


def _create_context(m: types.Model, d: types.Data, grad: bool = True):
  @kernel
  def _init_context(d: types.Data):
    worldid = wp.tid()
    d.efc.cost[worldid] = wp.inf
    d.efc.solver_niter[worldid] = 0
    d.efc.done[worldid] = False
    if grad:
      d.efc.search_dot[worldid] = 0.0

  @kernel
  def _jaref(m: types.Model, d: types.Data):
    efcid, dofid = wp.tid()

    if efcid >= d.nefc[0]:
      return

    worldid = d.efc.worldid[efcid]
    wp.atomic_add(
      d.efc.Jaref,
      efcid,
      d.efc.J[efcid, dofid] * d.qacc[worldid, dofid] - d.efc.aref[efcid] / float(m.nv),
    )

  @kernel
  def _search(d: types.Data):
    worldid, dofid = wp.tid()
    search = -1.0 * d.efc.Mgrad[worldid, dofid]
    d.efc.search[worldid, dofid] = search
    wp.atomic_add(d.efc.search_dot, worldid, search * search)

  wp.launch(_init_context, dim=(d.nworld), inputs=[d])

  # jaref = d.efc_J @ d.qacc - d.efc_aref
  d.efc.Jaref.zero_()

  wp.launch(_jaref, dim=(d.njmax, m.nv), inputs=[m, d])

  # Ma = qM @ qacc
  support.mul_m(m, d, d.efc.Ma, d.qacc, d.efc.done)

  _update_constraint(m, d)
  if grad:
    _update_gradient(m, d)

    # search = -Mgrad
    wp.launch(_search, dim=(d.nworld, m.nv), inputs=[d])


def _update_constraint(m: types.Model, d: types.Data):
  DSBL_FLOSS = m.opt.disableflags & types.DisableBit.FRICTIONLOSS

  @kernel
  def _init_cost(d: types.Data):
    worldid = wp.tid()

    if wp.static(m.opt.iterations) > 1:
      if d.efc.done[worldid]:
        return

    d.efc.prev_cost[worldid] = d.efc.cost[worldid]
    d.efc.cost[worldid] = 0.0
    d.efc.gauss[worldid] = 0.0

  if m.opt.cone == types.ConeType.PYRAMIDAL:

    @kernel
    def _efc_pyramidal(d: types.Data):
      efcid = wp.tid()

      if efcid >= d.nefc[0]:
        return

      worldid = d.efc.worldid[efcid]
      efc_D = d.efc.D[efcid]
      Jaref = d.efc.Jaref[efcid]

      cost = 0.5 * efc_D * Jaref * Jaref
      efc_force = -efc_D * Jaref

      ne = d.ne[0]
      nf = d.nf[0]

      if efcid < ne:
        # equality
        pass
      elif efcid < ne + nf and wp.static(not DSBL_FLOSS):
        # friction
        f = d.efc.frictionloss[efcid]
        if f > 0.0:
          rf = _safe_div(f, efc_D)
          if Jaref <= -rf:
            d.efc.force[efcid] = f
            d.efc.active[efcid] = False
            wp.atomic_add(d.efc.cost, worldid, -0.5 * rf - Jaref)
            return
          elif Jaref >= rf:
            d.efc.force[efcid] = -f
            d.efc.active[efcid] = False
            wp.atomic_add(d.efc.cost, worldid, -0.5 * rf + Jaref)
            return
      else:
        # limit, contact
        if Jaref >= 0.0:
          d.efc.force[efcid] = 0.0
          d.efc.active[efcid] = False
          return

      d.efc.force[efcid] = efc_force
      d.efc.active[efcid] = True
      wp.atomic_add(d.efc.cost, worldid, cost)

  elif m.opt.cone == types.ConeType.ELLIPTIC:

    @kernel
    def _u_elliptic(d: types.Data):
      conid, dimid = wp.tid()

      if conid >= d.ncon[0]:
        return

      if d.efc.done[d.contact.worldid[conid]]:
        return

      efcid = d.contact.efc_address[conid, dimid]

      condim = d.contact.dim[conid]
      d.efc.condim[efcid] = condim

      if condim == 1:
        return

      if dimid < condim:
        if dimid == 0:
          fri = d.contact.friction[conid][0] * wp.static(1.0 / m.opt.impratio)
        else:
          fri = d.contact.friction[conid][dimid - 1]
        u = d.efc.Jaref[efcid] * fri
        d.efc.u[conid, dimid] = u
        if dimid > 0:
          wp.atomic_add(d.efc.uu, conid, u * u)

    @kernel
    def _active_bottom_zone(d: types.Data):
      conid, dimid = wp.tid()

      if conid >= d.ncon[0]:
        return

      if d.efc.done[d.contact.worldid[conid]]:
        return

      condim = d.contact.dim[conid]
      if condim == 1:
        return

      mu = d.contact.friction[conid][0] * wp.static(1.0 / m.opt.impratio)
      n = d.efc.u[conid, 0]
      tt = d.efc.uu[conid]
      if tt <= 0.0:
        t = 0.0
      else:
        t = wp.sqrt(tt)

      # bottom zone: quadratic
      bottom_zone = ((t <= 0.0) and (n < 0.0)) or ((t > 0.0) and ((mu * n + t) <= 0.0))

      # update active
      efcid = d.contact.efc_address[conid, dimid]
      d.efc.active[efcid] = bottom_zone

    @kernel
    def _efc_elliptic0(d: types.Data):
      efcid = wp.tid()

      if efcid >= d.nefc[0]:
        return

      if d.efc.done[d.efc.worldid[efcid]]:
        return

      worldid = d.efc.worldid[efcid]
      efc_D = d.efc.D[efcid]
      Jaref = d.efc.Jaref[efcid]

      ne = d.ne[0]
      nf = d.nf[0]
      nl = d.nl[0]

      if efcid < ne:
        # equality
        d.efc.active[efcid] = True
      elif efcid < ne + nf and wp.static(not DSBL_FLOSS):
        # friction
        f = d.efc.frictionloss[efcid]
        if f > 0.0:
          rf = _safe_div(f, efc_D)
          if Jaref <= -rf:
            d.efc.force[efcid] = f
            d.efc.active[efcid] = False
            wp.atomic_add(d.efc.cost, worldid, -0.5 * rf - Jaref)
            return
          elif Jaref >= rf:
            d.efc.force[efcid] = -f
            d.efc.active[efcid] = False
            wp.atomic_add(d.efc.cost, worldid, -0.5 * rf + Jaref)
            return
      elif efcid < ne + nf + nl:
        # limits
        if Jaref < 0.0:
          d.efc.active[efcid] = True
        else:
          d.efc.force[efcid] = 0.0
          d.efc.active[efcid] = False
          return
      else:
        # contact
        if not d.efc.active[efcid]:  # calculated by _active_bottom_zone
          d.efc.force[efcid] = 0.0
          return

      d.efc.force[efcid] = -efc_D * Jaref
      wp.atomic_add(d.efc.cost, worldid, 0.5 * efc_D * Jaref * Jaref)

    @kernel
    def _efc_elliptic1(d: types.Data):
      conid, dimid = wp.tid()

      if conid >= d.ncon[0]:
        return

      if d.efc.done[d.contact.worldid[conid]]:
        return

      condim = d.contact.dim[conid]

      if condim == 1 or dimid >= condim:
        return

      friction = d.contact.friction[conid]
      efcid = d.contact.efc_address[conid, dimid]

      mu = d.contact.friction[conid][0] * wp.static(1.0 / m.opt.impratio)
      n = d.efc.u[conid, 0]
      tt = d.efc.uu[conid]
      if tt <= 0.0:
        t = 0.0
      else:
        t = wp.sqrt(tt)

      # middle zone: cone
      middle_zone = (t > 0.0) and (n < (mu * t)) and ((mu * n + t) > 0.0)

      # tangent and friction for middle zone:
      if middle_zone:
        mu2 = mu * mu
        dm = d.efc.D[d.contact.efc_address[conid, 0]] / wp.max(
          mu2 * float(1.0 + mu2), types.MJ_MINVAL
        )

        nmt = n - mu * t

        force = -dm * nmt * mu
        if dimid > 0:
          force_fri = -force / t
          force_fri *= d.efc.u[conid, dimid] * friction[dimid - 1]
          d.efc.force[efcid] += force_fri
        else:
          d.efc.force[efcid] += force

          worldid = d.contact.worldid[conid]
          wp.atomic_add(d.efc.cost, worldid, 0.5 * dm * nmt * nmt)

  @kernel
  def _zero_qfrc_constraint(d: types.Data):
    worldid, dofid = wp.tid()

    if wp.static(m.opt.iterations) > 1:
      if d.efc.done[worldid]:
        return

    d.qfrc_constraint[worldid, dofid] = 0.0

  @kernel
  def _qfrc_constraint(d: types.Data):
    dofid, efcid = wp.tid()

    if efcid >= d.nefc[0]:
      return

    worldid = d.efc.worldid[efcid]

    if wp.static(m.opt.iterations) > 1:
      if d.efc.done[worldid]:
        return

    wp.atomic_add(
      d.qfrc_constraint[worldid],
      dofid,
      d.efc.J[efcid, dofid] * d.efc.force[efcid],
    )

  @kernel
  def _gauss(d: types.Data):
    worldid, dofid = wp.tid()

    if wp.static(m.opt.iterations) > 1:
      if d.efc.done[worldid]:
        return

    gauss_cost = (
      0.5
      * (d.efc.Ma[worldid, dofid] - d.qfrc_smooth[worldid, dofid])
      * (d.qacc[worldid, dofid] - d.qacc_smooth[worldid, dofid])
    )
    wp.atomic_add(d.efc.gauss, worldid, gauss_cost)
    wp.atomic_add(d.efc.cost, worldid, gauss_cost)

  wp.launch(_init_cost, dim=(d.nworld), inputs=[d])

  if m.opt.cone == types.ConeType.ELLIPTIC:
    d.efc.uu.zero_()
    d.efc.active.zero_()
    d.efc.condim.fill_(-1)
    wp.launch(_u_elliptic, dim=(d.nconmax, m.condim_max), inputs=[d])
    wp.launch(_active_bottom_zone, dim=(d.nconmax, m.condim_max), inputs=[d])
    wp.launch(_efc_elliptic0, dim=(d.njmax), inputs=[d])
    wp.launch(_efc_elliptic1, dim=(d.nconmax, m.condim_max), inputs=[d])
  else:
    wp.launch(_efc_pyramidal, dim=(d.njmax,), inputs=[d])

  # qfrc_constraint = efc_J.T @ efc_force
  wp.launch(_zero_qfrc_constraint, dim=(d.nworld, m.nv), inputs=[d])

  wp.launch(_qfrc_constraint, dim=(m.nv, d.njmax), inputs=[d])

  # gauss = 0.5 * (Ma - qfrc_smooth).T @ (qacc - qacc_smooth)

  wp.launch(_gauss, dim=(d.nworld, m.nv), inputs=[d])


def _update_gradient(m: types.Model, d: types.Data):
  TILE = m.nv
  ITERATIONS = m.opt.iterations

  @kernel
  def _zero_grad_dot(d: types.Data):
    worldid = wp.tid()

    if wp.static(m.opt.iterations) > 1:
      if d.efc.done[worldid]:
        return

    d.efc.grad_dot[worldid] = 0.0

  @kernel
  def _grad(d: types.Data):
    worldid, dofid = wp.tid()

    if wp.static(m.opt.iterations) > 1:
      if d.efc.done[worldid]:
        return

    grad = (
      d.efc.Ma[worldid, dofid]
      - d.qfrc_smooth[worldid, dofid]
      - d.qfrc_constraint[worldid, dofid]
    )
    d.efc.grad[worldid, dofid] = grad
    wp.atomic_add(d.efc.grad_dot, worldid, grad * grad)

  if m.opt.is_sparse:

    @kernel
    def _zero_h_lower(m: types.Model, d: types.Data):
      # TODO(team): static m?
      worldid, elementid = wp.tid()

      if ITERATIONS > 1:
        if d.efc.done[worldid]:
          return

      rowid = m.dof_tri_row[elementid]
      colid = m.dof_tri_col[elementid]
      d.efc.h[worldid, rowid, colid] = 0.0

    @kernel
    def _set_h_qM_lower_sparse(m: types.Model, d: types.Data):
      # TODO(team): static m?
      worldid, elementid = wp.tid()

      if ITERATIONS > 1:
        if d.efc.done[worldid]:
          return

      i = m.qM_fullm_i[elementid]
      j = m.qM_fullm_j[elementid]
      d.efc.h[worldid, i, j] = d.qM[worldid, 0, elementid]

  else:

    @kernel
    def _copy_lower_triangle(m: types.Model, d: types.Data):
      # TODO(team): static m?
      worldid, elementid = wp.tid()

      if ITERATIONS > 1:
        if d.efc.done[worldid]:
          return

      rowid = m.dof_tri_row[elementid]
      colid = m.dof_tri_col[elementid]
      d.efc.h[worldid, rowid, colid] = d.qM[worldid, rowid, colid]

  # Optimization: launching _JTDAJ with limited number of blocks on a GPU.
  # Profiling suggests that only a fraction of blocks out of the original
  # d.njmax blocks do the actual work. It aims to minimize #CTAs with no
  # effective work. It launches with #blocks that's proportional to the number
  # of SMs on the GPU. We can now query the SM count:
  # https://github.com/NVIDIA/warp/commit/f3814e7e5459e5fd13032cf0fddb3daddd510f30

  # make dim_x and nblocks_perblock static arguments for _JTDAJ to allow unrolling the loop
  if wp.get_device().is_cuda:
    sm_count = wp.get_device().sm_count

    # Here we assume one block has 256 threads. We use a factor of 6, which
    # can be change in future to fine-tune the perf. The optimal factor will
    # depend on the kernel's occupancy, which determines how many blocks can
    # simultaneously run on the SM. TODO: This factor can be tuned further.
    dim_x = int((sm_count * 6 * 256) / m.dof_tri_row.size)
    dim_y = dim_x
  else:
    # fall back for CPU
    dim_x = d.njmax
    dim_y = d.nconmax

  nblocks_perblock = int((d.njmax + dim_x - 1) / dim_x)

  @kernel
  def _JTDAJ(m: types.Model, d: types.Data):
    # TODO(team): static m?
    efcid_temp, elementid = wp.tid()

    nefc = d.nefc[0]

    for i in range(nblocks_perblock):
      efcid = efcid_temp + i * dim_x

      if efcid >= nefc:
        return

      worldid = d.efc.worldid[efcid]

    efc_D = d.efc.D[efcid]
    active = d.efc.active[efcid]
    if efc_D == 0.0 or not active:
      return

    efc_D = d.efc.D[efcid]
    active = d.efc.active[efcid]

    if efc_D * float(active) == 0.0:
      return

    dofi = m.dof_tri_row[elementid]
    dofj = m.dof_tri_col[elementid]

    # TODO(team): sparse efc_J
    value = d.efc.J[efcid, dofi] * d.efc.J[efcid, dofj] * efc_D
    if value != 0.0:
      wp.atomic_add(d.efc.h[worldid, dofi], dofj, value)

  if m.opt.cone == types.ConeType.ELLIPTIC:
    # TODO(team): combine with _JTDAJ
    @kernel
    def _JTCJ(
      m: types.Model,
      d: types.Data,
      nblocks_perblock: int,
      dim_y: int,
      impratio: wp.float32,
    ):
      conid_tmp, elementid, dim1id, dim2id = wp.tid()

      # TODO(team): cone hessian upper/lower triangle
      for i in range(nblocks_perblock):
        conid = conid_tmp + i * dim_y

        if conid >= d.ncon[0]:
          return

        if d.efc.done[d.contact.worldid[conid]]:
          return

        condim = d.contact.dim[conid]

        if condim == 1:
          return

        if (dim1id >= condim) or (dim2id >= condim):
          return

        mu = d.contact.friction[conid][0] / impratio
        n = d.efc.u[conid, 0]
        tt = d.efc.uu[conid]
        if tt <= 0.0:
          t = 0.0
        else:
          t = wp.sqrt(tt)

        middle_zone = (t > 0) and (n < (mu * t)) and ((mu * n + t) > 0.0)

        if not middle_zone:
          return

        worldid = d.contact.worldid[conid]

        dof1id = m.dof_tri_row[elementid]
        dof2id = m.dof_tri_col[elementid]

        efc1id = d.contact.efc_address[conid, dim1id]
        efc2id = d.contact.efc_address[conid, dim2id]

        t = wp.max(t, types.MJ_MINVAL)
        ttt = wp.max(t * t * t, types.MJ_MINVAL)

        ui = d.efc.u[conid, dim1id]
        uj = d.efc.u[conid, dim2id]

        # set first row/column: (1, -mu/t * u)
        if dim1id == 0 and dim2id == 0:
          hcone = 1.0
        elif dim1id == 0:
          hcone = -mu / t * uj
        elif dim2id == 0:
          hcone = -mu / t * ui
        else:
          hcone = mu * n / ttt * ui * uj

          # add to diagonal: mu^2 - mu * n / t
          if dim1id == dim2id:
            hcone += mu * mu - mu * n / t

        # pre and post multiply by diag(mu, friction) scale by dm
        if dim1id == 0:
          fri1 = mu
        else:
          fri1 = d.contact.friction[conid][dim1id - 1]

        if dim2id == 0:
          fri2 = mu
        else:
          fri2 = d.contact.friction[conid][dim2id - 1]

        mu2 = mu * mu
        dm = d.efc.D[d.contact.efc_address[conid, 0]] / wp.max(
          mu2 * (1.0 + mu2), types.MJ_MINVAL
        )

        hcone *= dm * fri1 * fri2

        if hcone:
          wp.atomic_add(
            d.efc.h[worldid, dof1id],
            dof2id,
            d.efc.J[efc1id, dof1id] * d.efc.J[efc2id, dof2id] * hcone,
          )

  @kernel
  def _cholesky(d: types.Data):
    worldid = wp.tid()

    if wp.static(m.opt.iterations) > 1:
      if d.efc.done[worldid]:
        return

    mat_tile = wp.tile_load(d.efc.h[worldid], shape=(TILE, TILE))
    fact_tile = wp.tile_cholesky(mat_tile)
    input_tile = wp.tile_load(d.efc.grad[worldid], shape=TILE)
    output_tile = wp.tile_cholesky_solve(fact_tile, input_tile)
    wp.tile_store(d.efc.Mgrad[worldid], output_tile)

  # grad = Ma - qfrc_smooth - qfrc_constraint
  wp.launch(_zero_grad_dot, dim=(d.nworld), inputs=[d])

  wp.launch(_grad, dim=(d.nworld, m.nv), inputs=[d])

  if m.opt.solver == types.SolverType.CG:
    smooth.solve_m(m, d, d.efc.Mgrad, d.efc.grad)
  elif m.opt.solver == types.SolverType.NEWTON:
    # h = qM + (efc_J.T * efc_D * active) @ efc_J
    if m.opt.is_sparse:
      wp.launch(_zero_h_lower, dim=(d.nworld, m.dof_tri_row.size), inputs=[m, d])

      wp.launch(
        _set_h_qM_lower_sparse, dim=(d.nworld, m.qM_fullm_i.size), inputs=[m, d]
      )
    else:
      wp.launch(_copy_lower_triangle, dim=(d.nworld, m.dof_tri_row.size), inputs=[m, d])

    wp.launch(
      _JTDAJ,
      dim=(dim_x, m.dof_tri_row.size),
      inputs=[m, d],
    )

    if m.opt.cone == types.ConeType.ELLIPTIC:
      # TODO(team): optimize launch
      wp.launch(
        _JTCJ,
        dim=(dim_y, m.dof_tri_row.size, m.condim_max, m.condim_max),
        inputs=[m, d, int((d.nconmax + dim_y - 1) / dim_y), dim_y, m.opt.impratio],
      )

    wp.launch_tiled(_cholesky, dim=(d.nworld,), inputs=[d], block_dim=32)


@event_scope
def solve(m: types.Model, d: types.Data):
  """Finds forces that satisfy constraints."""
  ITERATIONS = m.opt.iterations

  @kernel
  def _zero_search_dot(d: types.Data):
    worldid = wp.tid()

    if wp.static(m.opt.iterations) > 1:
      if d.efc.done[worldid]:
        return

    d.efc.search_dot[worldid] = 0.0

  @kernel
  def _search_update(d: types.Data):
    worldid, dofid = wp.tid()

    if wp.static(m.opt.iterations) > 1:
      if d.efc.done[worldid]:
        return

    search = -1.0 * d.efc.Mgrad[worldid, dofid]

    if wp.static(m.opt.solver == types.SolverType.CG):
      search += d.efc.beta[worldid] * d.efc.search[worldid, dofid]

    d.efc.search[worldid, dofid] = search
    wp.atomic_add(d.efc.search_dot, worldid, search * search)

  @kernel
  def _done(m: types.Model, d: types.Data, solver_niter: int):
    # TODO(team): static m?
    worldid = wp.tid()

    if ITERATIONS > 1:
      if d.efc.done[worldid]:
        return

    improvement = _rescale(m, d.efc.prev_cost[worldid] - d.efc.cost[worldid])
    gradient = _rescale(m, wp.math.sqrt(d.efc.grad_dot[worldid]))
    d.efc.done[worldid] = (improvement < m.opt.tolerance) or (
      gradient < m.opt.tolerance
    )

  if m.opt.solver == types.SolverType.CG:

    @kernel
    def _prev_grad_Mgrad(d: types.Data):
      worldid, dofid = wp.tid()

      if wp.static(m.opt.iterations) > 1:
        if d.efc.done[worldid]:
          return

      d.efc.prev_grad[worldid, dofid] = d.efc.grad[worldid, dofid]
      d.efc.prev_Mgrad[worldid, dofid] = d.efc.Mgrad[worldid, dofid]

    @kernel
    def _zero_beta_num_den(d: types.Data):
      worldid = wp.tid()

      if wp.static(m.opt.iterations) > 1:
        if d.efc.done[worldid]:
          return

      d.efc.beta_num[worldid] = 0.0
      d.efc.beta_den[worldid] = 0.0

    @kernel
    def _beta_num_den(d: types.Data):
      worldid, dofid = wp.tid()

      if wp.static(m.opt.iterations) > 1:
        if d.efc.done[worldid]:
          return

      prev_Mgrad = d.efc.prev_Mgrad[worldid][dofid]
      wp.atomic_add(
        d.efc.beta_num,
        worldid,
        d.efc.grad[worldid, dofid] * (d.efc.Mgrad[worldid, dofid] - prev_Mgrad),
      )
      wp.atomic_add(
        d.efc.beta_den, worldid, d.efc.prev_grad[worldid, dofid] * prev_Mgrad
      )

    @kernel
    def _beta(d: types.Data):
      worldid = wp.tid()

      if wp.static(m.opt.iterations) > 1:
        if d.efc.done[worldid]:
          return

      d.efc.beta[worldid] = wp.max(
        0.0, d.efc.beta_num[worldid] / wp.max(types.MJ_MINVAL, d.efc.beta_den[worldid])
      )

  # warmstart
  wp.copy(d.qacc, d.qacc_warmstart)

  _create_context(m, d, grad=True)

  for i in range(m.opt.iterations):
    _linesearch(m, d)

    if m.opt.solver == types.SolverType.CG:
      wp.launch(_prev_grad_Mgrad, dim=(d.nworld, m.nv), inputs=[d])

    _update_constraint(m, d)
    _update_gradient(m, d)

    # polak-ribiere
    if m.opt.solver == types.SolverType.CG:
      wp.launch(_zero_beta_num_den, dim=(d.nworld), inputs=[d])

      wp.launch(_beta_num_den, dim=(d.nworld, m.nv), inputs=[d])

      wp.launch(_beta, dim=(d.nworld,), inputs=[d])

    wp.launch(_zero_search_dot, dim=(d.nworld), inputs=[d])

    wp.launch(_search_update, dim=(d.nworld, m.nv), inputs=[d])

    wp.launch(_done, dim=(d.nworld,), inputs=[m, d, i])

  wp.copy(d.qacc_warmstart, d.qacc)

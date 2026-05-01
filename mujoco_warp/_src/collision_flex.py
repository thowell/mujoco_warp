# Copyright 2026 The Newton Developers
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
"""Flex collision detection (geom vs flex triangles)."""

import warp as wp

from mujoco_warp._src import collision_primitive_core
from mujoco_warp._src.math import make_frame
from mujoco_warp._src.types import MJ_MAXCONPAIR
from mujoco_warp._src.types import MJ_MAXVAL
from mujoco_warp._src.types import MJ_MINMU
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import vec5
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False})


# TODO(team): generalize into a shared contact parameter mixing function
#   (mj_contactParam) that works for both geom-geom and geom-flex contacts.
@wp.func
def _mix_flex_contact_params(
  # In:
  a_condim: int,
  a_priority: int,
  a_solmix: float,
  a_solref: wp.vec2,
  a_solimp: vec5,
  a_friction: wp.vec3,
  a_gap: float,
  b_condim: int,
  b_priority: int,
  b_solmix: float,
  b_solref: wp.vec2,
  b_solimp: vec5,
  b_friction: wp.vec3,
  b_gap: float,
):
  """Mix contact parameters between geom and flex, matching mj_contactParam."""
  gap = a_gap + b_gap

  if a_priority > b_priority:
    condim = a_condim
    solref = a_solref
    solimp = a_solimp
    fri = a_friction
  elif a_priority < b_priority:
    condim = b_condim
    solref = b_solref
    solimp = b_solimp
    fri = b_friction
  else:
    # same priority
    condim = wp.max(a_condim, b_condim)

    # compute solver mix factor
    if a_solmix >= MJ_MINVAL and b_solmix >= MJ_MINVAL:
      mix = a_solmix / (a_solmix + b_solmix)
    elif a_solmix < MJ_MINVAL and b_solmix < MJ_MINVAL:
      mix = 0.5
    elif a_solmix < MJ_MINVAL:
      mix = 0.0
    else:
      mix = 1.0

    # solref: mix if both standard, min if either direct
    if a_solref[0] > 0.0 and b_solref[0] > 0.0:
      solref = wp.vec2(
        mix * a_solref[0] + (1.0 - mix) * b_solref[0],
        mix * a_solref[1] + (1.0 - mix) * b_solref[1],
      )
    else:
      solref = wp.vec2(
        wp.min(a_solref[0], b_solref[0]),
        wp.min(a_solref[1], b_solref[1]),
      )

    # solimp: mix
    solimp = vec5(
      mix * a_solimp[0] + (1.0 - mix) * b_solimp[0],
      mix * a_solimp[1] + (1.0 - mix) * b_solimp[1],
      mix * a_solimp[2] + (1.0 - mix) * b_solimp[2],
      mix * a_solimp[3] + (1.0 - mix) * b_solimp[3],
      mix * a_solimp[4] + (1.0 - mix) * b_solimp[4],
    )

    # friction: max
    fri = wp.vec3(
      wp.max(a_friction[0], b_friction[0]),
      wp.max(a_friction[1], b_friction[1]),
      wp.max(a_friction[2], b_friction[2]),
    )

  # unpack 5D friction with MJ_MINMU floor
  friction = vec5(
    wp.max(MJ_MINMU, fri[0]),
    wp.max(MJ_MINMU, fri[0]),
    wp.max(MJ_MINMU, fri[1]),
    wp.max(MJ_MINMU, fri[2]),
    wp.max(MJ_MINMU, fri[2]),
  )

  return condim, gap, solref, solimp, friction


@wp.func
def _write_flex_contact(
  # Data in:
  naconmax_in: int,
  # In:
  dist: float,
  pos: wp.vec3,
  frame: wp.mat33,
  margin: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2,
  solimp: vec5,
  geom: int,
  flexid: int,
  vertid: int,
  elemid: int,
  worldid: int,
  # Data out:
  contact_dist_out: wp.array[float],
  contact_pos_out: wp.array[wp.vec3],
  contact_frame_out: wp.array[wp.mat33],
  contact_includemargin_out: wp.array[float],
  contact_friction_out: wp.array[vec5],
  contact_solref_out: wp.array[wp.vec2],
  contact_solreffriction_out: wp.array[wp.vec2],
  contact_solimp_out: wp.array[vec5],
  contact_dim_out: wp.array[int],
  contact_geom_out: wp.array[wp.vec2i],
  contact_flex_out: wp.array[wp.vec2i],
  contact_vert_out: wp.array[wp.vec2i],
  contact_elem_out: wp.array[wp.vec2i],
  contact_worldid_out: wp.array[int],
  contact_type_out: wp.array[int],
  contact_geomcollisionid_out: wp.array[int],
  nacon_out: wp.array[int],
):
  if dist >= margin or dist >= MJ_MAXVAL:
    return

  id_ = wp.atomic_add(nacon_out, 0, 1)
  if id_ >= naconmax_in:
    return

  contact_dist_out[id_] = dist
  contact_pos_out[id_] = pos
  contact_frame_out[id_] = frame
  contact_includemargin_out[id_] = margin
  contact_friction_out[id_] = friction
  contact_solref_out[id_] = solref
  contact_solreffriction_out[id_] = wp.vec2(0.0, 0.0)
  contact_solimp_out[id_] = solimp
  contact_dim_out[id_] = condim
  contact_geom_out[id_] = wp.vec2i(geom, -1)
  contact_flex_out[id_] = wp.vec2i(-1, flexid)
  contact_vert_out[id_] = wp.vec2i(-1, vertid)
  contact_elem_out[id_] = wp.vec2i(-1, elemid)
  contact_worldid_out[id_] = worldid
  contact_type_out[id_] = 1
  contact_geomcollisionid_out[id_] = 0


@wp.func
def _collide_geom_triangle(
  # Data in:
  naconmax_in: int,
  # In:
  gtype: int,
  pos: wp.vec3,
  rot: wp.mat33,
  size_val: wp.vec3,
  t1: wp.vec3,
  t2: wp.vec3,
  t3: wp.vec3,
  tri_radius: float,
  margin: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2,
  solimp: vec5,
  geomid: int,
  flexid: int,
  vertex_id: int,
  elemid: int,
  worldid: int,
  # Data out:
  contact_dist_out: wp.array[float],
  contact_pos_out: wp.array[wp.vec3],
  contact_frame_out: wp.array[wp.mat33],
  contact_includemargin_out: wp.array[float],
  contact_friction_out: wp.array[vec5],
  contact_solref_out: wp.array[wp.vec2],
  contact_solreffriction_out: wp.array[wp.vec2],
  contact_solimp_out: wp.array[vec5],
  contact_dim_out: wp.array[int],
  contact_geom_out: wp.array[wp.vec2i],
  contact_flex_out: wp.array[wp.vec2i],
  contact_vert_out: wp.array[wp.vec2i],
  contact_elem_out: wp.array[wp.vec2i],
  contact_worldid_out: wp.array[int],
  contact_type_out: wp.array[int],
  contact_geomcollisionid_out: wp.array[int],
  nacon_out: wp.array[int],
):
  if gtype == int(GeomType.SPHERE):
    sphere_radius = size_val[0]
    dist, contact_pos, nrm = collision_primitive_core.sphere_triangle(pos, sphere_radius, t1, t2, t3, tri_radius)
    if dist < margin:
      _write_flex_contact(
        naconmax_in,
        dist,
        contact_pos,
        make_frame(nrm),
        margin,
        condim,
        friction,
        solref,
        solimp,
        geomid,
        flexid,
        vertex_id,
        elemid,
        worldid,
        contact_dist_out,
        contact_pos_out,
        contact_frame_out,
        contact_includemargin_out,
        contact_friction_out,
        contact_solref_out,
        contact_solreffriction_out,
        contact_solimp_out,
        contact_dim_out,
        contact_geom_out,
        contact_flex_out,
        contact_vert_out,
        contact_elem_out,
        contact_worldid_out,
        contact_type_out,
        contact_geomcollisionid_out,
        nacon_out,
      )
    return

  # Capsule, box, cylinder all return up to 2 contacts - compute then share writing code
  dists = wp.vec2(collision_primitive_core.MJ_MAXVAL, collision_primitive_core.MJ_MAXVAL)
  poss = collision_primitive_core.mat23f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  nrms = collision_primitive_core.mat23f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

  if gtype == int(GeomType.CAPSULE):
    cap_radius = size_val[0]
    cap_half_len = size_val[1]
    cap_axis = wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])
    dists, poss, nrms = collision_primitive_core.capsule_triangle(
      pos, cap_axis, cap_radius, cap_half_len, t1, t2, t3, tri_radius
    )
  elif gtype == int(GeomType.BOX):
    dists, poss, nrms = collision_primitive_core.box_triangle(pos, rot, size_val, t1, t2, t3, tri_radius)
  elif gtype == int(GeomType.CYLINDER):
    cyl_radius = size_val[0]
    cyl_half_height = size_val[1]
    cyl_axis = wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])
    dists, poss, nrms = collision_primitive_core.cylinder_triangle(
      pos, cyl_axis, cyl_radius, cyl_half_height, t1, t2, t3, tri_radius
    )

  # Write up to 2 contacts (shared code for capsule/box/cylinder)
  if dists[0] < margin:
    p1 = wp.vec3(poss[0, 0], poss[0, 1], poss[0, 2])
    n1 = wp.vec3(nrms[0, 0], nrms[0, 1], nrms[0, 2])
    _write_flex_contact(
      naconmax_in,
      dists[0],
      p1,
      make_frame(n1),
      margin,
      condim,
      friction,
      solref,
      solimp,
      geomid,
      flexid,
      vertex_id,
      elemid,
      worldid,
      contact_dist_out,
      contact_pos_out,
      contact_frame_out,
      contact_includemargin_out,
      contact_friction_out,
      contact_solref_out,
      contact_solreffriction_out,
      contact_solimp_out,
      contact_dim_out,
      contact_geom_out,
      contact_flex_out,
      contact_vert_out,
      contact_elem_out,
      contact_worldid_out,
      contact_type_out,
      contact_geomcollisionid_out,
      nacon_out,
    )
  if dists[1] < margin:
    p2 = wp.vec3(poss[1, 0], poss[1, 1], poss[1, 2])
    n2 = wp.vec3(nrms[1, 0], nrms[1, 1], nrms[1, 2])
    _write_flex_contact(
      naconmax_in,
      dists[1],
      p2,
      make_frame(n2),
      margin,
      condim,
      friction,
      solref,
      solimp,
      geomid,
      flexid,
      vertex_id,
      elemid,
      worldid,
      contact_dist_out,
      contact_pos_out,
      contact_frame_out,
      contact_includemargin_out,
      contact_friction_out,
      contact_solref_out,
      contact_solreffriction_out,
      contact_solimp_out,
      contact_dim_out,
      contact_geom_out,
      contact_flex_out,
      contact_vert_out,
      contact_elem_out,
      contact_worldid_out,
      contact_type_out,
      contact_geomcollisionid_out,
      nacon_out,
    )


@wp.kernel
def _flex_plane_narrowphase(
  # Model:
  ngeom: int,
  nflexvert: int,
  geom_type: wp.array[int],
  geom_condim: wp.array[int],
  geom_priority: wp.array[int],
  geom_solmix: wp.array2d[float],
  geom_solref: wp.array2d[wp.vec2],
  geom_solimp: wp.array2d[vec5],
  geom_friction: wp.array2d[wp.vec3],
  geom_margin: wp.array2d[float],
  geom_gap: wp.array2d[float],
  flex_condim: wp.array[int],
  flex_priority: wp.array[int],
  flex_solmix: wp.array[float],
  flex_solref: wp.array[wp.vec2],
  flex_solimp: wp.array[vec5],
  flex_friction: wp.array[wp.vec3],
  flex_margin: wp.array[float],
  flex_gap: wp.array[float],
  flex_vertadr: wp.array[int],
  flex_radius: wp.array[float],
  flex_vertflexid: wp.array[int],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  nworld_in: int,
  naconmax_in: int,
  # Data out:
  contact_dist_out: wp.array[float],
  contact_pos_out: wp.array[wp.vec3],
  contact_frame_out: wp.array[wp.mat33],
  contact_includemargin_out: wp.array[float],
  contact_friction_out: wp.array[vec5],
  contact_solref_out: wp.array[wp.vec2],
  contact_solreffriction_out: wp.array[wp.vec2],
  contact_solimp_out: wp.array[vec5],
  contact_dim_out: wp.array[int],
  contact_geom_out: wp.array[wp.vec2i],
  contact_flex_out: wp.array[wp.vec2i],
  contact_vert_out: wp.array[wp.vec2i],
  contact_elem_out: wp.array[wp.vec2i],
  contact_worldid_out: wp.array[int],
  contact_type_out: wp.array[int],
  contact_geomcollisionid_out: wp.array[int],
  nacon_out: wp.array[int],
):
  worldid, vertid = wp.tid()

  flexid = flex_vertflexid[vertid]
  radius = flex_radius[flexid]
  flex_margin_val = flex_margin[flexid]
  # Convert global vertid to local vertex index within this flex
  local_vertid = vertid - flex_vertadr[flexid]

  vert = flexvert_xpos_in[worldid, vertid]

  # TODO: Add a broadphase
  for geomid in range(ngeom):
    gtype = geom_type[geomid]
    if gtype != int(GeomType.PLANE):
      continue

    plane_pos = geom_xpos_in[worldid, geomid]
    plane_rot = geom_xmat_in[worldid, geomid]
    plane_normal = wp.vec3(plane_rot[0, 2], plane_rot[1, 2], plane_rot[2, 2])

    margin = geom_margin[worldid % geom_margin.shape[0], geomid] + flex_margin_val

    diff = vert - plane_pos
    signed_dist = wp.dot(diff, plane_normal)
    dist = signed_dist - radius

    if dist < margin:
      condim, gap, solref, solimp, friction = _mix_flex_contact_params(
        geom_condim[geomid],
        geom_priority[geomid],
        geom_solmix[worldid % geom_solmix.shape[0], geomid],
        geom_solref[worldid % geom_solref.shape[0], geomid],
        geom_solimp[worldid % geom_solimp.shape[0], geomid],
        geom_friction[worldid % geom_friction.shape[0], geomid],
        geom_gap[worldid % geom_gap.shape[0], geomid],
        flex_condim[flexid],
        flex_priority[flexid],
        flex_solmix[flexid],
        flex_solref[flexid],
        flex_solimp[flexid],
        flex_friction[flexid],
        flex_gap[flexid],
      )

      contact_pos = vert - plane_normal * (dist * 0.5 + radius)
      _write_flex_contact(
        naconmax_in,
        dist,
        contact_pos,
        make_frame(plane_normal),
        margin - gap,
        condim,
        friction,
        solref,
        solimp,
        geomid,
        flexid,
        local_vertid,
        -1,
        worldid,
        contact_dist_out,
        contact_pos_out,
        contact_frame_out,
        contact_includemargin_out,
        contact_friction_out,
        contact_solref_out,
        contact_solreffriction_out,
        contact_solimp_out,
        contact_dim_out,
        contact_geom_out,
        contact_flex_out,
        contact_vert_out,
        contact_elem_out,
        contact_worldid_out,
        contact_type_out,
        contact_geomcollisionid_out,
        nacon_out,
      )


@wp.kernel
def _flex_narrowphase_dim2(
  # Model:
  ngeom: int,
  nflex: int,
  geom_type: wp.array[int],
  geom_contype: wp.array[int],
  geom_conaffinity: wp.array[int],
  geom_condim: wp.array[int],
  geom_priority: wp.array[int],
  geom_solmix: wp.array2d[float],
  geom_solref: wp.array2d[wp.vec2],
  geom_solimp: wp.array2d[vec5],
  geom_size: wp.array2d[wp.vec3],
  geom_friction: wp.array2d[wp.vec3],
  geom_margin: wp.array2d[float],
  geom_gap: wp.array2d[float],
  flex_contype: wp.array[int],
  flex_conaffinity: wp.array[int],
  flex_condim: wp.array[int],
  flex_priority: wp.array[int],
  flex_solmix: wp.array[float],
  flex_solref: wp.array[wp.vec2],
  flex_solimp: wp.array[vec5],
  flex_friction: wp.array[wp.vec3],
  flex_margin: wp.array[float],
  flex_gap: wp.array[float],
  flex_dim: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemadr: wp.array[int],
  flex_elemnum: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_elem: wp.array[int],
  flex_radius: wp.array[float],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  nworld_in: int,
  naconmax_in: int,
  # Data out:
  contact_dist_out: wp.array[float],
  contact_pos_out: wp.array[wp.vec3],
  contact_frame_out: wp.array[wp.mat33],
  contact_includemargin_out: wp.array[float],
  contact_friction_out: wp.array[vec5],
  contact_solref_out: wp.array[wp.vec2],
  contact_solreffriction_out: wp.array[wp.vec2],
  contact_solimp_out: wp.array[vec5],
  contact_dim_out: wp.array[int],
  contact_geom_out: wp.array[wp.vec2i],
  contact_flex_out: wp.array[wp.vec2i],
  contact_vert_out: wp.array[wp.vec2i],
  contact_elem_out: wp.array[wp.vec2i],
  contact_worldid_out: wp.array[int],
  contact_type_out: wp.array[int],
  contact_geomcollisionid_out: wp.array[int],
  nacon_out: wp.array[int],
):
  worldid, elemid = wp.tid()

  flexid = int(-1)
  for i in range(nflex):
    if flex_dim[i] != 2:
      continue
    elem_adr = flex_elemadr[i]
    elem_num = flex_elemnum[i]
    if elemid >= elem_adr and elemid < elem_adr + elem_num:
      flexid = i
      break

  if flexid < 0:
    return

  vert_adr = flex_vertadr[flexid]
  tri_radius = flex_radius[flexid]
  tri_margin = flex_margin[flexid]

  elem_data_idx = flex_elemdataadr[flexid] + (elemid - flex_elemadr[flexid]) * 3
  v0_local = flex_elem[elem_data_idx]
  v1_local = flex_elem[elem_data_idx + 1]
  v2_local = flex_elem[elem_data_idx + 2]

  t1 = flexvert_xpos_in[worldid, vert_adr + v0_local]
  t2 = flexvert_xpos_in[worldid, vert_adr + v1_local]
  t3 = flexvert_xpos_in[worldid, vert_adr + v2_local]

  # TODO: Add a broadphase
  for geomid in range(ngeom):
    gtype = geom_type[geomid]
    if (
      gtype != int(GeomType.SPHERE)
      and gtype != int(GeomType.CAPSULE)
      and gtype != int(GeomType.BOX)
      and gtype != int(GeomType.CYLINDER)
    ):
      continue

    g_contype = geom_contype[geomid]
    g_conaffinity = geom_conaffinity[geomid]
    f_contype = flex_contype[flexid]
    f_conaffinity = flex_conaffinity[flexid]
    if not ((g_contype & f_conaffinity) or (f_contype & g_conaffinity)):
      continue

    geom_margin_val = geom_margin[worldid % geom_margin.shape[0], geomid]
    margin = geom_margin_val + tri_margin

    geom_pos = geom_xpos_in[worldid, geomid]
    geom_rot = geom_xmat_in[worldid, geomid]
    geom_size_val = geom_size[worldid % geom_size.shape[0], geomid]

    condim, gap, solref, solimp, friction = _mix_flex_contact_params(
      geom_condim[geomid],
      geom_priority[geomid],
      geom_solmix[worldid % geom_solmix.shape[0], geomid],
      geom_solref[worldid % geom_solref.shape[0], geomid],
      geom_solimp[worldid % geom_solimp.shape[0], geomid],
      geom_friction[worldid % geom_friction.shape[0], geomid],
      geom_gap[worldid % geom_gap.shape[0], geomid],
      flex_condim[flexid],
      flex_priority[flexid],
      flex_solmix[flexid],
      flex_solref[flexid],
      flex_solimp[flexid],
      flex_friction[flexid],
      flex_gap[flexid],
    )

    _collide_geom_triangle(
      naconmax_in,
      gtype,
      geom_pos,
      geom_rot,
      geom_size_val,
      t1,
      t2,
      t3,
      tri_radius,
      margin,
      condim,
      friction,
      solref,
      solimp,
      geomid,
      flexid,
      -1,
      elemid,
      worldid,
      contact_dist_out,
      contact_pos_out,
      contact_frame_out,
      contact_includemargin_out,
      contact_friction_out,
      contact_solref_out,
      contact_solreffriction_out,
      contact_solimp_out,
      contact_dim_out,
      contact_geom_out,
      contact_flex_out,
      contact_vert_out,
      contact_elem_out,
      contact_worldid_out,
      contact_type_out,
      contact_geomcollisionid_out,
      nacon_out,
    )


@wp.kernel
def _flex_narrowphase_dim3(
  # Model:
  ngeom: int,
  nflex: int,
  geom_type: wp.array[int],
  geom_contype: wp.array[int],
  geom_conaffinity: wp.array[int],
  geom_condim: wp.array[int],
  geom_priority: wp.array[int],
  geom_solmix: wp.array2d[float],
  geom_solref: wp.array2d[wp.vec2],
  geom_solimp: wp.array2d[vec5],
  geom_size: wp.array2d[wp.vec3],
  geom_friction: wp.array2d[wp.vec3],
  geom_margin: wp.array2d[float],
  geom_gap: wp.array2d[float],
  flex_contype: wp.array[int],
  flex_conaffinity: wp.array[int],
  flex_condim: wp.array[int],
  flex_priority: wp.array[int],
  flex_solmix: wp.array[float],
  flex_solref: wp.array[wp.vec2],
  flex_solimp: wp.array[vec5],
  flex_friction: wp.array[wp.vec3],
  flex_margin: wp.array[float],
  flex_gap: wp.array[float],
  flex_dim: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemadr: wp.array[int],
  flex_elemnum: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_elem: wp.array[int],
  flex_radius: wp.array[float],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  nworld_in: int,
  naconmax_in: int,
  # Data out:
  contact_dist_out: wp.array[float],
  contact_pos_out: wp.array[wp.vec3],
  contact_frame_out: wp.array[wp.mat33],
  contact_includemargin_out: wp.array[float],
  contact_friction_out: wp.array[vec5],
  contact_solref_out: wp.array[wp.vec2],
  contact_solreffriction_out: wp.array[wp.vec2],
  contact_solimp_out: wp.array[vec5],
  contact_dim_out: wp.array[int],
  contact_geom_out: wp.array[wp.vec2i],
  contact_flex_out: wp.array[wp.vec2i],
  contact_vert_out: wp.array[wp.vec2i],
  contact_elem_out: wp.array[wp.vec2i],
  contact_worldid_out: wp.array[int],
  contact_type_out: wp.array[int],
  contact_geomcollisionid_out: wp.array[int],
  nacon_out: wp.array[int],
):
  worldid, elemid = wp.tid()

  # Find which flex owns this element
  flexid = int(-1)
  for i in range(nflex):
    if flex_dim[i] != 3:
      continue
    elem_adr = flex_elemadr[i]
    elem_num = flex_elemnum[i]
    if elemid >= elem_adr and elemid < elem_adr + elem_num:
      flexid = i
      break

  if flexid < 0:
    return

  vert_adr = flex_vertadr[flexid]
  tri_radius = flex_radius[flexid]
  tri_margin = flex_margin[flexid]

  # Extract 4 tet vertex indices (dim+1 = 4 for dim=3)
  local_elemid = elemid - flex_elemadr[flexid]
  edata_idx = flex_elemdataadr[flexid] + local_elemid * 4
  v0 = flex_elem[edata_idx]
  v1 = flex_elem[edata_idx + 1]
  v2 = flex_elem[edata_idx + 2]
  v3 = flex_elem[edata_idx + 3]

  # Fetch world-space vertex positions
  p0 = flexvert_xpos_in[worldid, vert_adr + v0]
  p1 = flexvert_xpos_in[worldid, vert_adr + v1]
  p2 = flexvert_xpos_in[worldid, vert_adr + v2]
  p3 = flexvert_xpos_in[worldid, vert_adr + v3]

  # TODO: Add a broadphase
  for geomid in range(ngeom):
    gtype = geom_type[geomid]
    if (
      gtype != int(GeomType.SPHERE)
      and gtype != int(GeomType.CAPSULE)
      and gtype != int(GeomType.BOX)
      and gtype != int(GeomType.CYLINDER)
    ):
      continue

    g_contype = geom_contype[geomid]
    g_conaffinity = geom_conaffinity[geomid]
    f_contype = flex_contype[flexid]
    f_conaffinity = flex_conaffinity[flexid]
    if not ((g_contype & f_conaffinity) or (f_contype & g_conaffinity)):
      continue

    geom_margin_val = geom_margin[worldid % geom_margin.shape[0], geomid]
    margin = geom_margin_val + tri_margin

    geom_pos = geom_xpos_in[worldid, geomid]
    geom_rot = geom_xmat_in[worldid, geomid]
    geom_size_val = geom_size[worldid % geom_size.shape[0], geomid]

    condim, gap, solref, solimp, friction = _mix_flex_contact_params(
      geom_condim[geomid],
      geom_priority[geomid],
      geom_solmix[worldid % geom_solmix.shape[0], geomid],
      geom_solref[worldid % geom_solref.shape[0], geomid],
      geom_solimp[worldid % geom_solimp.shape[0], geomid],
      geom_friction[worldid % geom_friction.shape[0], geomid],
      geom_gap[worldid % geom_gap.shape[0], geomid],
      flex_condim[flexid],
      flex_priority[flexid],
      flex_solmix[flexid],
      flex_solref[flexid],
      flex_solimp[flexid],
      flex_friction[flexid],
      flex_gap[flexid],
    )

    # Test all 4 triangular faces of the tet against the geom.
    # Face k is the triangle opposite vertex k:
    #   Face 0: (v1, v2, v3)
    #   Face 1: (v0, v2, v3)
    #   Face 2: (v0, v1, v3)
    #   Face 3: (v0, v1, v2)
    for face in range(4):
      if face == 0:
        t1 = p1
        t2 = p2
        t3 = p3
      elif face == 1:
        t1 = p0
        t2 = p2
        t3 = p3
      elif face == 2:
        t1 = p0
        t2 = p1
        t3 = p3
      else:
        t1 = p0
        t2 = p1
        t3 = p2

      _collide_geom_triangle(
        naconmax_in,
        gtype,
        geom_pos,
        geom_rot,
        geom_size_val,
        t1,
        t2,
        t3,
        tri_radius,
        margin,
        condim,
        friction,
        solref,
        solimp,
        geomid,
        flexid,
        -1,
        elemid,
        worldid,
        contact_dist_out,
        contact_pos_out,
        contact_frame_out,
        contact_includemargin_out,
        contact_friction_out,
        contact_solref_out,
        contact_solreffriction_out,
        contact_solimp_out,
        contact_dim_out,
        contact_geom_out,
        contact_flex_out,
        contact_vert_out,
        contact_elem_out,
        contact_worldid_out,
        contact_type_out,
        contact_geomcollisionid_out,
        nacon_out,
      )


# TODO(flex): improve parallelism, currently runs single-threaded per world
@wp.kernel
def _filter_flex_contacts(
  # Data in:
  contact_type_in: wp.array[int],
  naconmax_in: int,
  nacon_in: wp.array[int],
  # In:
  contact_pos_io: wp.array[wp.vec3],
  contact_dist_io: wp.array[float],
  contact_geom_io: wp.array[wp.vec2i],
  contact_flex_io: wp.array[wp.vec2i],
  # Data out:
  contact_type_out: wp.array[int],
):
  """Filter flex contacts using greedy farthest-point selection.

  For each (geom, flex) pair that has more than MJ_MAXCONPAIR contacts,
  select the MJ_MAXCONPAIR most spatially diverse contacts (matching the
  filterFlexContacts algorithm in MuJoCo C). Excluded contacts have their
  type set to 0 so the constraint pipeline ignores them.
  """
  nacon = wp.min(nacon_in[0], naconmax_in)

  # Iterate over all contacts to find (geom, flex) pair groups.
  # Process each unique pair: for each contact i that is a flex contact
  # and hasn't been processed yet, find all contacts with the same pair.
  for i in range(nacon):
    # Skip non-flex contacts or already-excluded contacts
    if contact_flex_io[i][1] < 0:
      continue
    if contact_type_in[i] == 0:
      continue

    geom_id = contact_geom_io[i][0]
    flex_id = contact_flex_io[i][1]

    # Count contacts for this (geom, flex) pair
    n = int(0)
    for j in range(i, nacon):
      if contact_type_in[j] == 0:
        continue
      if contact_geom_io[j][0] == geom_id and contact_flex_io[j][1] == flex_id:
        n += 1

    # If within limit, skip filtering for this pair
    if n <= MJ_MAXCONPAIR:
      continue

    # Greedy farthest-point selection:
    # 1. Find deepest penetrating contact as the first selection
    best = int(-1)
    bestdist = float(-1.0)
    for j in range(i, nacon):
      if contact_type_in[j] == 0:
        continue
      if contact_geom_io[j][0] == geom_id and contact_flex_io[j][1] == flex_id:
        neg_dist = -contact_dist_io[j]
        if neg_dist > bestdist:
          bestdist = neg_dist
          best = j

    # 2. Greedy selection loop
    nselected = int(0)
    while nselected < MJ_MAXCONPAIR and best >= 0:
      # mark as selected using flag bit
      contact_type_out[best] = contact_type_in[best] | int(0x40000000)
      bestpos = contact_pos_io[best]

      # Find the unselected contact that is farthest from all selected contacts
      nextbest = int(-1)
      nextbestdist = float(-1.0)
      for j in range(i, nacon):
        if contact_type_in[j] == 0:
          continue
        if contact_geom_io[j][0] != geom_id or contact_flex_io[j][1] != flex_id:
          continue
        # Skip already selected
        if contact_type_out[j] & int(0x40000000):
          continue

        dx = contact_pos_io[j][0] - bestpos[0]
        dy = contact_pos_io[j][1] - bestpos[1]
        dz = contact_pos_io[j][2] - bestpos[2]
        d2 = dx * dx + dy * dy + dz * dz

        # O(n*k) approach: for each candidate, compute its minimum distance
        # to ALL selected contacts.
        min_d2 = d2
        for k in range(i, nacon):
          if contact_type_in[k] == 0:
            continue
          if contact_geom_io[k][0] != geom_id or contact_flex_io[k][1] != flex_id:
            continue
          if not (contact_type_out[k] & int(0x40000000)):
            continue
          # k is a selected contact
          sx = contact_pos_io[j][0] - contact_pos_io[k][0]
          sy = contact_pos_io[j][1] - contact_pos_io[k][1]
          sz = contact_pos_io[j][2] - contact_pos_io[k][2]
          sd2 = sx * sx + sy * sy + sz * sz
          if sd2 < min_d2:
            min_d2 = sd2

        if min_d2 > nextbestdist:
          nextbestdist = min_d2
          nextbest = j

      nselected += 1
      best = nextbest

    # 3. Exclude non-selected contacts for this pair
    for j in range(i, nacon):
      if contact_type_in[j] == 0:
        continue
      if contact_geom_io[j][0] != geom_id or contact_flex_io[j][1] != flex_id:
        continue
      if contact_type_out[j] & int(0x40000000):
        # Clear the selection flag
        contact_type_out[j] = contact_type_out[j] & ~int(0x40000000)
      else:
        # Not selected: exclude this contact
        contact_type_out[j] = 0


@event_scope
def flex_narrowphase(m: Model, d: Data):
  """Runs collision detection between geoms and flex elements."""
  if m.nflex == 0:
    return

  wp.launch(
    _flex_narrowphase_dim2,
    dim=(d.nworld, m.nflexelem),
    inputs=[
      m.ngeom,
      m.nflex,
      m.geom_type,
      m.geom_contype,
      m.geom_conaffinity,
      m.geom_condim,
      m.geom_priority,
      m.geom_solmix,
      m.geom_solref,
      m.geom_solimp,
      m.geom_size,
      m.geom_friction,
      m.geom_margin,
      m.geom_gap,
      m.flex_contype,
      m.flex_conaffinity,
      m.flex_condim,
      m.flex_priority,
      m.flex_solmix,
      m.flex_solref,
      m.flex_solimp,
      m.flex_friction,
      m.flex_margin,
      m.flex_gap,
      m.flex_dim,
      m.flex_vertadr,
      m.flex_elemadr,
      m.flex_elemnum,
      m.flex_elemdataadr,
      m.flex_elem,
      m.flex_radius,
      d.geom_xpos,
      d.geom_xmat,
      d.flexvert_xpos,
      d.nworld,
      d.naconmax,
    ],
    outputs=[
      d.contact.dist,
      d.contact.pos,
      d.contact.frame,
      d.contact.includemargin,
      d.contact.friction,
      d.contact.solref,
      d.contact.solreffriction,
      d.contact.solimp,
      d.contact.dim,
      d.contact.geom,
      d.contact.flex,
      d.contact.vert,
      d.contact.elem,
      d.contact.worldid,
      d.contact.type,
      d.contact.geomcollisionid,
      d.nacon,
    ],
  )

  wp.launch(
    _flex_narrowphase_dim3,
    dim=(d.nworld, m.nflexelem),
    inputs=[
      m.ngeom,
      m.nflex,
      m.geom_type,
      m.geom_contype,
      m.geom_conaffinity,
      m.geom_condim,
      m.geom_priority,
      m.geom_solmix,
      m.geom_solref,
      m.geom_solimp,
      m.geom_size,
      m.geom_friction,
      m.geom_margin,
      m.geom_gap,
      m.flex_contype,
      m.flex_conaffinity,
      m.flex_condim,
      m.flex_priority,
      m.flex_solmix,
      m.flex_solref,
      m.flex_solimp,
      m.flex_friction,
      m.flex_margin,
      m.flex_gap,
      m.flex_dim,
      m.flex_vertadr,
      m.flex_elemadr,
      m.flex_elemnum,
      m.flex_elemdataadr,
      m.flex_elem,
      m.flex_radius,
      d.geom_xpos,
      d.geom_xmat,
      d.flexvert_xpos,
      d.nworld,
      d.naconmax,
    ],
    outputs=[
      d.contact.dist,
      d.contact.pos,
      d.contact.frame,
      d.contact.includemargin,
      d.contact.friction,
      d.contact.solref,
      d.contact.solreffriction,
      d.contact.solimp,
      d.contact.dim,
      d.contact.geom,
      d.contact.flex,
      d.contact.vert,
      d.contact.elem,
      d.contact.worldid,
      d.contact.type,
      d.contact.geomcollisionid,
      d.nacon,
    ],
  )

  wp.launch(
    _flex_plane_narrowphase,
    dim=(d.nworld, m.nflexvert),
    inputs=[
      m.ngeom,
      m.nflexvert,
      m.geom_type,
      m.geom_condim,
      m.geom_priority,
      m.geom_solmix,
      m.geom_solref,
      m.geom_solimp,
      m.geom_friction,
      m.geom_margin,
      m.geom_gap,
      m.flex_condim,
      m.flex_priority,
      m.flex_solmix,
      m.flex_solref,
      m.flex_solimp,
      m.flex_friction,
      m.flex_margin,
      m.flex_gap,
      m.flex_vertadr,
      m.flex_radius,
      m.flex_vertflexid,
      d.geom_xpos,
      d.geom_xmat,
      d.flexvert_xpos,
      d.nworld,
      d.naconmax,
    ],
    outputs=[
      d.contact.dist,
      d.contact.pos,
      d.contact.frame,
      d.contact.includemargin,
      d.contact.friction,
      d.contact.solref,
      d.contact.solreffriction,
      d.contact.solimp,
      d.contact.dim,
      d.contact.geom,
      d.contact.flex,
      d.contact.vert,
      d.contact.elem,
      d.contact.worldid,
      d.contact.type,
      d.contact.geomcollisionid,
      d.nacon,
    ],
  )

  # Filter flex contacts: limit contacts per (geom, flex) pair to
  # MJ_MAXCONPAIR (matching filterFlexContacts in MuJoCo C)
  wp.launch(
    _filter_flex_contacts,
    dim=d.nworld,
    inputs=[
      d.contact.type,
      d.naconmax,
      d.nacon,
      d.contact.pos,
      d.contact.dist,
      d.contact.geom,
      d.contact.flex,
    ],
    outputs=[
      d.contact.type,
    ],
  )

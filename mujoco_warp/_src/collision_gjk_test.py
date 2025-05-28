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
from absl.testing import absltest

from . import test_util
from .collision_gjk import gjk
from .collision_primitive import Geom
from .types import Data
from .types import GeomType
from .types import Model


def _geom_dist(m: Model, d: Data, gid1: int, gid2: int):
  @wp.kernel
  def _gjk_kernel(
    # Model:
    geom_type: wp.array(dtype=int),
    geom_dataid: wp.array(dtype=int),
    geom_size: wp.array2d(dtype=wp.vec3),
    mesh_vertadr: wp.array(dtype=int),
    mesh_vertnum: wp.array(dtype=int),
    mesh_vert: wp.array(dtype=wp.vec3),
    # Data in:
    geom_xpos_in: wp.array2d(dtype=wp.vec3),
    geom_xmat_in: wp.array2d(dtype=wp.mat33),
    # In:
    gid1: int,
    gid2: int,
    # Out:
    dist_out: wp.array(dtype=float),
  ):
    MESHGEOM = int(GeomType.MESH.value)

    geom1 = Geom()
    geomtype1 = geom_type[gid1]
    geom1.pos = geom_xpos_in[0, gid1]
    geom1.rot = geom_xmat_in[0, gid1]
    geom1.size = geom_size[0, gid1]

    if geom_dataid[gid1] >= 0 and geom_type[gid1] == MESHGEOM:
      dataid = geom_dataid[gid1]
      geom1.vertadr = mesh_vertadr[dataid]
      geom1.vertnum = mesh_vertnum[dataid]
      geom1.vert = mesh_vert

    geom2 = Geom()
    geomtype2 = geom_type[gid2]
    geom2.pos = geom_xpos_in[0, gid2]
    geom2.rot = geom_xmat_in[0, gid2]
    geom2.size = geom_size[0, gid2]

    if geom_dataid[gid2] >= 0 and geom_type[gid2] == MESHGEOM:
      dataid = geom_dataid[gid2]
      geom2.vertadr = mesh_vertadr[dataid]
      geom2.vertnum = mesh_vertnum[dataid]
      geom2.vert = mesh_vert

    x_1 = geom_xpos_in[0, gid1]
    x_2 = geom_xpos_in[0, gid2]
    result = gjk(1e-6, 20, geom1, geom2, x_1, x_2, geomtype1, geomtype2)
    dist_out[0] = result.dist

  dist_out = wp.array(shape=(1,), dtype=float)
  wp.launch(
    _gjk_kernel,
    dim=(1,),
    inputs=[      
      m.geom_type,
      m.geom_dataid,
      m.geom_size,
      m.mesh_vertadr,
      m.mesh_vertnum,
      m.mesh_vert,
      d.geom_xpos,
      d.geom_xmat,
      gid1,
      gid2,
    ],
    outputs=[
      dist_out,
    ],
  )
  return dist_out.numpy()[0]


class GJKTest(absltest.TestCase):
  """Tests for closest points between two convex geoms."""

  def test_spheres_nontouching(self):
    """Test closest points between two spheres not touching"""

    _, _, m, d = test_util.fixture(
      xml=f"""
      <mujoco>
        <worldbody>
          <geom name="geom1" type="sphere" pos="-1.5 0 0" size="1"/>
          <geom name="geom2" type="sphere" pos="1.5 0 0" size="1"/>
        </worldbody>
       </mujoco>
       """
    )

    dist = _geom_dist(m, d, 0, 1)
    self.assertEqual(1.0, dist)

  def test_spheres_touching(self):
    """Test closest points between two touching spheres"""

    _, _, m, d = test_util.fixture(
      xml=f"""
      <mujoco>
        <worldbody>
          <geom name="geom1" type="sphere" pos="-1 0 0" size="1"/>
          <geom name="geom2" type="sphere" pos="1 0 0" size="1"/>
        </worldbody>
       </mujoco>
       """
    )

    dist = _geom_dist(m, d, 0, 1)
    self.assertEqual(0.0, dist)

  def test_box_mesh_touching(self):
    """Test closest points between a mesh and box"""

    _, _, m, d = test_util.fixture(
      xml=f"""
      <mujoco model="MuJoCo Model">
        <asset>
          <mesh name="smallbox" scale="0.1 0.1 0.1"
                vertex="-1 -1 -1
                         1 -1 -1
                         1  1 -1
                         1  1  1
                         1 -1  1
                        -1  1 -1
                        -1  1  1
                        -1 -1  1"/>
         </asset>
         <worldbody>
           <geom name="geom1" pos="0 0 .90" type="box" size="0.5 0.5 0.1"/>
           <geom pos="0 0 1.2" name="geom2" type="mesh" mesh="smallbox"/>
          </worldbody>
       </mujoco>
       """
    )

    dist = _geom_dist(m, d, 0, 1)
    self.assertAlmostEqual(0.1, dist)


if __name__ == "__main__":
  wp.init()
  absltest.main()

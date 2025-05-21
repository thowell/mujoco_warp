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

from .collision_gjk import gjk

from . import test_util


class GJKTest(absltest.TestCase):
  """Tests for closest points between two convex geoms."""

  def test_spheres_nontouching(self):
    """Test closest points between two spheres"""

    _, mjd, m, d = test_util.fixture(
      xml=f"""
      <mujoco>
        <worldbody>
          <geom name="geom1" type="sphere" pos="-1.5 0 0" size="1"/>
          <geom name="geom2" type="sphere" pos="1.5 0 0" size="1"/>
        </worldbody>
       </mujoco>
       """)
    

    geom1 = 
    
    gjk(1e-6, 10, )

    tolerance: float, gjk_iterations: int,
        geom1: Geom, geom2: Geom, 
        x1_0: wp.vec3, x2_0: wp.vec3,
        geomtype1: int, geomtype2: int,
        verts: wp.array(dtype=wp.vec3)):
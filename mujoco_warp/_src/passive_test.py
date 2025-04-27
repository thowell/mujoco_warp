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

"""Tests for passive force functions."""

import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjwarp

from . import test_util

# tolerance for difference between MuJoCo and MJWarp passive force calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class PassiveTest(parameterized.TestCase):
  def test_passive(self):
    """Tests passive."""
    _, mjd, m, d = test_util.fixture("pendula.xml")

    for arr in (d.qfrc_spring, d.qfrc_damper, d.qfrc_passive):
      arr.zero_()

    mjwarp.passive(m, d)

    _assert_eq(d.qfrc_spring.numpy()[0], mjd.qfrc_spring, "qfrc_spring")
    _assert_eq(d.qfrc_damper.numpy()[0], mjd.qfrc_damper, "qfrc_damper")
    _assert_eq(d.qfrc_passive.numpy()[0], mjd.qfrc_passive, "qfrc_passive")

  # TODO(team): test DisableBit.PASSIVE

  @parameterized.parameters(
    (1, 0, 0, 0, 0),
    (0, 1, 0, 0, 0),
    (0, 0, 1, 0, 0),
    (0, 0, 0, 1, 0),
    (0, 0, 0, 0, 1),
    (1, 1, 1, 1, 1),
  )
  def test_fluid(self, density, viscosity, wind0, wind1, wind2):
    """Tests fluid model."""

    _, mjd, m, d = test_util.fixture(
      xml=f"""
      <mujoco>
        <option density="{density}" viscosity="{viscosity}" wind="{wind0} {wind1} {wind2}"/>
        <worldbody>
          <body>
            <geom type="box" size=".1 .1 .1"/>
            <freejoint/>
          </body>
        </worldbody>
        <keyframe>
          <key qvel="1 1 1 1 1 1"/>
        </keyframe>
      </mujoco>
    """,
      keyframe=0,
    )

    for arr in (d.qfrc_passive, d.qfrc_fluid):
      arr.zero_()

    mjwarp.passive(m, d)

    _assert_eq(d.qfrc_passive.numpy()[0], mjd.qfrc_passive, "qfrc_passive")
    _assert_eq(d.qfrc_fluid.numpy()[0], mjd.qfrc_fluid, "qfrc_fluid")


if __name__ == "__main__":
  wp.init()
  absltest.main()

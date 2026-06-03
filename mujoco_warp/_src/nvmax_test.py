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
# ==============================================================================

"""Tests for nv_compact active-DOF bookkeeping."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjwarp
from mujoco_warp._src import nvmax

# An actuated 2-hinge arm (tree 0, dofs 0-1) plus two free bodies
# (tree 1, dofs 2-7 and tree 2, dofs 8-13).
_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body>
      <joint name="j0" type="hinge"/>
      <geom type="capsule" fromto="0 0 0 0 0 .3" size=".05"/>
      <body pos="0 0 .3">
        <joint name="j1" type="hinge"/>
        <geom type="capsule" fromto="0 0 0 0 0 .3" size=".05"/>
      </body>
    </body>
    <body pos="1 0 1"><freejoint/><geom name="ball0" type="sphere" size=".1"/></body>
    <body pos="2 0 1"><freejoint/><geom name="ball1" type="sphere" size=".1"/></body>
  </worldbody>
  <actuator>
    <motor joint="j0"/>
    <motor joint="j1"/>
  </actuator>
</mujoco>
"""


def _setup(nworld=1, nv_max=None, sparse=False):
  mjm = mujoco.MjModel.from_xml_string(_XML)
  if sparse:
    mjm.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
  m = mjwarp.put_model(mjm)
  d = mjwarp.make_data(mjm, nworld=nworld, nv_max=nv_max)
  return mjm, m, d


# Two free spheres resting (penetrating) on a floor, plus an actuated hinge arm.
# Generates contacts so the constrained solver does real work.
_CONTACT_XML = """
<mujoco>
  <option jacobian="sparse" solver="Newton" iterations="20"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body>
      <joint name="j0" type="hinge"/>
      <geom type="capsule" fromto="0 0 .5 .3 0 .5" size=".05"/>
    </body>
    <body pos="1 0 .08"><freejoint/><geom type="sphere" size=".1"/></body>
    <body pos="2 0 .08"><freejoint/><geom type="sphere" size=".1"/></body>
  </worldbody>
  <actuator><motor joint="j0"/></actuator>
</mujoco>
"""


def _setup_contact(nv_max=None, sparse=False):
  mjm = mujoco.MjModel.from_xml_string(_CONTACT_XML)
  if sparse:
    mjm.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
  else:
    mjm.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
  m = mjwarp.put_model(mjm)
  d = mjwarp.make_data(mjm, nv_max=nv_max)
  return mjm, m, d


class NvCompactBookkeepingTest(absltest.TestCase):
  def test_compaction_mapping(self):
    """Manually setting awake trees results in correct compact maps."""
    _, m, d = _setup()
    # Set only tree 0 (actuated arm, dofs 0-1) awake
    tree_awake = np.zeros((d.nworld, m.ntree), dtype=np.int32)
    tree_awake[0, 0] = 1
    d.tree_awake = wp.array(tree_awake, dtype=int)

    nvmax.compact_dofs(m, d)

    self.assertEqual(d.ncdof.numpy()[0], 2)
    # arm dofs 0,1 map to compacted 0,1; the rest are inactive (-1).
    dof_cdof = d.dof_cdof.numpy()[0]
    np.testing.assert_array_equal(dof_cdof[:2], [0, 1])
    self.assertTrue((dof_cdof[2:] == -1).all())
    np.testing.assert_array_equal(d.cdof_dof.numpy()[0][:2], [0, 1])

  def test_compaction_multi_world(self):
    """Compacting DOFs works independently per world."""
    _, m, d = _setup(nworld=2)
    # world 0: tree 0 (arm, dofs 0-1) and tree 1 (free body, dofs 2-7) awake
    # world 1: tree 0 awake
    tree_awake = np.zeros((2, m.ntree), dtype=np.int32)
    tree_awake[0, 0] = 1
    tree_awake[0, 1] = 1
    tree_awake[1, 0] = 1
    d.tree_awake = wp.array(tree_awake, dtype=int)

    nvmax.compact_dofs(m, d)

    # world 0: arm (2) + free body (6) = 8; world 1: arm only = 2
    np.testing.assert_array_equal(d.ncdof.numpy(), [8, 2])
    np.testing.assert_array_equal(d.dof_cdof.numpy()[0][:8], np.arange(8))
    self.assertTrue((d.dof_cdof.numpy()[1][2:] == -1).all())

  def test_overflow_clamps_and_warns(self):
    """When active DOFs exceed nv_max, ncdof is clamped to nv_max."""
    _, m, d = _setup(nv_max=4)
    # Set all trees awake
    tree_awake = np.ones((d.nworld, m.ntree), dtype=np.int32)
    d.tree_awake = wp.array(tree_awake, dtype=int)

    nvmax.compact_dofs(m, d)

    self.assertEqual(d.ncdof.numpy()[0], 4)


class NvCompactSmoothSolveTest(parameterized.TestCase):
  @parameterized.parameters(True, False)
  def test_smooth_solve_equivalence_all_active(self, sparse):
    """With every tree active and nv_max=nv, compacted qacc_smooth matches baseline."""
    _, m, d = _setup(sparse=sparse)
    mjwarp.forward(m, d)
    baseline = d.qacc_smooth.numpy().copy()

    tree_awake = np.ones((d.nworld, m.ntree), dtype=np.int32)
    d.tree_awake = wp.array(tree_awake, dtype=int)
    nvmax.compact_dofs(m, d)
    self.assertEqual(d.ncdof.numpy()[0], m.nv)

    ctx = nvmax.create_nvcompact_context(m, d)
    nvmax.smooth_solve_compact(m, d, ctx)

    np.testing.assert_allclose(d.qacc_smooth.numpy(), baseline, rtol=1e-4, atol=1e-5)

  @parameterized.parameters(True, False)
  def test_smooth_solve_partial_active_freezes_rest(self, sparse):
    """Active trees match baseline (M is block-diagonal); inactive DOFs are frozen to 0."""
    _, m, d = _setup(sparse=sparse)
    mjwarp.forward(m, d)
    baseline = d.qacc_smooth.numpy().copy()

    # only the actuated arm tree (dofs 0-1) is active
    tree_awake = np.zeros((d.nworld, m.ntree), dtype=np.int32)
    tree_awake[0, 0] = 1
    d.tree_awake = wp.array(tree_awake, dtype=int)
    nvmax.compact_dofs(m, d)
    self.assertEqual(d.ncdof.numpy()[0], 2)

    ctx = nvmax.create_nvcompact_context(m, d)
    nvmax.smooth_solve_compact(m, d, ctx)

    out = d.qacc_smooth.numpy()
    np.testing.assert_allclose(out[0, :2], baseline[0, :2], rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(out[0, 2:], np.zeros(m.nv - 2))


class NvCompactConstrainedSolveTest(parameterized.TestCase):
  @parameterized.parameters(True, False)
  def test_constrained_solve_equivalence_all_active(self, sparse):
    """With every tree active and nv_max=nv, the compacted Newton solve matches baseline qacc."""
    _, m, d = _setup_contact(sparse=sparse)
    mjwarp.forward(m, d)  # full baseline solve (also builds efc.J, M)
    self.assertGreater(d.nacon.numpy()[0], 0)  # contacts exist
    baseline_qacc = d.qacc.numpy().copy()

    tree_awake = np.ones((d.nworld, m.ntree), dtype=np.int32)
    d.tree_awake = wp.array(tree_awake, dtype=int)
    nvmax.compact_dofs(m, d)
    self.assertEqual(d.ncdof.numpy()[0], m.nv)

    ctx = nvmax.create_nvcompact_context(m, d)
    nvmax.solve_compact(m, d, ctx)

    np.testing.assert_allclose(d.qacc.numpy(), baseline_qacc, rtol=1e-3, atol=1e-4)


if __name__ == "__main__":
  wp.clear_kernel_cache()
  absltest.main()

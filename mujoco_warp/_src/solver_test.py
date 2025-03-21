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

"""Tests for solver functions."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath

import mujoco_warp as mjwarp

from . import solver

# tolerance for difference between MuJoCo and MJWarp smooth calculations - mostly
# due to float precision
_TOLERANCE = 5e-3


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class SolverTest(parameterized.TestCase):
  def _load(
    self,
    fname: str,
    is_sparse: bool = True,
    cone: int = mujoco.mjtCone.mjCONE_PYRAMIDAL,
    solver_: int = mujoco.mjtSolver.mjSOL_NEWTON,
    iterations: int = 2,
    ls_iterations: int = 4,
    nworld: int = 1,
    njmax: int = 512,
    keyframe: int = 0,
    ls_parallel: bool = False,
  ):
    path = epath.resource_path("mujoco_warp") / "test_data" / fname
    mjm = mujoco.MjModel.from_xml_path(path.as_posix())
    mjm.opt.jacobian = is_sparse
    mjm.opt.iterations = iterations
    mjm.opt.ls_iterations = ls_iterations
    mjm.opt.cone = cone
    mjm.opt.solver = solver_
    mjm.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, keyframe)
    mujoco.mj_step(mjm, mjd)
    m = mjwarp.put_model(mjm)
    m.opt.ls_parallel = ls_parallel
    d = mjwarp.put_data(mjm, mjd, nworld=nworld, njmax=njmax)
    return mjm, mjd, m, d

  @parameterized.parameters(
    (mujoco.mjtCone.mjCONE_PYRAMIDAL, mujoco.mjtSolver.mjSOL_CG, 25, 5, False, False),
    (
      mujoco.mjtCone.mjCONE_PYRAMIDAL,
      mujoco.mjtSolver.mjSOL_NEWTON,
      2,
      4,
      False,
      False,
    ),
    (mujoco.mjtCone.mjCONE_PYRAMIDAL, mujoco.mjtSolver.mjSOL_NEWTON, 2, 4, True, True),
  )
  def test_solve(self, cone, solver_, iterations, ls_iterations, sparse, ls_parallel):
    """Tests solve."""
    for keyframe in range(3):
      mjm, mjd, m, d = self._load(
        "humanoid/humanoid.xml",
        is_sparse=sparse,
        cone=cone,
        solver_=solver_,
        iterations=iterations,
        ls_iterations=ls_iterations,
        keyframe=keyframe,
        ls_parallel=ls_parallel,
      )

      def cost(qacc):
        jaref = np.zeros(mjd.nefc, dtype=float)
        cost = np.zeros(1)
        mujoco.mj_mulJacVec(mjm, mjd, jaref, qacc)
        mujoco.mj_constraintUpdate(mjm, mjd, jaref - mjd.efc_aref, cost, 0)
        return cost

      mj_cost = cost(mjd.qacc)

      solver._create_context(m, d)

      mjwarp_cost = d.efc.cost.numpy()[0] - d.efc.gauss.numpy()[0]

      _assert_eq(mjwarp_cost, mj_cost, name="cost")

      qacc_warmstart = mjd.qacc_warmstart.copy()
      mujoco.mj_forward(mjm, mjd)
      mjd.qacc_warmstart = qacc_warmstart

      m = mjwarp.put_model(mjm)
      d = mjwarp.put_data(mjm, mjd, njmax=mjd.nefc)
      d.qacc.zero_()
      d.qfrc_constraint.zero_()
      d.efc.force.zero_()

      if solver_ == mujoco.mjtSolver.mjSOL_CG:
        mjwarp.factor_m(m, d)
      mjwarp.solve(m, d)

      mj_cost = cost(mjd.qacc)
      mjwarp_cost = cost(d.qacc.numpy()[0])
      self.assertLessEqual(mjwarp_cost, mj_cost * 1.025)

      if m.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON:
        _assert_eq(d.qacc.numpy()[0], mjd.qacc, "qacc")
        _assert_eq(d.qfrc_constraint.numpy()[0], mjd.qfrc_constraint, "qfrc_constraint")
        _assert_eq(d.efc.force.numpy()[: mjd.nefc], mjd.efc_force, "efc_force")

  @parameterized.parameters(
    (mujoco.mjtCone.mjCONE_PYRAMIDAL, mujoco.mjtSolver.mjSOL_CG, 25, 5),
    (mujoco.mjtCone.mjCONE_PYRAMIDAL, mujoco.mjtSolver.mjSOL_NEWTON, 2, 4),
  )
  def test_solve_batch(self, cone, solver_, iterations, ls_iterations):
    """Tests solve (batch)."""
    mjm0, mjd0, _, _ = self._load(
      "humanoid/humanoid.xml",
      is_sparse=False,
      cone=cone,
      solver_=solver_,
      iterations=iterations,
      ls_iterations=ls_iterations,
      keyframe=0,
    )
    qacc_warmstart0 = mjd0.qacc_warmstart.copy()
    mujoco.mj_forward(mjm0, mjd0)
    mjd0.qacc_warmstart = qacc_warmstart0

    mjm1, mjd1, _, _ = self._load(
      "humanoid/humanoid.xml",
      is_sparse=False,
      cone=cone,
      solver_=solver_,
      iterations=iterations,
      ls_iterations=ls_iterations,
      keyframe=2,
    )
    qacc_warmstart1 = mjd1.qacc_warmstart.copy()
    mujoco.mj_forward(mjm1, mjd1)
    mjd1.qacc_warmstart = qacc_warmstart1

    mjm2, mjd2, _, _ = self._load(
      "humanoid/humanoid.xml",
      is_sparse=False,
      cone=cone,
      solver_=solver_,
      iterations=iterations,
      ls_iterations=ls_iterations,
      keyframe=1,
    )
    qacc_warmstart2 = mjd2.qacc_warmstart.copy()
    mujoco.mj_forward(mjm2, mjd2)
    mjd2.qacc_warmstart = qacc_warmstart2

    nefc_active = mjd0.nefc + mjd1.nefc + mjd2.nefc

    mjm, _, m, d = self._load(
      "humanoid/humanoid.xml",
      is_sparse=False,
      cone=cone,
      solver_=solver_,
      iterations=iterations,
      ls_iterations=ls_iterations,
      nworld=3,
      njmax=2 * nefc_active,
    )

    d.nefc = wp.array([nefc_active], dtype=wp.int32, ndim=1)

    nefc_fill = d.njmax - nefc_active

    qacc_warmstart = np.vstack(
      [
        np.expand_dims(qacc_warmstart0, axis=0),
        np.expand_dims(qacc_warmstart1, axis=0),
        np.expand_dims(qacc_warmstart2, axis=0),
      ]
    )

    qM0 = np.zeros((mjm0.nv, mjm0.nv))
    mujoco.mj_fullM(mjm0, qM0, mjd0.qM)
    qM1 = np.zeros((mjm1.nv, mjm1.nv))
    mujoco.mj_fullM(mjm1, qM1, mjd1.qM)
    qM2 = np.zeros((mjm2.nv, mjm2.nv))
    mujoco.mj_fullM(mjm2, qM2, mjd2.qM)

    qM = np.vstack(
      [
        np.expand_dims(qM0, axis=0),
        np.expand_dims(qM1, axis=0),
        np.expand_dims(qM2, axis=0),
      ]
    )
    qacc_smooth = np.vstack(
      [
        np.expand_dims(mjd0.qacc_smooth, axis=0),
        np.expand_dims(mjd1.qacc_smooth, axis=0),
        np.expand_dims(mjd2.qacc_smooth, axis=0),
      ]
    )
    qfrc_smooth = np.vstack(
      [
        np.expand_dims(mjd0.qfrc_smooth, axis=0),
        np.expand_dims(mjd1.qfrc_smooth, axis=0),
        np.expand_dims(mjd2.qfrc_smooth, axis=0),
      ]
    )

    efc_J0 = mjd0.efc_J.reshape((mjd0.nefc, mjm0.nv))
    efc_J1 = mjd1.efc_J.reshape((mjd1.nefc, mjm1.nv))
    efc_J2 = mjd2.efc_J.reshape((mjd2.nefc, mjm2.nv))

    efc_J_fill = np.vstack([efc_J0, efc_J1, efc_J2, np.zeros((nefc_fill, mjm.nv))])

    efc_D_fill = np.concatenate(
      [mjd0.efc_D, mjd1.efc_D, mjd2.efc_D, np.zeros(nefc_fill)]
    )

    efc_aref_fill = np.concatenate(
      [mjd0.efc_aref, mjd1.efc_aref, mjd2.efc_aref, np.zeros(nefc_fill)]
    )

    efc_worldid = np.concatenate(
      [[0] * mjd0.nefc, [1] * mjd1.nefc, [2] * mjd2.nefc, [-1] * nefc_fill]
    )

    d.qacc_warmstart = wp.from_numpy(qacc_warmstart, dtype=wp.float32)
    d.qM = wp.from_numpy(qM, dtype=wp.float32)
    d.qacc_smooth = wp.from_numpy(qacc_smooth, dtype=wp.float32)
    d.qfrc_smooth = wp.from_numpy(qfrc_smooth, dtype=wp.float32)
    d.efc.J = wp.from_numpy(efc_J_fill, dtype=wp.float32)
    d.efc.D = wp.from_numpy(efc_D_fill, dtype=wp.float32)
    d.efc.aref = wp.from_numpy(efc_aref_fill, dtype=wp.float32)
    d.efc.worldid = wp.from_numpy(efc_worldid, dtype=wp.int32)

    if solver_ == mujoco.mjtSolver.mjSOL_CG:
      m0 = mjwarp.put_model(mjm0)
      d0 = mjwarp.put_data(mjm0, mjd0)
      mjwarp.factor_m(m0, d0)
      qLD0 = d0.qLD.numpy()

      m1 = mjwarp.put_model(mjm1)
      d1 = mjwarp.put_data(mjm1, mjd1)
      mjwarp.factor_m(m1, d1)
      qLD1 = d1.qLD.numpy()

      m2 = mjwarp.put_model(mjm2)
      d2 = mjwarp.put_data(mjm2, mjd2)
      mjwarp.factor_m(m2, d2)
      qLD2 = d2.qLD.numpy()

      qLD = np.vstack([qLD0, qLD1, qLD2])
      d.qLD = wp.from_numpy(qLD, dtype=wp.float32)

    d.qacc.zero_()
    d.qfrc_constraint.zero_()
    d.efc.force.zero_()
    solver.solve(m, d)

    def cost(m, d, qacc):
      jaref = np.zeros(d.nefc, dtype=float)
      cost = np.zeros(1)
      mujoco.mj_mulJacVec(m, d, jaref, qacc)
      mujoco.mj_constraintUpdate(m, d, jaref - d.efc_aref, cost, 0)
      return cost

    mj_cost0 = cost(mjm0, mjd0, mjd0.qacc)
    mjwarp_cost0 = cost(mjm0, mjd0, d.qacc.numpy()[0])
    self.assertLessEqual(mjwarp_cost0, mj_cost0 * 1.025)

    mj_cost1 = cost(mjm1, mjd1, mjd1.qacc)
    mjwarp_cost1 = cost(mjm1, mjd1, d.qacc.numpy()[1])
    self.assertLessEqual(mjwarp_cost1, mj_cost1 * 1.025)

    mj_cost2 = cost(mjm2, mjd2, mjd2.qacc)
    mjwarp_cost2 = cost(mjm2, mjd2, d.qacc.numpy()[2])
    self.assertLessEqual(mjwarp_cost2, mj_cost2 * 1.025)

    if m.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON:
      _assert_eq(d.qacc.numpy()[0], mjd0.qacc, "qacc0")
      _assert_eq(d.qacc.numpy()[1], mjd1.qacc, "qacc1")
      _assert_eq(d.qacc.numpy()[2], mjd2.qacc, "qacc2")

      _assert_eq(d.qfrc_constraint.numpy()[0], mjd0.qfrc_constraint, "qfrc_constraint0")
      _assert_eq(d.qfrc_constraint.numpy()[1], mjd1.qfrc_constraint, "qfrc_constraint1")
      _assert_eq(d.qfrc_constraint.numpy()[2], mjd2.qfrc_constraint, "qfrc_constraint2")

      _assert_eq(
        d.efc.force.numpy()[: mjd0.nefc],
        mjd0.efc_force,
        "efc_force0",
      )
      _assert_eq(
        d.efc.force.numpy()[mjd0.nefc : mjd0.nefc + mjd1.nefc],
        mjd1.efc_force,
        "efc_force1",
      )
      _assert_eq(
        d.efc.force.numpy()[mjd0.nefc + mjd1.nefc : mjd0.nefc + mjd1.nefc + mjd2.nefc],
        mjd2.efc_force,
        "efc_force2",
      )

  def test_frictionloss(self):
    """Tests solver with frictionloss."""
    mjm = mujoco.MjModel.from_xml_string(""""
      <mujoco>
        <worldbody>
          <body>
            <joint type="slide" axis="0 0 1" frictionloss=".2"/>
            <geom type="sphere" size=".1"/>
          </body>
        </worldbody>
        <keyframe>
          <key qpos="1" qvel="1"/>
        </keyframe>
      </mujoco>
    """)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    mujoco.mj_forward(mjm, mjd)

    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd)
    mjwarp.solve(m, d)

    _assert_eq(d.qacc.numpy()[0], mjd.qacc, "qacc")
    _assert_eq(d.qfrc_constraint.numpy()[0], mjd.qfrc_constraint, "qfrc_constraint")
    _assert_eq(d.efc.force.numpy()[: mjd.nefc], mjd.efc_force, "efc_force")


if __name__ == "__main__":
  wp.init()
  absltest.main()

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

"""Tests for sleep and wake features."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjwarp
from mujoco_warp import test_data
from mujoco_warp._src import io
from mujoco_warp._src import sleep
from mujoco_warp._src import types
from mujoco_warp._src.types import SleepState

wp.set_module_options({"enable_backward": False})


class SleepTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    io.ENABLE_ISLANDS = True

  def tearDown(self):
    io.ENABLE_ISLANDS = False
    super().tearDown()

  def test_sleep_initiation(self):
    """Verify that a stationary body on a flat plane goes to sleep."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option sleep_tolerance="0.01">
          <flag sleep="enable" island="enable"/>
        </option>
        <worldbody>
          <geom type="plane" size="10 10 .1"/>
          <body name="box" pos="0 0 0.1">
            <joint type="free"/>
            <geom type="box" size=".1 .1 .1" mass="1.0"/>
          </body>
        </worldbody>
      </mujoco>
      """,
    )

    # Set initial state: close to the ground, with zero velocity
    qpos = d.qpos.numpy()
    qpos[0, 2] = 0.1  # sit on floor
    d.qpos = wp.array(qpos, dtype=float)

    # Run a few steps to let it come to rest and count down
    for _ in range(15):
      mjwarp.step(m, d)

    # After 15 steps under sleep tolerance, it should have gone to sleep
    tree_asleep = d.tree_asleep.numpy()[0]
    tree_awake = d.tree_awake.numpy()[0]
    body_awake = d.body_awake.numpy()[0]

    # The box tree ID is 0. Since it's asleep, it should be in a cycle (value >= 0)
    # Since there's only 1 tree, it should cycle to itself: tree_asleep[0] == 0
    self.assertEqual(tree_asleep[0], 0, f"Expected self-cycle (0) for tree 0, got {tree_asleep[0]}")
    self.assertEqual(tree_awake[0], 0, "Tree should be asleep")
    self.assertEqual(body_awake[1], SleepState.ASLEEP, "Body should be asleep")  # body 1 is the box, 0 is world

    # Dof velocity and acceleration should be zeroed
    qvel = d.qvel.numpy()[0]
    qacc = d.qacc.numpy()[0]
    print("Final qvel:", qvel)
    self.assertTrue((qvel == 0.0).all())
    self.assertTrue((qacc == 0.0).all())

  def test_collision_waking(self):
    """Verify that a moving body colliding with a sleeping body wakes it up."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag sleep="enable" island="enable"/>
        </option>
        <worldbody>
          <geom type="plane" size="10 10 .1"/>
          <body name="target" pos="0 0 0.1">
            <joint type="free"/>
            <geom type="sphere" size=".1" mass="1.0"/>
          </body>
          <body name="bullet" pos="-0.3 0 0.1">
            <joint type="free"/>
            <geom type="sphere" size=".1" mass="1.0"/>
          </body>
        </worldbody>
      </mujoco>
      """,
    )

    # Set initial state: target is sitting, bullet is moving towards target
    qvel = d.qvel.numpy()
    qvel[0, 6] = 20.0  # high bullet velocity along x to ensure collision within few steps
    d.qvel = wp.array(qvel, dtype=float)

    # Put target (tree 0) to sleep manually
    tree_asleep = d.tree_asleep.numpy()
    tree_asleep[0, 0] = 0  # self-cycle for target
    d.tree_asleep = wp.array(tree_asleep, dtype=int)

    # Update sleep arrays to reflect the target is asleep
    sleep.update_sleep(m, d)
    self.assertEqual(d.tree_awake.numpy()[0, 0], 0, "Target tree should start asleep")

    # Run physics until contact occurs
    for _ in range(5):
      mjwarp.step(m, d)

    # Target should be wake up because of contact!
    self.assertEqual(d.tree_awake.numpy()[0, 0], 1, "Target tree should have woken up upon contact")
    self.assertLess(d.tree_asleep.numpy()[0, 0], 0, "Target tree_asleep should be negative (awake)")

  def test_tendon_waking(self):
    """Verify that pulling an awake body wakes up a connected sleeping body through a tendon."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag sleep="enable" island="enable"/>
        </option>
        <worldbody>
          <body name="b1" pos="0 0 0.5">
            <joint name="j1" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1"/>
          </body>
          <body name="b2" pos="1 0 0.5">
            <joint name="j2" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1"/>
          </body>
        </worldbody>
        <tendon>
          <fixed name="ten" limited="true" range="0 0.9">
            <joint joint="j1" coef="1.0"/>
            <joint joint="j2" coef="-1.0"/>
          </fixed>
        </tendon>
      </mujoco>
      """,
    )

    # Put b2 (tree 1) to sleep manually
    tree_asleep = d.tree_asleep.numpy()
    tree_asleep[0, 1] = 1  # self-cycle for b2
    d.tree_asleep = wp.array(tree_asleep, dtype=int)
    sleep.update_sleep(m, d)

    # Force b1 (tree 0) to move down, which will pull on b2 via tendon limit
    qvel = d.qvel.numpy()
    qvel[0, 0] = -5.0  # b1 moving down
    d.qvel = wp.array(qvel, dtype=float)

    # Run two steps: the first step integrates b1's velocity to move it down,
    # violating the tendon limit. The second step detects the violation in the
    # position phase and wakes up b2.
    mjwarp.step(m, d)
    mjwarp.step(m, d)

    # b2 (tree 1) should be woken up by tendon pull!
    self.assertEqual(d.tree_awake.numpy()[0, 1], 1, "b2 should have woken up due to tendon constraint")

  def test_equality_waking(self):
    """Verify that moving an awake body wakes up a connected sleeping body.

    This is mediated through a weld equality.
    """
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag sleep="enable" island="enable"/>
        </option>
        <worldbody>
          <body name="b1" pos="0 0 0.5">
            <joint type="free"/>
            <geom type="sphere" size=".1"/>
          </body>
          <body name="b2" pos="1 0 0.5">
            <joint type="free"/>
            <geom type="sphere" size=".1"/>
          </body>
        </worldbody>
        <equality>
          <weld body1="b1" body2="b2"/>
        </equality>
      </mujoco>
      """,
    )

    # Put b2 (tree 1) to sleep manually
    tree_asleep = d.tree_asleep.numpy()
    tree_asleep[0, 1] = 1  # self-cycle for b2
    d.tree_asleep = wp.array(tree_asleep, dtype=int)
    sleep.update_sleep(m, d)

    # Force b1 (tree 0) to move
    qvel = d.qvel.numpy()
    qvel[0, 0] = 2.0  # b1 moving
    d.qvel = wp.array(qvel, dtype=float)

    # Run step
    mjwarp.step(m, d)

    # b2 (tree 1) should be woken up by equality weld!
    self.assertEqual(d.tree_awake.numpy()[0, 1], 1, "b2 should have woken up due to weld constraint")

  def test_waking_unaffected_by_sleeping(self):
    """Verify that sleeping trees do not affect the physical rollout trajectories of awake trees."""
    xml = """
    <mujoco>
      <option sleep_tolerance="0.01">
        <flag sleep="enable" island="enable"/>
      </option>
      <worldbody>
        <geom type="plane" size="10 10 .1"/>
        <body name="b1" pos="0 0 0.1">
          <joint type="free"/>
          <geom type="box" size=".1 .1 .1" mass="1.0"/>
        </body>
        <body name="b2" pos="1 0 0.5">
          <joint type="free"/>
          <geom type="box" size=".1 .1 .1" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    # 1. Run with sleep enabled
    _, _, m_sleep, d_sleep = test_data.fixture(
      xml=xml,
    )

    # Place b1 exactly on the ground so it sleeps
    qpos_sleep = d_sleep.qpos.numpy()
    qpos_sleep[0, 2] = 0.1  # b1 sitting
    d_sleep.qpos = wp.array(qpos_sleep, dtype=float)

    # 2. Run with sleep disabled
    _, _, m_nosleep, d_nosleep = test_data.fixture(
      xml=xml,
    )

    qpos_nosleep = d_nosleep.qpos.numpy()
    qpos_nosleep[0, 2] = 0.1  # b1 sitting
    d_nosleep.qpos = wp.array(qpos_nosleep, dtype=float)

    # Step both simulators in parallel
    for step_idx in range(25):
      mjwarp.step(m_sleep, d_sleep)
      mjwarp.step(m_nosleep, d_nosleep)

      # b2 (tree 1) must remain awake in both simulations and have identical states
      qpos_s = d_sleep.qpos.numpy()[0]
      qpos_ns = d_nosleep.qpos.numpy()[0]
      qvel_s = d_sleep.qvel.numpy()[0]
      qvel_ns = d_nosleep.qvel.numpy()[0]

      # Verify that b2's position and velocity matches exactly (to 6 decimal places)
      np.testing.assert_allclose(qpos_s[7:14], qpos_ns[7:14], rtol=1e-6, atol=1e-6)
      np.testing.assert_allclose(qvel_s[6:12], qvel_ns[6:12], rtol=1e-6, atol=1e-6)

    # Verify that b1 indeed went to sleep in d_sleep
    self.assertEqual(d_sleep.tree_awake.numpy()[0, 0], 0, "b1 should have gone to sleep")

  def test_settle_zero_velocity(self):
    """Verify that a moving body in zero-gravity with damping goes to sleep.

    It must settle and have exactly zero velocity.
    """
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity="0 0 0" sleep_tolerance="0.01">
          <flag sleep="enable" island="enable"/>
        </option>
        <worldbody>
          <body name="box" pos="0 0 0">
            <joint type="slide" axis="1 0 0" damping="400.0"/>
            <geom type="box" size=".1 .1 .1" mass="1.0"/>
          </body>
        </worldbody>
      </mujoco>
      """,
    )

    # Give initial velocity
    qvel = d.qvel.numpy()
    qvel[0, 0] = 5.0
    d.qvel = wp.array(qvel, dtype=float)

    # Run steps to let it settle
    for _ in range(25):
      mjwarp.step(m, d)

    # It must be asleep
    self.assertEqual(d.tree_awake.numpy()[0, 0], 0, "Tree should have gone to sleep")
    self.assertEqual(d.body_awake.numpy()[0, 1], SleepState.ASLEEP, "Body should have gone to sleep")

    # Velocity and acceleration must be exactly zero (bitwise)
    qvel_final = d.qvel.numpy()[0]
    qacc_final = d.qacc.numpy()[0]
    self.assertEqual(qvel_final[0], 0.0, "Velocity should be exactly zeroed")
    self.assertEqual(qacc_final[0], 0.0, "Acceleration should be exactly zeroed")

  def test_slack_tendon_sleeping(self):
    """Verify that a slack tendon does not wake connected sleeping bodies."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag sleep="enable" island="enable"/>
        </option>
        <worldbody>
          <body name="b1" pos="0 0 0.5">
            <joint name="j1" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1"/>
          </body>
          <body name="b2" pos="1 0 0.5">
            <joint name="j2" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1"/>
          </body>
        </worldbody>
        <tendon>
          <fixed name="ten" limited="true" range="0 1.5">
            <joint joint="j1" coef="1.0"/>
            <joint joint="j2" coef="-1.0"/>
          </fixed>
        </tendon>
      </mujoco>
      """,
    )

    # Put b2 (tree 1) to sleep manually
    tree_asleep = d.tree_asleep.numpy()
    tree_asleep[0, 1] = 1  # self-cycle for b2
    d.tree_asleep = wp.array(tree_asleep, dtype=int)
    sleep.update_sleep(m, d)

    # Force b1 (tree 0) to move slowly, tendon is well within range 0-1.5, so it is slack
    qvel = d.qvel.numpy()
    qvel[0, 0] = 0.1  # moving b1 slightly
    d.qvel = wp.array(qvel, dtype=float)

    # Run step
    mjwarp.step(m, d)

    # b2 (tree 1) must remain asleep because the tendon is slack!
    self.assertEqual(d.tree_awake.numpy()[0, 1], 0, "b2 should remain asleep since tendon is slack")

  def test_equality_sleep_stability(self):
    """Verify that weld-connected bodies stay asleep together.

    They should not trigger a self-waking loop.
    """
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag sleep="enable" island="enable"/>
        </option>
        <worldbody>
          <body name="b1" pos="0 0 0.5">
            <joint type="free"/>
            <geom type="sphere" size=".1"/>
          </body>
          <body name="b2" pos="1 0 0.5">
            <joint type="free"/>
            <geom type="sphere" size=".1"/>
          </body>
        </worldbody>
        <equality>
          <weld body1="b1" body2="b2"/>
        </equality>
      </mujoco>
      """,
    )

    # Put both b1 and b2 to sleep manually in a cycle: tree_asleep[0] = 1, tree_asleep[1] = 0
    tree_asleep = d.tree_asleep.numpy()
    tree_asleep[0, 0] = 1
    tree_asleep[0, 1] = 0
    d.tree_asleep = wp.array(tree_asleep, dtype=int)
    sleep.update_sleep(m, d)

    self.assertEqual(d.tree_awake.numpy()[0, 0], 0, "b1 should start asleep")
    self.assertEqual(d.tree_awake.numpy()[0, 1], 0, "b2 should start asleep")

    # Run step
    mjwarp.step(m, d)

    # Both must remain asleep!
    self.assertEqual(d.tree_awake.numpy()[0, 0], 0, "b1 should remain asleep after a step")
    self.assertEqual(d.tree_awake.numpy()[0, 1], 0, "b2 should remain asleep after a step")

  def test_mj_wake_tree(self):
    """Direct unit test of cycle-traversal wake_tree function."""

    @wp.kernel(module="unique", enable_backward=False)
    def _test_wake_tree_kernel(
      # Model:
      ntree: int,
      # In:
      worldid: int,
      target_tree: int,
      wakeval: int,
      # Data out:
      tree_asleep_out: wp.array2d[int],
      # Out:
      woke_count_out: wp.array[int],
    ):
      woke_count_out[0] = sleep._wake_tree(ntree, worldid, target_tree, wakeval, tree_asleep_out)

    k_awake = sleep.K_AWAKE_VAL

    # Setup array: shape (1, 4), which contains 1 awake tree (index 0) and two cycles (1-2 and 3-3)
    asleep_init = np.array([[k_awake, 2, 1, 3]], dtype=np.int32)
    tree_asleep = wp.array(asleep_init, dtype=int)
    woke_count = wp.zeros((1,), dtype=int)

    # Launch kernel to execute the wp.func on device
    wp.launch(
      _test_wake_tree_kernel,
      dim=1,
      inputs=[4, 0, 0, k_awake],
      outputs=[tree_asleep, woke_count],
    )
    self.assertEqual(woke_count.numpy()[0], 0)
    np.testing.assert_array_equal(tree_asleep.numpy(), asleep_init)

    # Wake tree 1 (part of cycle 1-2) -> woke count should be 2
    wp.launch(
      _test_wake_tree_kernel,
      dim=1,
      inputs=[4, 0, 1, k_awake],
      outputs=[tree_asleep, woke_count],
    )
    self.assertEqual(woke_count.numpy()[0], 2)
    np.testing.assert_array_equal(tree_asleep.numpy()[0], [k_awake, k_awake, k_awake, 3])

    # Wake tree 3 (self-cycle 3) -> woke count should be 1
    wp.launch(
      _test_wake_tree_kernel,
      dim=1,
      inputs=[4, 0, 3, k_awake],
      outputs=[tree_asleep, woke_count],
    )
    self.assertEqual(woke_count.numpy()[0], 1)
    np.testing.assert_array_equal(tree_asleep.numpy()[0], [k_awake, k_awake, k_awake, k_awake])

  def test_multitree_tendon_waking(self):
    """Verify pulling wakes up multiple connected sleeping bodies through a single tendon."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag sleep="enable" island="enable"/>
        </option>
        <worldbody>
          <body name="b1" pos="0 0 0.5">
            <joint name="j1" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1"/>
          </body>
          <body name="b2" pos="1 0 0.5">
            <joint name="j2" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1"/>
          </body>
          <body name="b3" pos="2 0 0.5">
            <joint name="j3" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1"/>
          </body>
        </worldbody>
        <tendon>
          <fixed name="ten" limited="true" range="0 0.9">
            <joint joint="j1" coef="1.0"/>
            <joint joint="j2" coef="-1.0"/>
            <joint joint="j3" coef="0.5"/>
          </fixed>
        </tendon>
      </mujoco>
      """,
    )

    # Put both b2 (tree 1) and b3 (tree 2) to sleep manually
    tree_asleep = d.tree_asleep.numpy()
    tree_asleep[0, 1] = 1  # self-cycle for b2
    tree_asleep[0, 2] = 2  # self-cycle for b3
    d.tree_asleep = wp.array(tree_asleep, dtype=int)
    sleep.update_sleep(m, d)

    self.assertEqual(d.tree_awake.numpy()[0, 0], 1, "b1 should start awake")
    self.assertEqual(d.tree_awake.numpy()[0, 1], 0, "b2 should start asleep")
    self.assertEqual(d.tree_awake.numpy()[0, 2], 0, "b3 should start asleep")

    # Force b1 (tree 0) to move down, which will pull on b2 and b3 via tendon limit
    qvel = d.qvel.numpy()
    qvel[0, 0] = -5.0  # b1 moving down
    d.qvel = wp.array(qvel, dtype=float)

    # Run two steps to trigger position-phase tendon limit waking
    mjwarp.step(m, d)
    mjwarp.step(m, d)

    # BOTH b2 and b3 should be woken up!
    self.assertEqual(d.tree_awake.numpy()[0, 1], 1, "b2 should have woken up due to tendon constraint")
    self.assertEqual(d.tree_awake.numpy()[0, 2], 1, "b3 should have woken up due to tendon constraint")

  def test_tendon_equality_waking(self):
    """Verify that tendon equality constraint wakes up connected sleeping bodies."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body name="b1" pos="0 0 0.5">
            <joint name="j1" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1"/>
          </body>
          <body name="b2" pos="0.5 0 0.5">
            <joint name="j2" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1"/>
          </body>
          <body name="b3" pos="1.0 0 0.5">
            <joint name="j3" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1"/>
          </body>
          <body name="b4" pos="1.5 0 0.5">
            <joint name="j4" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1"/>
          </body>
        </worldbody>
        <tendon>
          <fixed name="ten1">
            <joint joint="j1" coef="1.0"/>
            <joint joint="j2" coef="-1.0"/>
          </fixed>
          <fixed name="ten2">
            <joint joint="j3" coef="1.0"/>
            <joint joint="j4" coef="-1.0"/>
          </fixed>
        </tendon>
        <equality>
          <tendon tendon1="ten1" tendon2="ten2" polycoef="0 1 0 0 0"/>
        </equality>
      </mujoco>
      """,
    )

    # Enable sleep programmatically
    mjm.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP
    m.opt.enableflags |= types.EnableBit.SLEEP

    # Put b3 (tree 2) and b4 (tree 3) to sleep manually
    # b1 is tree 0, b2 is tree 1, b3 is tree 2, b4 is tree 3
    tree_asleep = d.tree_asleep.numpy()
    tree_asleep[0, 2] = 2  # self-cycle for b3
    tree_asleep[0, 3] = 3  # self-cycle for b4
    d.tree_asleep = wp.array(tree_asleep, dtype=int)
    sleep.update_sleep(m, d)

    self.assertEqual(d.tree_awake.numpy()[0, 0], 1, "b1 should start awake")
    self.assertEqual(d.tree_awake.numpy()[0, 1], 1, "b2 should start awake")
    self.assertEqual(d.tree_awake.numpy()[0, 2], 0, "b3 should start asleep")
    self.assertEqual(d.tree_awake.numpy()[0, 3], 0, "b4 should start asleep")

    # Force b1 (tree 0) to move down, which will affect ten1 and couple to ten2
    qvel = d.qvel.numpy()
    qvel[0, 0] = -5.0  # b1 moving down
    d.qvel = wp.array(qvel, dtype=float)

    # Run step
    mjwarp.step(m, d)

    # Both b3 and b4 should have woken up!
    self.assertEqual(d.tree_awake.numpy()[0, 2], 1, "b3 should have woken up due to tendon equality constraint")
    self.assertEqual(d.tree_awake.numpy()[0, 3], 1, "b4 should have woken up due to tendon equality constraint")


if __name__ == "__main__":
  absltest.main()

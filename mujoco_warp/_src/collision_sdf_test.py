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
# ==============================================================================.

"""Tests for SDF collisions, gradient evaluation, and dynamic stability."""

import numpy as np
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import collision_sdf


class CollisionSdfTest(absltest.TestCase):
  def test_sample_volume_sdf_values(self):
    """Tests sample_volume_sdf on a cube octree SDF."""
    _, _, m, _ = test_data.fixture(
      xml="""
      <mujoco>
        <asset>
          <mesh name="cube"
           vertex="1 1 1  1 1 -1  1 -1 1  1 -1 -1  -1 1 1  -1 1 -1  -1 -1 1  -1 -1 -1"/>
        </asset>
        <worldbody>
          <geom type="sdf" mesh="cube"/>
        </worldbody>
      </mujoco>
      """
    )

    @wp.kernel
    def eval_sdf(
      oct_child: wp.array[collision_sdf.vec8i],
      oct_aabb: wp.array2d[wp.vec3],
      oct_coeff: wp.array[collision_sdf.vec8],
      points: wp.array[wp.vec3],
      out_sdf: wp.array[float],
    ):
      tid = wp.tid()
      vd = collision_sdf.VolumeData()
      vd.oct_child = oct_child
      vd.oct_aabb = oct_aabb
      vd.oct_coeff = oct_coeff
      vd.root = 0
      vd.center = oct_aabb[0, 0]
      vd.half_size = oct_aabb[0, 1]
      vd.valid = True
      out_sdf[tid] = collision_sdf.sample_volume_sdf(points[tid], vd)

    test_pts = [
      wp.vec3(0.0, 0.0, 0.0),  # center
      wp.vec3(1.0, 0.0, 0.0),  # surface
      wp.vec3(0.0, 1.0, 0.0),  # surface
      wp.vec3(0.0, 0.0, 1.0),  # surface
      wp.vec3(1.5, 0.0, 0.0),  # exterior
    ]
    pts_wp = wp.array(test_pts, dtype=wp.vec3)
    out_wp = wp.zeros(len(test_pts), dtype=float)
    wp.launch(eval_sdf, dim=len(test_pts), inputs=[m.oct_child, m.oct_aabb, m.oct_coeff, pts_wp, out_wp])
    wp.synchronize()

    vals = out_wp.numpy()
    # Center should be inside (negative distance)
    self.assertLess(vals[0], -0.5)
    # Surface points should have near-zero distance (within octree approximation tolerance)
    self.assertAlmostEqual(vals[1], 0.0, delta=0.08)
    self.assertAlmostEqual(vals[2], 0.0, delta=0.08)
    self.assertAlmostEqual(vals[3], 0.0, delta=0.08)
    # Exterior point should have positive distance
    self.assertGreater(vals[4], 0.3)

  def test_sample_volume_grad_exterior_unit_norm(self):
    """Tests that sample_volume_grad produces unit normal vectors in the exterior domain."""
    _, _, m, _ = test_data.fixture(
      xml="""
      <mujoco>
        <asset>
          <mesh name="cube"
           vertex="1 1 1  1 1 -1  1 -1 1  1 -1 -1  -1 1 1  -1 1 -1  -1 -1 1  -1 -1 -1"/>
        </asset>
        <worldbody>
          <geom type="sdf" mesh="cube"/>
        </worldbody>
      </mujoco>
      """
    )

    @wp.kernel
    def eval_grad(
      oct_child: wp.array[collision_sdf.vec8i],
      oct_aabb: wp.array2d[wp.vec3],
      oct_coeff: wp.array[collision_sdf.vec8],
      points: wp.array[wp.vec3],
      out_grad: wp.array[wp.vec3],
    ):
      tid = wp.tid()
      vd = collision_sdf.VolumeData()
      vd.oct_child = oct_child
      vd.oct_aabb = oct_aabb
      vd.oct_coeff = oct_coeff
      vd.root = 0
      vd.center = oct_aabb[0, 0]
      vd.half_size = oct_aabb[0, 1]
      vd.valid = True
      out_grad[tid] = collision_sdf.sample_volume_grad(points[tid], vd)

    test_pts = [
      wp.vec3(1.5, 0.0, 0.0),
      wp.vec3(0.0, 1.8, 0.0),
      wp.vec3(0.0, 0.0, 2.0),
      wp.vec3(1.2, 1.2, 1.2),
      wp.vec3(-1.5, 0.5, -0.8),
    ]
    pts_wp = wp.array(test_pts, dtype=wp.vec3)
    out_wp = wp.zeros(len(test_pts), dtype=wp.vec3)
    wp.launch(eval_grad, dim=len(test_pts), inputs=[m.oct_child, m.oct_aabb, m.oct_coeff, pts_wp, out_wp])
    wp.synchronize()

    grads = out_wp.numpy()
    for i, g in enumerate(grads):
      norm = np.linalg.norm(g)
      self.assertAlmostEqual(norm, 1.0, delta=1e-3, msg=f"Exterior gradient at {test_pts[i]} not unit norm: {g}")

  def test_sdf_margin_clearance(self):
    """Tests that positive clearance is reported when bodies are separated by margin."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option sdf_iterations="10" sdf_initpoints="40"/>
        <asset>
          <mesh name="cube_mesh" vertex="1 1 1  1 1 -1  1 -1 1  1 -1 -1  -1 1 1  -1 1 -1  -1 -1 1  -1 -1 -1"/>
          <mesh name="cube_sdf" vertex="1 1 1  1 1 -1  1 -1 1  1 -1 -1  -1 1 1  -1 1 -1  -1 -1 1  -1 -1 -1"/>
        </asset>
        <worldbody>
          <body pos="0 0 0">
            <geom type="mesh" mesh="cube_mesh" margin="0.1"/>
          </body>
          <body pos="0 0 2.05">
            <freejoint/>
            <geom type="sdf" mesh="cube_sdf" margin="0.1"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    mjw.collision(m, d)

    nacon = d.nacon.numpy()[0]
    self.assertGreater(nacon, 0, "Expected margin contacts to be generated")
    dists = d.contact.dist.numpy()[:nacon]
    # Separation is 0.05m, margin is 0.10m. All contacts should report
    # strictly positive distance ~0.05m.
    for i, dist in enumerate(dists):
      self.assertGreater(dist, 0.0, f"Contact {i} dist={dist} is not positive clearance")
      self.assertAlmostEqual(dist, 0.05, delta=0.01)

  def test_frank_wolfe_triangle_convergence(self):
    """Tests Frank-Wolfe optimization convergence on a triangle simplex."""

    @wp.func
    def sdf_test_sphere(p: wp.vec3) -> float:
      return wp.length(p) - 0.5

    @wp.func
    def grad_test_sphere(p: wp.vec3) -> wp.vec3:
      return wp.normalize(p)

    @wp.kernel
    def run_fw(out_pt: wp.array[wp.vec3]):
      v0 = wp.vec3(1.0, -0.5, 0.0)
      v1 = wp.vec3(1.0, 0.5, 0.0)
      v2 = wp.vec3(1.0, 0.0, 0.8)
      # Closest point on triangle at x=1 to sphere at origin is (1, 0, 0)
      x = v0 * (1.0 / 3.0) + v1 * (1.0 / 3.0) + v2 * (1.0 / 3.0)
      for k in range(30):
        g = grad_test_sphere(x)
        d0 = wp.dot(v0, g)
        d1 = wp.dot(v1, g)
        d2 = wp.dot(v2, g)
        s = v0
        min_d = d0
        if d1 < min_d:
          s = v1
          min_d = d1
        if d2 < min_d:
          s = v2
        gamma = 2.0 / float(k + 2)
        x = x + (s - x) * gamma
      out_pt[0] = x

    out = wp.zeros(1, dtype=wp.vec3)
    wp.launch(run_fw, dim=1, inputs=[out])
    wp.synchronize()
    pt = out.numpy()[0]
    np.testing.assert_allclose(pt, [1.0, 0.0, 0.0], atol=5e-2)

  def test_mesh_sdf_dynamic_stability(self):
    """Tests dynamic stability over 10 timesteps of a freejoint SDF mesh cube.

    The SDF cube interacts under gravity and contact forces with a mesh cube.
    """
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option sdf_iterations="10" sdf_initpoints="40"/>
        <asset>
          <mesh name="cube_mesh"
           vertex="1 1 1  1 1 -1  1 -1 1  1 -1 -1  -1 1 1  -1 1 -1  -1 -1 1  -1 -1 -1"/>
          <mesh name="cube_sdf"
           vertex="1 1 1  1 1 -1  1 -1 1  1 -1 -1  -1 1 1  -1 1 -1  -1 -1 1  -1 -1 -1"/>
        </asset>
        <worldbody>
          <body pos="0 0 0">
            <geom type="mesh" mesh="cube_mesh"/>
          </body>
          <body pos="0 0 2.2" euler="30 0 0">
            <freejoint/>
            <geom type="sdf" mesh="cube_sdf"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )

    for _ in range(10):
      mjw.step(m, d)
      self.assertGreater(d.nacon.numpy()[0], 0, "Expected collisions during dynamic step")

    wp.synchronize()
    pos = d.qpos.numpy()[0][:3]
    vel = d.qvel.numpy()[0][:3]

    # Check for NaNs or infinities
    self.assertFalse(np.isnan(pos).any(), "NaN found in qpos")
    self.assertFalse(np.isnan(vel).any(), "NaN found in qvel")

    # SDF cube started at z=2.2m and remains supported by contact forces
    self.assertGreater(pos[2], 2.0, f"Cube fell through bottom: z={pos[2]}")
    self.assertLess(pos[2], 2.5, f"Cube launched upward: z={pos[2]}")

    # Velocities should remain bounded (no explosions or launching)
    self.assertLess(np.linalg.norm(vel), 5.0, f"Velocity exploded: |v|={np.linalg.norm(vel)}")

  def test_mesh_sdf_deduplication(self):
    """Tests that Halton candidate contacts are spatially deduplicated."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option sdf_iterations="10" sdf_initpoints="40"/>
        <asset>
          <mesh name="cube_mesh"
           vertex="1 1 1  1 1 -1  1 -1 1  1 -1 -1  -1 1 1  -1 1 -1  -1 -1 1  -1 -1 -1"/>
          <mesh name="cube_sdf"
           vertex="1 1 1  1 1 -1  1 -1 1  1 -1 -1  -1 1 1  -1 1 -1  -1 -1 1  -1 -1 -1"/>
        </asset>
        <worldbody>
          <body pos="0 0 0">
            <geom type="mesh" mesh="cube_mesh"/>
          </body>
          <body pos="0 0 1.95">
            <freejoint/>
            <geom type="sdf" mesh="cube_sdf"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    mjw.collision(m, d)
    nacon = d.nacon.numpy()[0]
    # With 40 init points, without deduplication, contacts explode towards 40.
    # Deduplication ensures we get a bounded set of unique contact points (<= 16).
    self.assertGreater(nacon, 0, "Expected contacts between colliding cubes")
    self.assertLessEqual(nacon, 16, f"Contact explosion: got {nacon} contacts, expected <= 16")

    # Verify that all retained contacts are spatially distinct (not near-duplicates)
    positions = d.contact.pos.numpy()[:nacon]
    for i in range(nacon):
      for j in range(i + 1, nacon):
        dist = np.linalg.norm(positions[i] - positions[j])
        self.assertGreater(dist, 0.05, f"Near-duplicate contacts found between {i} and {j}: dist={dist}")

  def test_mesh_sdf_batched(self):
    """Tests batched multi-world SDF collisions across 4 worlds."""
    nworld = 4
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option sdf_iterations="10" sdf_initpoints="40"/>
        <asset>
          <mesh name="cube_mesh"
           vertex="1 1 1  1 1 -1  1 -1 1  1 -1 -1  -1 1 1  -1 1 -1  -1 -1 1  -1 -1 -1"/>
          <mesh name="cube_sdf"
           vertex="1 1 1  1 1 -1  1 -1 1  1 -1 -1  -1 1 1  -1 1 -1  -1 -1 1  -1 -1 -1"/>
        </asset>
        <worldbody>
          <body pos="0 0 0">
            <geom type="mesh" mesh="cube_mesh"/>
          </body>
          <body pos="0 0 1.95">
            <freejoint/>
            <geom type="sdf" mesh="cube_sdf"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    mjw.collision(m, d)
    nacon = d.nacon.numpy()[0]
    self.assertGreater(nacon, 0, "Expected contacts in batched collision")
    worldids = d.contact.worldid.numpy()[:nacon]
    unique_worlds = np.unique(worldids)
    # Contacts should be detected in each of the 4 worlds
    self.assertEqual(len(unique_worlds), nworld, f"Expected contacts in all {nworld} worlds, got {unique_worlds}")

  def test_mesh_sdf_small_scale(self):
    """Tests SDF collision on sub-centimeter scale geometry (10mm cube)."""
    # Scaled down to 10mm (half-size = 0.005m = 5mm)
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option sdf_iterations="10" sdf_initpoints="40"/>
        <asset>
          <mesh name="small_mesh"
           scale="0.005 0.005 0.005"
           vertex="1 1 1  1 1 -1  1 -1 1  1 -1 -1  -1 1 1  -1 1 -1  -1 -1 1  -1 -1 -1"/>
          <mesh name="small_sdf"
           scale="0.005 0.005 0.005"
           vertex="1 1 1  1 1 -1  1 -1 1  1 -1 -1  -1 1 1  -1 1 -1  -1 -1 1  -1 -1 -1"/>
        </asset>
        <worldbody>
          <body pos="0 0 0">
            <geom type="mesh" mesh="small_mesh"/>
          </body>
          <body pos="0 0 0.0095">
            <freejoint/>
            <geom type="sdf" mesh="small_sdf"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    mjw.collision(m, d)
    nacon = d.nacon.numpy()[0]
    self.assertGreater(nacon, 0, "Expected contacts for small-scale geometry")
    self.assertLessEqual(nacon, 16, f"Contact explosion on small scale: {nacon}")
    dists = d.contact.dist.numpy()[:nacon]
    # Penetration exists and distances are negative or flush
    self.assertLess(np.min(dists), -0.0003, f"Expected penetration distance: {dists}")
    for dist in dists:
      self.assertLessEqual(dist, 1e-6, f"Expected non-positive distance: {dist}")

  def test_sdf_sdf_contact_position_midpoint(self):
    """Tests that SDF-SDF contact position offset has the correct sign along normal."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option sdf_iterations="10" sdf_initpoints="40"/>
        <asset>
          <mesh name="cube1"
           vertex="1 1 1  1 1 -1  1 -1 1  1 -1 -1  -1 1 1  -1 1 -1  -1 -1 1  -1 -1 -1"/>
          <mesh name="cube2"
           vertex="1 1 1  1 1 -1  1 -1 1  1 -1 -1  -1 1 1  -1 1 -1  -1 -1 1  -1 -1 -1"/>
        </asset>
        <worldbody>
          <body pos="0 0 0">
            <geom type="sdf" mesh="cube1"/>
          </body>
          <body pos="0 0 1.9">
            <freejoint/>
            <geom type="sdf" mesh="cube2"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    mjw.collision(m, d)

    nacon = d.nacon.numpy()[0]
    self.assertGreater(nacon, 0, "Expected contacts between colliding SDF cubes")

    # Penetration along Z: normal points in +Z from geom 1 to geom 2.
    dists = d.contact.dist.numpy()[:nacon]
    normals = d.contact.frame.numpy()[:nacon, 0]
    for dist in dists:
      self.assertLess(dist, 0.0, f"Expected penetration dist < 0, got {dist}")
    for n in normals:
      self.assertGreater(n[2], 0.9, f"Expected normal pointing along +Z, got {n}")

    # Contact position offset must follow: pos + normal * (0.5 * dist).
    # Since dist < 0 and normal points along +Z, contact position must be offset
    # in the -Z direction (towards geom 1), so mean Z must be <= 0.95.
    positions = d.contact.pos.numpy()[:nacon]
    mean_z = float(np.mean(positions[:, 2]))
    self.assertLessEqual(mean_z, 0.95, f"Contact pos Z shifted in wrong direction: {mean_z} > 0.95")

  def test_grad_box(self):
    """Tests that grad_box handles face centers and axes without producing NaNs."""

    @wp.kernel
    def eval_grad_box(
      points: wp.array[wp.vec3],
      size: wp.vec3,
      out_grad: wp.array[wp.vec3],
    ):
      tid = wp.tid()
      out_grad[tid] = collision_sdf.grad_box(points[tid], size)

    test_pts = [
      wp.vec3(0.0, 0.0, 0.0),  # origin
      wp.vec3(1.0, 0.0, 0.0),  # boundary face center (+X)
      wp.vec3(1.05, 0.0, 0.0),  # exterior along +X
      wp.vec3(0.0, 1.05, 0.0),  # exterior along +Y
      wp.vec3(0.0, 0.0, 1.05),  # exterior along +Z
      wp.vec3(1.0, 1.0, 0.0),  # boundary edge
      wp.vec3(1.0, 1.0, 1.0),  # boundary corner
      wp.vec3(2.0, 0.0, 0.0),  # exterior along +X
    ]
    size = wp.vec3(1.0, 1.0, 1.0)
    pts_wp = wp.array(test_pts, dtype=wp.vec3)
    out_wp = wp.zeros(len(test_pts), dtype=wp.vec3)
    wp.launch(eval_grad_box, dim=len(test_pts), inputs=[pts_wp, size, out_wp])
    wp.synchronize()

    grads = out_wp.numpy()
    self.assertFalse(np.isnan(grads).any(), f"NaN in box gradients: {grads}")
    # Exterior points along axes should point along outward face normal
    np.testing.assert_allclose(grads[2], [1.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(grads[3], [0.0, 1.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(grads[4], [0.0, 0.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(grads[7], [1.0, 0.0, 0.0], atol=1e-5)

  def test_grad_ellipsoid(self):
    """Tests that grad_ellipsoid handles singular points without producing NaNs."""

    @wp.kernel
    def eval_grad_ellipsoid(
      points: wp.array[wp.vec3],
      size: wp.vec3,
      out_grad: wp.array[wp.vec3],
    ):
      tid = wp.tid()
      out_grad[tid] = collision_sdf.grad_ellipsoid(points[tid], size)

    test_pts = [
      wp.vec3(0.0, 0.0, 0.0),  # origin (singular)
      wp.vec3(2.0, 0.0, 0.0),  # exterior along X
      wp.vec3(0.0, 3.0, 0.0),  # exterior along Y
      wp.vec3(0.0, 0.0, 1.5),  # exterior along Z
    ]
    size = wp.vec3(1.0, 2.0, 1.0)
    pts_wp = wp.array(test_pts, dtype=wp.vec3)
    out_wp = wp.zeros(len(test_pts), dtype=wp.vec3)
    wp.launch(eval_grad_ellipsoid, dim=len(test_pts), inputs=[pts_wp, size, out_wp])
    wp.synchronize()

    grads = out_wp.numpy()
    self.assertFalse(np.isnan(grads).any(), f"NaN in ellipsoid gradients: {grads}")
    np.testing.assert_allclose(grads[1], [1.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(grads[2], [0.0, 1.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(grads[3], [0.0, 0.0, 1.0], atol=1e-5)


if __name__ == "__main__":
  absltest.main()

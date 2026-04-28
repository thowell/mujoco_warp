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
"""Solver utility functions.

Mirrors MuJoCo C's engine_util_solve: QCQP solvers for elliptic friction cone
projection.
"""

import warp as wp


@wp.func
def qcqp2(
  # In:
  A00: float,
  A01: float,
  A11: float,
  b0: float,
  b1: float,
  d0: float,
  d1: float,
  r: float,
):
  """Solve 2D QCQP: min 0.5*x'*A*x + x'*b s.t. sum(xi/di)^2 <= r^2.

  Returns (v0, v1, active) where active=1 if constraint is active.
  """
  # Scale A,b so constraint becomes x'*x <= r*r
  sb0 = b0 * d0
  sb1 = b1 * d1
  sA00 = A00 * d0 * d0
  sA11 = A11 * d1 * d1
  sA01 = A01 * d0 * d1

  # Newton iteration on dual variable lambda
  la = float(0.0)
  v0 = float(0.0)
  v1 = float(0.0)

  for _iter in range(20):
    # det(A + la*I)
    det = (sA00 + la) * (sA11 + la) - sA01 * sA01

    if det < 1e-10:
      return v0, v1, 0

    detinv = 1.0 / det
    P00 = (sA11 + la) * detinv
    P11 = (sA00 + la) * detinv
    P01 = -sA01 * detinv

    # v = -P*b
    v0 = -P00 * sb0 - P01 * sb1
    v1 = -P01 * sb0 - P11 * sb1

    # val = v'*v - r*r
    val = v0 * v0 + v1 * v1 - r * r

    if val < 1e-10:
      break

    # deriv = -2 * v' * P * v
    deriv = -2.0 * (P00 * v0 * v0 + 2.0 * P01 * v0 * v1 + P11 * v1 * v1)

    delta = -val / deriv
    if delta < 1e-10:
      break

    la += delta

  # Undo scaling
  v0 = v0 * d0
  v1 = v1 * d1

  active = 0
  if la != 0.0:
    active = 1
  return v0, v1, active


@wp.func
def qcqp3(
  # In:
  A00: float,
  A01: float,
  A02: float,
  A11: float,
  A12: float,
  A22: float,
  b0: float,
  b1: float,
  b2: float,
  d0: float,
  d1: float,
  d2: float,
  r: float,
):
  """Solve 3D QCQP: min 0.5*x'*A*x + x'*b s.t. sum(xi/di)^2 <= r^2.

  Returns (v0, v1, v2, active).
  """
  # Scale A,b
  sb0 = b0 * d0
  sb1 = b1 * d1
  sb2 = b2 * d2
  sA00 = A00 * d0 * d0
  sA11 = A11 * d1 * d1
  sA22 = A22 * d2 * d2
  sA01 = A01 * d0 * d1
  sA02 = A02 * d0 * d2
  sA12 = A12 * d1 * d2

  la = float(0.0)
  v0 = float(0.0)
  v1 = float(0.0)
  v2 = float(0.0)

  for _iter in range(20):
    # Cofactors (unscaled P)
    P00 = (sA11 + la) * (sA22 + la) - sA12 * sA12
    P11 = (sA00 + la) * (sA22 + la) - sA02 * sA02
    P22 = (sA00 + la) * (sA11 + la) - sA01 * sA01
    P01 = sA02 * sA12 - sA01 * (sA22 + la)
    P02 = sA01 * sA12 - sA02 * (sA11 + la)
    P12 = sA01 * sA02 - sA12 * (sA00 + la)

    det = (sA00 + la) * P00 + sA01 * P01 + sA02 * P02

    if det < 1e-10:
      return v0, v1, v2, 0

    detinv = 1.0 / det
    P00 *= detinv
    P11 *= detinv
    P22 *= detinv
    P01 *= detinv
    P02 *= detinv
    P12 *= detinv

    v0 = -P00 * sb0 - P01 * sb1 - P02 * sb2
    v1 = -P01 * sb0 - P11 * sb1 - P12 * sb2
    v2 = -P02 * sb0 - P12 * sb1 - P22 * sb2

    val = v0 * v0 + v1 * v1 + v2 * v2 - r * r

    if val < 1e-10:
      break

    deriv = -2.0 * (P00 * v0 * v0 + P11 * v1 * v1 + P22 * v2 * v2) - 4.0 * (P01 * v0 * v1 + P02 * v0 * v2 + P12 * v1 * v2)

    delta = -val / deriv
    if delta < 1e-10:
      break

    la += delta

  v0 = v0 * d0
  v1 = v1 * d1
  v2 = v2 * d2

  active = 0
  if la != 0.0:
    active = 1
  return v0, v1, v2, active


@wp.func
def qcqp5(
  # In:
  A00: float,
  A01: float,
  A02: float,
  A03: float,
  A04: float,
  A11: float,
  A12: float,
  A13: float,
  A14: float,
  A22: float,
  A23: float,
  A24: float,
  A33: float,
  A34: float,
  A44: float,
  # 5-vector
  b0: float,
  b1: float,
  b2: float,
  b3: float,
  b4: float,
  # 5 scaling values
  d0: float,
  d1: float,
  d2: float,
  d3: float,
  d4: float,
  # constraint radius
  r: float,
):
  """Solve 5D QCQP via Newton on dual variable with explicit Cholesky."""
  # Scale A, b
  sb0 = b0 * d0
  sb1 = b1 * d1
  sb2 = b2 * d2
  sb3 = b3 * d3
  sb4 = b4 * d4
  sA00 = A00 * d0 * d0
  sA01 = A01 * d0 * d1
  sA02 = A02 * d0 * d2
  sA03 = A03 * d0 * d3
  sA04 = A04 * d0 * d4
  sA11 = A11 * d1 * d1
  sA12 = A12 * d1 * d2
  sA13 = A13 * d1 * d3
  sA14 = A14 * d1 * d4
  sA22 = A22 * d2 * d2
  sA23 = A23 * d2 * d3
  sA24 = A24 * d2 * d4
  sA33 = A33 * d3 * d3
  sA34 = A34 * d3 * d4
  sA44 = A44 * d4 * d4

  la = float(0.0)
  v0_out = float(0.0)
  v1_out = float(0.0)
  v2_out = float(0.0)
  v3_out = float(0.0)
  v4_out = float(0.0)

  for _iter in range(20):
    # M = sA + la*I
    M00 = sA00 + la
    M11 = sA11 + la
    M22 = sA22 + la
    M33 = sA33 + la
    M44 = sA44 + la

    # Cholesky: L*L' = M (lower triangle)
    # Row 0
    if M00 < 1e-10:
      return v0_out, v1_out, v2_out, v3_out, v4_out, 0
    L00 = wp.sqrt(M00)
    L10 = sA01 / L00
    L20 = sA02 / L00
    L30 = sA03 / L00
    L40 = sA04 / L00

    # Row 1
    s1 = M11 - L10 * L10
    if s1 < 1e-10:
      return v0_out, v1_out, v2_out, v3_out, v4_out, 0
    L11 = wp.sqrt(s1)
    L21 = (sA12 - L20 * L10) / L11
    L31 = (sA13 - L30 * L10) / L11
    L41 = (sA14 - L40 * L10) / L11

    # Row 2
    s2 = M22 - L20 * L20 - L21 * L21
    if s2 < 1e-10:
      return v0_out, v1_out, v2_out, v3_out, v4_out, 0
    L22 = wp.sqrt(s2)
    L32 = (sA23 - L30 * L20 - L31 * L21) / L22
    L42 = (sA24 - L40 * L20 - L41 * L21) / L22

    # Row 3
    s3 = M33 - L30 * L30 - L31 * L31 - L32 * L32
    if s3 < 1e-10:
      return v0_out, v1_out, v2_out, v3_out, v4_out, 0
    L33 = wp.sqrt(s3)
    L43 = (sA34 - L40 * L30 - L41 * L31 - L42 * L32) / L33

    # Row 4
    s4 = M44 - L40 * L40 - L41 * L41 - L42 * L42 - L43 * L43
    if s4 < 1e-10:
      return v0_out, v1_out, v2_out, v3_out, v4_out, 0
    L44 = wp.sqrt(s4)

    # Forward solve: L * y = -sb
    y0 = -sb0 / L00
    y1 = (-sb1 - L10 * y0) / L11
    y2 = (-sb2 - L20 * y0 - L21 * y1) / L22
    y3 = (-sb3 - L30 * y0 - L31 * y1 - L32 * y2) / L33
    y4 = (-sb4 - L40 * y0 - L41 * y1 - L42 * y2 - L43 * y3) / L44

    # Back solve: L' * v = y
    v4_out = y4 / L44
    v3_out = (y3 - L43 * v4_out) / L33
    v2_out = (y2 - L32 * v3_out - L42 * v4_out) / L22
    v1_out = (y1 - L21 * v2_out - L31 * v3_out - L41 * v4_out) / L11
    v0_out = (y0 - L10 * v1_out - L20 * v2_out - L30 * v3_out - L40 * v4_out) / L00

    # val = v'*v - r*r
    val = v0_out * v0_out + v1_out * v1_out + v2_out * v2_out + v3_out * v3_out + v4_out * v4_out - r * r

    if val < 1e-10:
      break

    # For derivative: solve L L' tmp = v
    t0 = v0_out / L00
    t1 = (v1_out - L10 * t0) / L11
    t2 = (v2_out - L20 * t0 - L21 * t1) / L22
    t3 = (v3_out - L30 * t0 - L31 * t1 - L32 * t2) / L33
    t4 = (v4_out - L40 * t0 - L41 * t1 - L42 * t2 - L43 * t3) / L44

    u4 = t4 / L44
    u3 = (t3 - L43 * u4) / L33
    u2 = (t2 - L32 * u3 - L42 * u4) / L22
    u1 = (t1 - L21 * u2 - L31 * u3 - L41 * u4) / L11
    u0 = (t0 - L10 * u1 - L20 * u2 - L30 * u3 - L40 * u4) / L00

    # deriv = -2 * v' * u
    deriv = -2.0 * (v0_out * u0 + v1_out * u1 + v2_out * u2 + v3_out * u3 + v4_out * u4)

    delta = -val / deriv
    if delta < 1e-10:
      break

    la += delta

  # Undo scaling
  v0_out *= d0
  v1_out *= d1
  v2_out *= d2
  v3_out *= d3
  v4_out *= d4

  active = 0
  if la != 0.0:
    active = 1
  return v0_out, v1_out, v2_out, v3_out, v4_out, active

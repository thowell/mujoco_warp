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

"""Shared utilities and flags for MJWarp CLI tools."""

import time
from typing import Callable, Tuple, get_type_hints

import mujoco
import numpy as np
import warp as wp
from absl import flags
from etils import epath

import mujoco_warp as mjw
from mujoco_warp._src import warp_util
from mujoco_warp._src.io import load_trajectory
from mujoco_warp._src.io import override_model
from mujoco_warp._src.util_misc import halton

# shared flags for cli tool
NWORLD = flags.DEFINE_string("nworld", "8192", "number of parallel rollouts (comma-separated list for multi-gpu)")


def parse_nworld(val: str) -> list[int] | int:
  if "," in val:
    return [int(x) for x in val.split(",")]
  return int(val)


NSTEP = flags.DEFINE_integer("nstep", 1000, "number of steps per rollout")
NCONMAX = flags.DEFINE_integer("nconmax", None, "override maximum number of contacts per world")
NJMAX = flags.DEFINE_integer("njmax", None, "override maximum number of constraints per world")
NJMAX_NNZ = flags.DEFINE_integer("njmax_nnz", None, "override maximum number of non-zeros in constraint Jacobian")
NCCDMAX = flags.DEFINE_integer("nccdmax", None, "override maximum number of CCD contacts per world")
OVERRIDE = flags.DEFINE_multi_string("override", [], "Model overrides (notation: foo.bar = baz)", short_name="o")
KEYFRAME = flags.DEFINE_integer("keyframe", 0, "keyframe to initialize simulation.")
EVENT_TRACE = flags.DEFINE_bool("event_trace", False, "print an event trace report")
NOISE_STD = flags.DEFINE_float("noise_std", 0.01, "add noise to ctrl signal (standard deviation)")
NOISE_RATE = flags.DEFINE_float("noise_rate", 0.1, "add noise to ctrl signal (noise rate)")
NVMAX = flags.DEFINE_integer("nvmax", None, "maximum active DOFs per world")


INIT_ASLEEP = flags.DEFINE_bool(
  "init_asleep", False, "initialize all trees as asleep before simulation (requires sleep enabled)"
)

DEVICE = flags.DEFINE_string("device", None, "override the default Warp device")
REPLAY = flags.DEFINE_string("replay", None, "NPZ file with ctrl sequence to replay")

RENDER_WIDTH = flags.DEFINE_integer("render_width", 64, "render width (pixels)")
RENDER_HEIGHT = flags.DEFINE_integer("render_height", 64, "render height (pixels)")
RENDER_RGB = flags.DEFINE_bool("render_rgb", True, "render RGB image")
RENDER_DEPTH = flags.DEFINE_bool("render_depth", True, "render depth image")
RENDER_TEXTURES = flags.DEFINE_bool("render_textures", True, "use textures")
RENDER_SHADOWS = flags.DEFINE_bool("render_shadows", False, "use shadows")
RENDER_BACKFACE_CULLING = flags.DEFINE_bool(
  "render_backface_culling",
  True,
  "enable renderer backface culling (RenderContext.enable_backface_culling)",
)
RENDER_SKYBOX = flags.DEFINE_bool("render_skybox", True, "render skybox texture if available")


def load_model(path: epath.Path) -> mujoco.MjModel:
  """Load a MuJoCo model from a path, handling resources and plugins."""
  if not path.exists():
    resource_path = epath.resource_path("mujoco_warp") / path
    if not resource_path.exists():
      raise FileNotFoundError(f"file not found: {path}\nalso tried: {resource_path}")
    path = resource_path

  if path.suffix == ".mjb":
    return mujoco.MjModel.from_binary_path(path.as_posix())

  spec = mujoco.MjSpec.from_file(path.as_posix())
  if any(p.plugin_name.startswith("mujoco.sdf") for p in spec.plugins):
    from mujoco_warp.test_data.collision_sdf.utils import register_sdf_plugins as register_sdf_plugins

    register_sdf_plugins(mjw)

  mjm = spec.compile()

  if OVERRIDE.value:
    override_model(mjm, OVERRIDE.value)

  return mjm


@wp.kernel
def _ctrl_noise(
  # Model:
  opt_timestep: wp.array[float],
  actuator_ctrllimited: wp.array[bool],
  actuator_ctrlrange: wp.array2d[wp.vec2],
  # Data in:
  ctrl_in: wp.array2d[float],
  # In:
  ctrl_center: wp.array[float],
  step: int,
  ctrlnoisestd: float,
  ctrlnoiserate: float,
  world_offset: int,
  # Data out:
  ctrl_out: wp.array2d[float],
):
  worldid_local, actid = wp.tid()
  worldid_global = worldid_local + world_offset

  # convert rate and scale to discrete time (Ornstein-Uhlenbeck)
  rate = wp.exp(-opt_timestep[worldid_local % opt_timestep.shape[0]] / ctrlnoiserate)
  scale = ctrlnoisestd * wp.sqrt(1.0 - rate * rate)

  midpoint = 0.0
  halfrange = 1.0
  ctrlrange = actuator_ctrlrange[worldid_local % actuator_ctrlrange.shape[0], actid]
  is_limited = actuator_ctrllimited[actid]
  if is_limited:
    midpoint = 0.5 * (ctrlrange[1] + ctrlrange[0])
    halfrange = 0.5 * (ctrlrange[1] - ctrlrange[0])
  if ctrl_center.shape[0] > 0:
    midpoint = ctrl_center[actid]

  # exponential convergence to midpoint at ctrlnoiserate
  ctrl = rate * ctrl_in[worldid_local, actid] + (1.0 - rate) * midpoint

  # add noise (use global world ID)
  ctrl += scale * halfrange * (2.0 * halton((step + 1) * (worldid_global + 1), actid + 2) - 1.0)

  # clip to range if limited
  if is_limited:
    ctrl = wp.clamp(ctrl, ctrlrange[0], ctrlrange[1])

  ctrl_out[worldid_local, actid] = ctrl


def init_structs(
  fn: Callable[..., None],
  mjm: mujoco.MjModel,
  device: str | wp.Device | None = None,
  nworld: int | None = None,
) -> Tuple[mjw.Model, mjw.Data, mjw.RenderContext | None, list[np.ndarray] | None]:
  """Initialize device structs."""
  mjd = mujoco.MjData(mjm)
  ctrls = None
  if REPLAY.value:
    ctrls = load_trajectory(REPLAY.value, mjm, mjd)
    # default nstep to trajectory length when not explicitly set
    if flags.FLAGS["nstep"].using_default_value:
      flags.FLAGS.nstep = len(ctrls)
  elif mjm.nkey > 0 and KEYFRAME.value > -1:
    mujoco.mj_resetDataKeyframe(mjm, mjd, KEYFRAME.value)
    ctrls = [mjd.ctrl.copy() for _ in range(NSTEP.value)]

  device = wp.get_device(device or DEVICE.value)
  if nworld is None:
    parsed = parse_nworld(NWORLD.value)
    if isinstance(parsed, list):
      raise ValueError("nworld list configuration must be resolved to a single integer for init_structs")
    nworld = parsed

  with wp.ScopedDevice(device):
    m = mjw.put_model(mjm)
    if OVERRIDE.value:
      override_model(m, OVERRIDE.value)
    if INIT_ASLEEP.value:
      mjd.tree_asleep[:] = np.arange(mjm.ntree, dtype=np.int32)

    d = mjw.put_data(
      mjm,
      mjd,
      nworld=nworld,
      nconmax=NCONMAX.value,
      njmax=NJMAX.value,
      njmax_nnz=NJMAX_NNZ.value,
      nccdmax=NCCDMAX.value,
      nvmax=NVMAX.value,
    )

    if mjw.RenderContext not in get_type_hints(fn).values():
      return m, d, None, ctrls

    rc = mjw.create_render_context(
      mjm,
      nworld=nworld,
      cam_res=(RENDER_WIDTH.value, RENDER_HEIGHT.value),
      render_rgb=RENDER_RGB.value,
      render_depth=RENDER_DEPTH.value,
      use_textures=RENDER_TEXTURES.value,
      use_shadows=RENDER_SHADOWS.value,
      enable_backface_culling=RENDER_BACKFACE_CULLING.value,
      render_skybox=RENDER_SKYBOX.value,
    )

    return m, d, rc, ctrls


def unroll(
  fn: Callable[..., None],
  m: mjw.Model,
  d: mjw.Data,
  rc: mjw.RenderContext | None,
  callback: Callable[[int, dict, float], None] | None = None,
  ctrls: list[np.ndarray] | None = None,
) -> float:
  """Unroll a function on batched Data and return some statistics.

  Args:
    fn: Function to unroll (e.g. mjw.step).
    m: Model.
    d: Data.
    rc: Render context (optional).
    callback: Optional callback called after each step with (step count, trace, latency).
    ctrls: Optional control trajectory.

  Returns:
    jit_duration: Time to JIT capture the function.
  """
  device = wp.get_device(DEVICE.value)
  device_str = str(device)
  return unroll_multigpu(
    fn=fn,
    models={device_str: m},
    datas={device_str: d},
    rcs={device_str: rc},
    devices=[device],
    callback=callback,
    ctrls=ctrls,
  )


def _sum_trace(stack1, stack2):
  """Recursively sum event trace stacks."""
  ret = {}

  for k in stack1.keys() | stack2.keys():
    if k not in stack1:
      ret[k] = stack2[k]
    elif k not in stack2:
      ret[k] = stack1[k]
    else:
      times1, sub_stack1 = stack1[k]
      times2, sub_stack2 = stack2[k]
      times = [t1 + t2 for t1, t2 in zip(times1, times2)]
      ret[k] = (times, _sum_trace(sub_stack1, sub_stack2))

  return ret


def unroll_multigpu(
  fn: Callable[..., None],
  models: dict[str, mjw.Model],
  datas: dict[str, mjw.Data],
  rcs: dict[str, mjw.RenderContext | None],
  devices: list[wp.Device],
  callback: Callable[[int, dict, float], None] | None = None,
  ctrls: list[np.ndarray] | None = None,
) -> float:
  """Unroll a function on multiple GPUs and return JIT compilation time."""
  device_strs = [str(device) for device in devices]
  # Calculate world offsets for noise invariance
  offsets = {}
  current_offset = 0
  for idx, device in enumerate(devices):
    device_str = device_strs[idx]
    offsets[device_str] = current_offset
    current_offset += datas[device_str].nworld

  # Capture graphs on each device
  jit_beg = time.perf_counter()
  captures = {}
  tracers = {}

  for idx, device in enumerate(devices):
    device_str = device_strs[idx]
    with wp.ScopedDevice(device):
      tracers[device_str] = warp_util.EventTracer(device=device, enabled=EVENT_TRACE.value)
      with tracers[device_str]:
        with wp.ScopedCapture(device) as capture:
          m = models[device_str]
          d = datas[device_str]
          rc = rcs[device_str]
          if rc is not None:
            fn(m, d, rc)
          else:
            fn(m, d)
        captures[device_str] = capture.graph
  jit_end = time.perf_counter()

  # Main rollout loop
  for i in range(NSTEP.value):
    # 1. Launch noise on all devices
    if ctrls is not None:
      for idx, device in enumerate(devices):
        device_str = device_strs[idx]
        with wp.ScopedDevice(device):
          m = models[device_str]
          d = datas[device_str]
          center = wp.array(ctrls[i], dtype=wp.float32, device=device)
          wp.launch(
            _ctrl_noise,
            dim=(d.nworld, m.nu),
            inputs=[
              m.opt.timestep,
              m.actuator_ctrllimited,
              m.actuator_ctrlrange,
              d.ctrl,
              center,
              i,
              NOISE_STD.value,
              NOISE_RATE.value,
              offsets[device_str],
            ],
            outputs=[d.ctrl],
            device=device,
          )
      # Synchronize noise launch on all devices before starting physics timer
      for device in devices:
        wp.synchronize_device(device)

    step_beg = time.perf_counter()

    # 2. Launch graphs on all devices
    for idx, device in enumerate(devices):
      device_str = device_strs[idx]
      with wp.ScopedDevice(device):
        wp.capture_launch(captures[device_str])

    # 3. Synchronize all devices
    for device in devices:
      wp.synchronize_device(device)

    step_end = time.perf_counter()
    latency = step_end - step_beg

    if callback:
      # Aggregate traces from all devices
      step_trace = {}
      if EVENT_TRACE.value:
        for idx, device in enumerate(devices):
          device_str = device_strs[idx]
          step_trace = _sum_trace(step_trace, tracers[device_str].trace())
      callback(i, step_trace, latency)

  return jit_end - jit_beg

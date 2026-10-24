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

"""Compare mixed and separate cube lifts using per-world tree sleeping (PR #1680).

uv run python benchmarks/heterogeneous_cube_lift/benchmark.py --assets /path/to/menagerie

Each case advances 8192 worlds of each robot through the same five-second replay.
Inactive robot and cube trees use ALWAYS; active trees use AUTO_NEVER. Separate
models stay resident and advance on the same stream. Setup and reset are untimed.
"""

import argparse
import dataclasses
import hashlib
import json
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw
from mujoco_warp._src import sleep
from mujoco_warp._src.types import OverflowType
from mujoco_warp._src.types import SleepPolicy

ROOT = Path(__file__).resolve().parents[2]
KINDS = ("franka", "aloha", "so101")
MENAGERIE_REF = "affef0836947b64cc06c4ab1cbf0152835693374"


@wp.kernel
def replay_control(sequence: wp.array2d[float], index: wp.array[int], ctrl_out: wp.array2d[float]):
  world, actuator = wp.tid()
  ctrl_out[world, actuator] = sequence[wp.min(index[0], sequence.shape[0] - 1), actuator]


@wp.kernel
def advance(index_out: wp.array[int]):
  index_out[0] += 1


def model(kind, assets, sleeping=True):
  spec = mujoco.MjSpec.from_string('<mujoco><worldbody><geom name="floor" type="plane" size="0 0 .05"/></worldbody></mujoco>')
  spec.option.timestep = 0.002
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  spec.option.impratio = 10
  spec.option.iterations = 50
  spec.option.ls_iterations = 40
  spec.option.enableflags = int(mujoco.mjtEnableBit.mjENBL_SLEEP) if sleeping else 0
  initial = {}
  for name in KINDS if kind == "mixed" else (kind,):
    prefix = name + "/"
    child = mujoco.MjSpec.from_file(str(Path(__file__).parent / f"{name}.xml"))
    folder = {"franka": "franka_emika_panda", "aloha": "aloha", "so101": "robotstudio_so101"}[name]
    for mesh in child.meshes:
      mesh.file = str(assets / folder / "assets" / Path(mesh.file).name)
    child.delete(child.geom("floor"))
    cm = child.compile()
    replay = np.load(Path(__file__).parent / f"{name}.npz")
    qpos = replay["qpos"][0].copy()
    offset = 1.5 * KINDS.index(name)
    qpos[cm.joint("cube_joint").qposadr[0] + 1] += offset
    initial[prefix] = cm, qpos, replay["ctrl"]
    for key in list(child.keys):
      child.delete(key)
    spec.attach(child, prefix=prefix, frame=spec.worldbody.add_frame(pos=[0, offset, 0]))
  mjm = spec.compile()
  mjd = mujoco.MjData(mjm)
  controls = np.zeros((2500, mjm.nu), dtype=np.float32)
  for prefix, (cm, qpos, ctrl) in initial.items():
    for j in range(cm.njnt):
      dest = mjm.joint(prefix + cm.joint(j).name)
      start = cm.jnt_qposadr[j]
      end = cm.jnt_qposadr[j + 1] if j + 1 < cm.njnt else cm.nq
      mjd.qpos[dest.qposadr[0] : dest.qposadr[0] + end - start] = qpos[start:end]
    for a in range(cm.nu):
      controls[:, mjm.actuator(prefix + cm.actuator(a).name).id] = ctrl[:, a]
  mjd.ctrl[:] = controls[0]
  return mjm, mjd, controls


def arrays_bytes(value, seen=None):
  """Count unique Warp array storage, including nested model/data structures."""
  seen = set() if seen is None else seen
  if isinstance(value, wp.array):
    if value.ptr in seen:
      return 0
    seen.add(value.ptr)
    return value.capacity
  if dataclasses.is_dataclass(value):
    return sum(arrays_bytes(getattr(value, f.name), seen) for f in dataclasses.fields(value))
  return 0


def create(kind, args):
  sleeping = args.case != "separate_no_sleep"
  mjm, mjd, controls = model(kind, args.assets, sleeping)
  nworld = args.worlds * (3 if kind == "mixed" else 1)
  m = mjw.put_model(mjm, batch_sizes={"tree_sleep_policy": nworld})
  policy = np.full((nworld, mjm.ntree), int(SleepPolicy.AUTO_NEVER), dtype=np.int32)
  if kind == "mixed":
    for group, active in enumerate(KINDS):
      for tree in range(mjm.ntree):
        body = np.flatnonzero(mjm.body_treeid == tree)[0]
        if not mjm.body(body).name.startswith(active + "/"):
          policy[group * args.worlds : (group + 1) * args.worlds, tree] = int(SleepPolicy.ALWAYS)
  m.tree_sleep_policy.assign(policy)
  # Capacities are identical per world except nvmax, which reflects each layout's active DOFs.
  nvmax = int(np.max((policy != int(SleepPolicy.ALWAYS)) @ mjm.tree_dofnum))
  d = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=64, nccdmax=64, njmax=256, nvmax=nvmax if sleeping else None)
  state = (kind, mjm, mjd, m, d, policy)
  reset(state)
  return state, wp.array(controls, dtype=float)


def reset(state):
  _, mjm, mjd, m, d, policy = state
  mjw.reset_data(m, d)
  d.qpos.assign(np.tile(mjd.qpos, (d.nworld, 1)).astype(np.float32))
  d.ctrl.assign(np.tile(mjd.ctrl, (d.nworld, 1)).astype(np.float32))
  eq_active = np.tile(mjd.eq_active, (d.nworld, 1))
  for eq in range(mjm.neq):
    if mjm.eq_type[eq] != mujoco.mjtEq.mjEQ_JOINT:
      raise ValueError("Expected finger joint equality constraints")
    for joint in (mjm.eq_obj1id[eq], mjm.eq_obj2id[eq]):
      if joint >= 0:
        tree = mjm.body_treeid[mjm.jnt_bodyid[joint]]
        eq_active[:, eq] &= policy[:, tree] != int(SleepPolicy.ALWAYS)
  d.eq_active.assign(eq_active)
  if m.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP):
    sleep.update_sleep(m, d)


def diagnostics(state):
  kind, mjm, _, _, d, policy = state
  overflow = d.overflow.numpy()
  capacity_mask = int(OverflowType.ALL)
  if np.any(overflow & capacity_mask):
    raise RuntimeError(f"{kind}: overflow: {np.unique(overflow).tolist()}")
  qvel = d.qvel.numpy()
  if not np.isfinite(d.qpos.numpy()).all() or not np.isfinite(qvel).all():
    raise RuntimeError(f"{kind}: nonfinite state")
  inactive = policy[:, mjm.body_treeid[mjm.dof_bodyid]] == int(SleepPolicy.ALWAYS)
  if np.any(qvel[inactive] != 0):
    raise RuntimeError(f"{kind}: inactive tree moved")
  awake = d.tree_awake.numpy()
  # tree_awake is a per-tree flag, not the compacted tree list.
  expected = policy != int(SleepPolicy.ALWAYS)
  if state[3].opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP) and not np.array_equal(awake > 0, expected):
    raise RuntimeError(f"{kind}: unexpected awake trees: {awake[0]}, expected {expected[0]}")
  return {
    "kind": kind,
    "nworld": d.nworld,
    "nq": mjm.nq,
    "nv": mjm.nv,
    "ntree": mjm.ntree,
    "nvmax": d.nvmax,
    "contacts_total": int(d.nacon.numpy()[0]),
    "constraints_max": int(d.nefc.numpy().max()),
    "solver_iterations_mean": float(d.solver_niter.numpy().mean()),
    "overflow_values": np.unique(overflow).tolist(),
  }


def snapshot(states, storage):
  for kind, mjm, _, _, d, _ in states:
    groups = KINDS if kind == "mixed" else (kind,)
    qpos, qvel = d.qpos.numpy(), d.qvel.numpy()
    for group, name in enumerate(groups):
      count = d.nworld // 3 if kind == "mixed" else d.nworld
      row = group * count if kind == "mixed" else 0
      for field, array, addresses, total in (("qpos", qpos, mjm.jnt_qposadr, mjm.nq), ("qvel", qvel, mjm.jnt_dofadr, mjm.nv)):
        indices = []
        for j in range(mjm.njnt):
          if mjm.joint(j).name.startswith(name + "/"):
            indices.extend(range(addresses[j], addresses[j + 1] if j + 1 < mjm.njnt else total))
        values = array[row : row + count, indices]
        storage.setdefault(f"{name}_{field}_spread", []).append(np.max(np.abs(values - values[0]), axis=0))
        storage.setdefault(f"{name}_{field}", []).append(values[0].copy())


def worker(args):
  wp.init()
  wp.set_device(args.device)
  wp.config.log_level = wp.LOG_WARNING
  if not wp.get_device().is_cuda:
    raise ValueError("This benchmark requires CUDA graph execution")
  free_start = wp.get_device().free_memory
  start = time.perf_counter()
  created = [create(k, args) for k in (KINDS if args.case.startswith("separate") else ("mixed",))]
  states = [s for s, _ in created]
  index = wp.zeros(1, dtype=int)
  wp.synchronize()
  setup_s = time.perf_counter() - start
  start = time.perf_counter()
  for state in states:
    mjw.step(state[3], state[4])
  wp.synchronize()
  with wp.ScopedCapture() as capture:
    for state, controls in created:
      wp.launch(replay_control, (state[4].nworld, state[1].nu), inputs=[controls, index], outputs=[state[4].ctrl])
      mjw.step(state[3], state[4])
    wp.launch(advance, 1, outputs=[index])
  for _ in range(args.warmup):
    wp.capture_launch(capture.graph)
  wp.synchronize()
  compile_warmup_s = time.perf_counter() - start
  # All separate models remain resident; one graph advances every world once in either layout.
  gpu_ms, wall_ms = [], []
  begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
  samples_diagnostics = []
  for _ in range(args.repeats):
    index.zero_()
    for state in states:
      reset(state)
    wp.synchronize()
    start = time.perf_counter()
    wp.record_event(begin)
    for _ in range(args.steps):
      wp.capture_launch(capture.graph)
    wp.record_event(end)
    wp.synchronize()
    wall_ms.append(1000 * (time.perf_counter() - start) / args.steps)
    gpu_ms.append(wp.get_event_elapsed_time(begin, end) / args.steps)
    samples_diagnostics.append([diagnostics(state) for state in states])
  memory = {
    "model_array_bytes": sum(arrays_bytes(s[3]) for s in states),
    "data_array_bytes": sum(arrays_bytes(s[4]) for s in states),
    "replay_array_bytes": sum(arrays_bytes(c) for _, c in created) + arrays_bytes(index),
    "device_used_delta_bytes": free_start - wp.get_device().free_memory,
  }
  # Replay outside timing and sample state and awake-tree checks every 100 ms.
  trajectory = {}
  index.zero_()
  for state in states:
    reset(state)
  for step in range(args.steps):
    wp.capture_launch(capture.graph)
    if step % 50 == 0 or step == args.steps - 1:
      for state in states:
        diagnostics(state)
      snapshot(states, trajectory)
  lifts = {}
  for state in states:
    kind, mjm, mjd, _, d, _ = state
    qpos = d.qpos.numpy()
    for group, name in enumerate(KINDS if kind == "mixed" else (kind,)):
      adr = mjm.joint(name + "/cube_joint").qposadr[0] + 2
      row = group * args.worlds if kind == "mixed" else 0
      height = qpos[row : row + args.worlds, adr] - mjd.qpos[adr]
      lifts[name] = float(height.min())
      if height.min() < 0.03:
        raise RuntimeError(f"{name}: cube did not lift: {height.min()} m")
  np.savez(args.output / f"{args.case}_trajectory.npz", **trajectory)
  nworld = sum(s[4].nworld for s in states)
  result = {
    "case": args.case,
    "minimum_cube_lift_m": lifts,
    "worlds": nworld,
    "setup_s": setup_s,
    "compile_warmup_s": compile_warmup_s,
    "gpu_batch_step_ms": gpu_ms,
    "wall_batch_step_ms": wall_ms,
    "world_steps_per_second": nworld * 1000 / float(np.median(wall_ms)),
    "simulated_seconds_per_second": nworld * 2 / float(np.median(wall_ms)),
    "memory": memory,
    "diagnostics": samples_diagnostics,
    "gpu": wp.get_device().name,
    "warp": wp.__version__,
    "mujoco": mujoco.__version__,
  }
  (args.output / f"{args.case}.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--assets", type=Path, required=True, help="Menagerie checkout at the documented pinned revision")
  parser.add_argument("--output", type=Path, default=Path("heterogeneous-results"))
  parser.add_argument("--worlds", type=int, default=8192, help="Worlds of EACH type")
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--warmup", type=int, default=10)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--case", choices=("mixed", "separate", "separate_no_sleep"), help=argparse.SUPPRESS)
  args = parser.parse_args()
  args.steps = 2500
  if min(args.worlds, args.repeats, args.warmup) < 1:
    parser.error("Counts and capacities must be positive")
  args.assets = args.assets.resolve()
  args.output = args.output.resolve()
  args.output.mkdir(parents=True, exist_ok=True)
  if args.case:
    worker(args)
    return
  revision = subprocess.check_output(["git", "-C", str(args.assets), "rev-parse", "HEAD"], text=True).strip()
  if revision != MENAGERIE_REF:
    parser.error(f"Expected Menagerie {MENAGERIE_REF}, got {revision}")
  child_args = []
  for key, value in vars(args).items():
    if key not in ("case", "steps"):
      child_args.extend(["--" + key, str(value)])
  results = []
  for case in ("mixed", "separate", "separate_no_sleep"):
    print(f"Running {case} in a fresh process...", flush=True)
    with (args.output / f"{case}.log").open("w") as log:
      subprocess.run(
        ["uv", "run", "python", str(Path(__file__).resolve()), *child_args, "--case", case],
        stdout=log,
        stderr=subprocess.STDOUT,
        check=True,
        cwd=ROOT,
      )
    results.append(json.loads((args.output / f"{case}.json").read_text()))
  mixed = np.load(args.output / "mixed_trajectory.npz")
  parity = {}
  cube_position_errors = {}
  world_spread = {}
  failures = []
  # Contact order can differ across layouts. Bound raw qpos component differences
  # by 0.002 and qvel by 0.1, and report actual errors and within-layout variation.
  for case in ("separate", "separate_no_sleep"):
    with np.load(args.output / f"{case}_trajectory.npz") as reference:
      for key in reference.files:
        delta = float(np.max(np.abs(mixed[key] - reference[key])))
        if key.endswith("_spread"):
          world_spread[f"mixed/{key}"] = float(mixed[key].max())
          world_spread[f"{case}/{key}"] = float(reference[key].max())
          tolerance = 0.002 if "qpos" in key else 0.1
          if max(world_spread[f"mixed/{key}"], world_spread[f"{case}/{key}"]) > tolerance:
            failures.append(f"{case}/{key}: within-layout spread exceeds {tolerance}")
          continue
        if key.endswith("qpos"):
          # Each source scene appends the cube's free joint after its robot joints.
          cube_position_errors[f"{case}/{key}"] = float(
            np.linalg.norm(mixed[key][:, -7:-4] - reference[key][:, -7:-4], axis=1).max()
          )
        if not key.endswith("_spread") and not np.allclose(
          mixed[key], reference[key], atol=0.002 if key.endswith("qpos") else 0.1, rtol=0
        ):
          failures.append(f"{case}/{key}: max absolute error {delta:g}")
        parity[f"{case}/{key}"] = delta
  report = {
    "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
    "menagerie_commit": revision,
    "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "trajectory_parity_failures": failures,
    "trajectory_max_absolute_errors": parity,
    "cube_position_errors_m": cube_position_errors,
    "within_layout_max_spread": world_spread,
    "results": results,
  }
  (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
  if failures:
    raise RuntimeError(
      "Trajectory parity failed; timings are not validated equal work. See results.json: " + "; ".join(failures)
    )
  print("case              worlds  wall ms/step  GPU ms/step  world-steps/s  arrays MiB  GPU delta MiB")
  for result in results:
    mem = result["memory"]
    arrays = mem["model_array_bytes"] + mem["data_array_bytes"] + mem["replay_array_bytes"]
    print(
      f"{result['case']:17} {result['worlds']:6} {np.median(result['wall_batch_step_ms']):13.3f} "
      f"{np.median(result['gpu_batch_step_ms']):12.3f} {result['world_steps_per_second']:14,.0f} "
      f"{arrays / 2**20:11.2f} {mem['device_used_delta_bytes'] / 2**20:14.2f}"
    )
  print("Trajectory parity passed; full samples and metadata:", args.output / "results.json")


if __name__ == "__main__":
  main()

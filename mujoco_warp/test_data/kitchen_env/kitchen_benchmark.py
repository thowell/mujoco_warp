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

"""Create kitchen environment with robot from MuJoCo Menagerie."""

import os
import subprocess
from typing import Sequence

import mujoco
from absl import app
from absl import flags
from etils import epath

# script path to get correct xml and mesh path
_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# menagerie path is used to load robot assets
# resource paths do not have glob implemented, so we use epath.Path
_MENAGERIE_PATH = epath.Path(__file__).parent.parent / "mujoco_menagerie"

# commit sha of the mujoco menagerie github repository
_MENAGERIE_COMMIT_SHA = "14ceccf557cc47240202f2354d684eca58ff8de4"

_ROBOT = flags.DEFINE_enum(
  "robot",
  "mujoco_humanoid",
  [
    # robotic arms
    "panda",
    "fr3",
    "google_robot",
    "gen3",
    "iiwa_14",
    "tiago",
    "sawyer",
    "vx300",
    "arm100",
    "lite6",
    "xarm7",
    "z1",
    "ur10e",
    "ur5e",
    # humanoids
    "mujoco_humanoid",
    "berkeley_humanoid",
    "t1",
    "h1",
    "g1",
    "talos",
    "op3",
    # quadrupeds
    "spot",
    "anymal_b",
    "anymal_c",
    "barkour_v0",
    "a1",
    "go1",
    "go2",
    # bipedal
    "cassie",
  ],
  "the robot to use",
)


def main(argv: Sequence[str]):
  """Create kitchen environment with robot from MuJoCo Menagerie."""

  # processing robot
  match _ROBOT.value:
    case "panda":
      robot_path = _load_from_menagerie("franka_emika_panda/mjx_panda.xml")
    case "fr3":
      robot_path = _load_from_menagerie("franka_fr3/fr3.xml")
    case "google_robot":
      robot_path = _load_from_menagerie("google_robot/robot.xml")
    case "gen3":
      robot_path = _load_from_menagerie("kinova_gen3/gen3.xml")
    case "iiwa_14":
      robot_path = _load_from_menagerie("kuka_iiwa_14/iiwa14.xml")
    case "tiago":
      robot_path = _load_from_menagerie("pal_tiago/tiago.xml")
    case "sawyer":
      robot_path = _load_from_menagerie("rethink_robotics_sawyer/sawyer.xml")
    case "vx300":
      robot_path = _load_from_menagerie("trossen_vx300s/vx300s.xml")
    case "arm100":
      robot_path = _load_from_menagerie("trs_so_arm100/so_arm100.xml")
    case "lite6":
      robot_path = _load_from_menagerie("ufactory_lite6/lite6.xml")
    case "xarm7":
      robot_path = _load_from_menagerie("ufactory_xarm7/xarm7.xml")
    case "z1":
      robot_path = _load_from_menagerie("unitree_z1/z1.xml")
    case "ur10e":
      robot_path = _load_from_menagerie("universal_robots_ur10e/ur10e.xml")
    case "ur5e":
      robot_path = _load_from_menagerie("universal_robots_ur5e/ur5e.xml")
    case "mujoco_humanoid":
      robot_path = epath.Path(_SCRIPT_DIR + "/../humanoid/humanoid.xml")
    case "berkeley_humanoid":
      robot_path = _load_from_menagerie("berkeley_humanoid/berkeley_humanoid.xml")
    case "t1":
      robot_path = _load_from_menagerie("booster_t1/t1.xml")
    case "h1":
      robot_path = _load_from_menagerie("unitree_h1/h1.xml")
    case "g1":
      robot_path = _load_from_menagerie("unitree_g1/g1.xml")
    case "talos":
      robot_path = _load_from_menagerie("pal_talos/talos.xml")
    case "op3":
      robot_path = _load_from_menagerie("robotis_op3/op3.xml")
    case "spot":
      robot_path = _load_from_menagerie("boston_dynamics_spot/spot.xml")
    case "anymal_b":
      robot_path = _load_from_menagerie("anybotics_anymal_b/anymal_b.xml")
    case "anymal_c":
      robot_path = _load_from_menagerie("anybotics_anymal_c/anymal_c.xml")
    case "barkour_v0":
      robot_path = _load_from_menagerie("google_barkour_v0/barkour_v0.xml")
    case "a1":
      robot_path = _load_from_menagerie("unitree_a1/a1.xml")
    case "go1":
      robot_path = _load_from_menagerie("unitree_go1/go1.xml")
    case "go2":
      robot_path = _load_from_menagerie("unitree_go2/go2.xml")
    case "cassie":
      robot_path = _load_from_menagerie("agility_cassie/cassie.xml")
    case _:
      raise FileNotFoundError(f"Robot provided is unknown.")

  # verify that kitchen and robot .xml files exist
  kitchen_path = epath.Path(_SCRIPT_DIR + "/kitchen_env.xml")
  if not kitchen_path.exists():
    raise FileNotFoundError(f"kitchen_env.xml not found.")
  if not robot_path.exists():
    raise FileNotFoundError(f"robot xml not found.")
  
  # create directory with kitchen + robot assets
  subprocess.run(f'mkdir -p {_SCRIPT_DIR}/kitchen{_ROBOT.value}/assets', shell=True, text=True)
  subprocess.run(f'cp -r {_SCRIPT_DIR}/assets {_SCRIPT_DIR}/kitchen{_ROBOT.value}', shell=True, text=True)
  # TODO(team): robot without assets (eg, humanoid)
  subprocess.run(f'cp -r {os.path.dirname(robot_path)}/assets {_SCRIPT_DIR}/kitchen{_ROBOT.value}', shell=True, text=True)
  
  # create xml
  spec = mujoco.MjSpec.from_file(kitchen_path.as_posix())
  spec_xml = spec.to_xml().replace("assets/", f"kitchen{_ROBOT.value}/assets/")
  spec = mujoco.MjSpec.from_string(spec_xml)
  robot = mujoco.MjSpec.from_file(robot_path.as_posix())

  # add robot to environment
  attach_frame = spec.worldbody.add_frame(pos=[0, 0, 0])
  spec.attach(robot, frame=attach_frame, prefix="robot")
  spec.to_xml() # write to file


def _load_from_menagerie(asset_path: str) -> str:
  """Load an asset from the mujoco menagerie."""
  # Ensure menagerie exists, and otherwise clone it
  _menagerie_exists()
  return _MENAGERIE_PATH / asset_path


def _menagerie_exists() -> None:
  """Ensure mujoco_menagerie exists, downloading it if necessary."""
  if not _MENAGERIE_PATH.exists():
    print("mujoco_menagerie not found. Downloading...")

    try:
      _clone_with_progress(
        "https://github.com/deepmind/mujoco_menagerie.git",
        str(_MENAGERIE_PATH),
        _MENAGERIE_COMMIT_SHA,
      )
      print("Successfully downloaded mujoco_menagerie")
    except subprocess.CalledProcessError as e:
      print(f"Error downloading mujoco_menagerie: {e}", file=sys.stderr)
      raise


def _clone_with_progress(repo_url: str, target_path: str, commit_sha: str) -> None:
  """Clone a git repo with progress bar."""
  process = subprocess.Popen(
    ["git", "clone", "--progress", repo_url, target_path],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    universal_newlines=True,
  )

  while True:
    # Read output line by line.
    output = process.stderr.readline()  # pytype: disable=attribute-error
    if not output and process.poll() is not None:
      break
    if output:
      if "Receiving objects:" in output:
        try:
          percent = int(output.split("%")[0].split(":")[-1].strip())
          # if percent > current:
          #   current = percent
        except (ValueError, IndexError):
          pass

  if process.returncode != 0:
    raise subprocess.CalledProcessError(process.returncode, ["git", "clone"])

  # checkout specific commit
  print(f"Checking out commit {commit_sha}")
  subprocess.run(
    ["git", "-C", target_path, "checkout", commit_sha],
    check=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )


if __name__ == "__main__":
  app.run(main)

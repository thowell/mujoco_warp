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

from .types import Data
from .types import DisableBit
from .types import Model
from .types import ObjType
from .types import SensorType
from .warp_util import event_scope
from .warp_util import kernel


@wp.func
def _joint_pos(m: Model, d: Data, worldid: int, objid: int) -> wp.float32:
  return d.qpos[worldid, m.jnt_qposadr[objid]]


@wp.func
def _actuator_length(m: Model, d: Data, worldid: int, objid: int) -> wp.float32:
  return d.actuator_length[worldid, objid]


@wp.func
def _ball_quat(m: Model, d: Data, worldid: int, objid: int) -> wp.quat:
  jnt_qposadr = m.jnt_qposadr[objid]
  quat = wp.quat(
    d.qpos[worldid, jnt_qposadr + 0],
    d.qpos[worldid, jnt_qposadr + 1],
    d.qpos[worldid, jnt_qposadr + 2],
    d.qpos[worldid, jnt_qposadr + 3],
  )
  return wp.normalize(quat)


@wp.func
def _frame_pos(
  m: Model, d: Data, worldid: int, objid: int, objtype: int, refid: int
) -> wp.vec3:
  if objtype == int(ObjType.BODY.value):
    xpos = d.xipos[worldid, objid]
    if refid == -1:
      return xpos
    xpos_ref = d.xipos[worldid, refid]
    xmat_ref = d.ximat[worldid, refid]
  elif objtype == int(ObjType.XBODY.value):
    xpos = d.xpos[worldid, objid]
    if refid == -1:
      return xpos
    xpos_ref = d.xpos[worldid, refid]
    xmat_ref = d.xmat[worldid, refid]
  elif objtype == int(ObjType.GEOM.value):
    xpos = d.geom_xpos[worldid, objid]
    if refid == -1:
      return xpos
    xpos_ref = d.geom_xpos[worldid, refid]
    xmat_ref = d.geom_xmat[worldid, refid]
  elif objtype == int(ObjType.SITE.value):
    xpos = d.site_xpos[worldid, objid]
    if refid == -1:
      return xpos
    xpos_ref = d.site_xpos[worldid, refid]
    xmat_ref = d.site_xmat[worldid, refid]

  # TODO(team): camera

  else:  # UNKNOWN
    return wp.vec3(0.0)

  return wp.transpose(xmat_ref) @ (xpos - xpos_ref)


@event_scope
def sensor_pos(m: Model, d: Data):
  """Compute position-dependent sensor values."""

  @kernel
  def _sensor_pos(m: Model, d: Data):
    worldid, posid = wp.tid()
    posadr = m.sensor_pos_adr[posid]
    sensortype = m.sensor_type[posadr]
    objid = m.sensor_objid[posadr]
    adr = m.sensor_adr[posadr]

    if sensortype == int(SensorType.JOINTPOS.value):
      d.sensordata[worldid, adr] = _joint_pos(m, d, worldid, objid)
    elif sensortype == int(SensorType.ACTUATORPOS.value):
      d.sensordata[worldid, adr] = _actuator_length(m, d, worldid, objid)
    elif sensortype == int(SensorType.BALLQUAT.value):
      quat = _ball_quat(m, d, worldid, objid)
      d.sensordata[worldid, adr + 0] = quat[0]
      d.sensordata[worldid, adr + 1] = quat[1]
      d.sensordata[worldid, adr + 2] = quat[2]
      d.sensordata[worldid, adr + 3] = quat[3]
    elif sensortype == int(SensorType.FRAMEPOS.value):
      objtype = m.sensor_objtype[posadr]
      refid = m.sensor_refid[posadr]
      framepos = _frame_pos(m, d, worldid, objid, objtype, refid)
      d.sensordata[worldid, adr + 0] = framepos[0]
      d.sensordata[worldid, adr + 1] = framepos[1]
      d.sensordata[worldid, adr + 2] = framepos[2]

  if (m.sensor_pos_adr.size == 0) or (m.opt.disableflags & DisableBit.SENSOR):
    return

  wp.launch(_sensor_pos, dim=(d.nworld, m.sensor_pos_adr.size), inputs=[m, d])


@wp.func
def _joint_vel(m: Model, d: Data, worldid: int, objid: int) -> wp.float32:
  return d.qvel[worldid, m.jnt_dofadr[objid]]


@event_scope
def sensor_vel(m: Model, d: Data):
  """Compute velocity-dependent sensor values."""

  @kernel
  def _sensor_vel(m: Model, d: Data):
    worldid, velid = wp.tid()
    veladr = m.sensor_vel_adr[velid]
    sensortype = m.sensor_type[veladr]
    objid = m.sensor_objid[veladr]
    adr = m.sensor_adr[veladr]

    if sensortype == int(SensorType.JOINTVEL.value):
      d.sensordata[worldid, adr] = _joint_vel(m, d, worldid, objid)

  if (m.sensor_vel_adr.size == 0) or (m.opt.disableflags & DisableBit.SENSOR):
    return

  wp.launch(_sensor_vel, dim=(d.nworld, m.sensor_vel_adr.size), inputs=[m, d])


@wp.func
def _actuator_force(m: Model, d: Data, worldid: int, objid: int) -> wp.float32:
  return d.actuator_force[worldid, objid]


@event_scope
def sensor_acc(m: Model, d: Data):
  """Compute acceleration-dependent sensor values."""

  @kernel
  def _sensor_acc(m: Model, d: Data):
    worldid, accid = wp.tid()
    accadr = m.sensor_acc_adr[accid]
    sensortype = m.sensor_type[accadr]
    objid = m.sensor_objid[accadr]
    adr = m.sensor_adr[accadr]

    if sensortype == int(SensorType.ACTUATORFRC.value):
      d.sensordata[worldid, adr] = _actuator_force(m, d, worldid, objid)

  if (m.sensor_acc_adr.size == 0) or (m.opt.disableflags & DisableBit.SENSOR):
    return

  wp.launch(_sensor_acc, dim=(d.nworld, m.sensor_acc_adr.size), inputs=[m, d])

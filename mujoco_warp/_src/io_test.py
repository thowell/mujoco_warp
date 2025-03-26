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

"""Tests for io functions."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjwarp


class IOTest(absltest.TestCase):
  def test_equality(self):
    mjm = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <body name="body1">
          <geom type="sphere" size=".1"/>
          <freejoint/>
        </body>
        <body name="body2">
          <geom type="sphere" size=".1"/>
          <freejoint/>
        </body>
      </worldbody>  
      <equality>
        <connect body1="body1" body2="body2" anchor="0 0 0"/>                      
      </equality>              
    </mujoco>
    """)

    with self.assertRaises(NotImplementedError):
      mjwarp.put_model(mjm)

  def test_sensor(self):
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body>
            <geom type="sphere" size=".1"/>
            <joint name="slide" type="slide"/>
          </body>
        </worldbody>   
        <sensor>
          <jointpos joint="slide"/>                      
        </sensor> 
      </mujoco>
    """)

    with self.assertRaises(NotImplementedError):
      mjwarp.put_model(mjm)

  def test_tendon(self):
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body>          
            <geom type="sphere" size=".1"/>
            <site name="site0"/>
            <joint name="slide" type="slide"/>
            <body pos="0 0 .1">
              <geom type="sphere" size=".1"/>
              <site name="site1"/>
            </body>
          </body>
        </worldbody>  
        <tendon>
          <spatial>
            <site site="site0"/>
            <site site="site1"/>
          </spatial>                      
        </tendon>              
      </mujoco>
    """)

    with self.assertRaises(NotImplementedError):
      mjwarp.put_model(mjm)

  def test_actuator_trntype(self):
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body name="body">          
            <geom type="sphere" size=".1"/>
            <site name="site0"/>
            <joint type="slide"/>
          </body>
          <site name="site1"/>
        </worldbody>  
        <tendon>
          <spatial name="tendon">
            <site site="site0"/>
            <site site="site1"/>
          </spatial>                      
        </tendon>
        <actuator>
          <general cranksite="site0" slidersite="site1" cranklength=".1"/>
          <general tendon="tendon"/>
          <general site="site0" refsite="site1"/>
          <general body="body" ctrlrange="0 1"/>
        </actuator>           
      </mujoco>
    """)

    with self.assertRaises(NotImplementedError):
      mjwarp.put_model(mjm)

  def test_actuator_dyntype(self):
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body>          
            <geom type="sphere" size=".1"/>
            <joint name="slide" type="slide"/>
          </body>
        </worldbody>  
        <actuator>
          <general joint="slide" dyntype="integrator"/>
          <general joint="slide" dyntype="filter"/>
          <general joint="slide" dyntype="muscle"/>
        </actuator>
      </mujoco>
    """)

    with self.assertRaises(NotImplementedError):
      mjwarp.put_model(mjm)

  def test_actuator_gaintype(self):
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <site name="siteworld"/>
          <body>          
            <geom type="sphere" size=".1"/>
            <site name="site0"/>
            <joint name="slide" type="slide"/>
          </body>
        </worldbody>  
        <tendon>
          <spatial name="tendon">
            <site site="siteworld"/>
            <site site="site0"/>
          </spatial>                      
        </tendon>
        <actuator>
          <muscle tendon="tendon" lengthrange="0 1"/>
          </actuator>
      </mujoco>
    """)

    with self.assertRaises(NotImplementedError):
      mjwarp.put_model(mjm)


if __name__ == "__main__":
  wp.init()
  absltest.main()

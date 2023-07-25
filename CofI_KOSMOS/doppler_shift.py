# -*- coding: utf-8 -*-
"""doppler_shift

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17UBNF18EsaXyL6oxunUTgyY8k6cffViS
"""

import math
from astropy import units as u

__all__ = ['doppler_shift']
def doppler_shift(rest, obs):
  """
  Function for calculating radial velocity.

  Parameters:

  rest: Quantity
  The wavelength at rest, preferably in Angstroms. Must have the same units as obs.

  obs: Quantity
  The observed wavelength in your data, preferably in Angstroms. Must have the same units as rest.

  Returns:

  vel: Quantity
  Radial velocity in km/s.

  """
  c = 2.99792458 * math.pow(10, 10) * u.cm/u.s
  z = (obs - rest) / rest
  vel = (c * (math.pow(z) + 2 * z) / (math.pow(z) + 2 * z + 2))
  vel = vel / 100000 / vel.unit * u.km / u.s
  return vel

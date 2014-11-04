#!/usr/bin/python
# Filename: readhdf.py
# Author: wgx
from pyhdf.SD import SD,SDC
import numpy as np
def read(filename,fieldname):
	hdf = SD(filename,SDC.READ)
	data = hdf.select(fieldname)
	hdf.end()
	return data[:]

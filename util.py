#!/usr/bin/python
# Filename: util.py
# Author: wgx
import os
from pyhdf.SD import SD,SDC
import numpy as np
def readhdf(filename,fieldname):
	hdf = SD(filename,SDC.READ)
        data = hdf.select(fieldname)[:].copy()
	hdf.end()
	return data
def lsfiles(path=os.getcwd(),keys=''):
	result = []
	if os.path.exists(path) == False:
		raise IOError('Path not Exists!')
	for root,dirs,files in os.walk(path):
		for fp in files:
			if keys in fp:
				result.append(os.path.join(root,fp))
	return result
def num_to_day(str):
    year = str[:4]
    num = int(str[4:])
    mon_num = [31,28,31,30,31,30,31,31,30,31,30,31]
    if is_r(int(year)):
        mon_num[1] = 29
    m = 0
    d = 0
    for i in range(12):
        if num - mon_num[i] > 0 :
            num -= mon_num[i]
        else :
            m = i + 1
            d = num
            break
    return "%4s%02d%02d" % (year,m,d)
def is_r(year):
    r = False
    if year % 4==0:
        r = True
    if year%100==0 and year%400!=0:
        r = False
    return r

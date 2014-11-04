#!/usr/bin/python
# Filename: lsdir.py
# Author: wgx
import os
def lsallfiles(path=os.getcwd(),keys=''):
	result = []
	if os.path.exists(path) == False:
		raise IOError('Path not Exists!')
	for root,dirs,files in os.walk(path):
		for fp in files:
			if keys in fp:
				result.append(os.path.join(root,fp))
	return result

from __future__ import division

import numpy as np

def _s2n(string):
	"""Convert a string to a float or int.
	"""

	return float(string) if '.' in string else int(string)
	

def fastec(fname):
	"""Load in metadata for Fastec cameras.

	Parameters
	----------
	fname : str
	    File to open.

	Returns
	-------
	config : dict
	    Nested dictionary.
	"""

	config = {}
	with open(fname, 'r') as fp:
		lines = fp.readlines()
		
	for idx, line in enumerate(lines):
		if line.startswith('['):
			print line
			param = line[1:-2]
			d = {}
		if line.startswith('\t'):
			key, val = line[1:-1].split('=')
			if val.isdigit():  # this will be a number
				val = _s2n(val)
			elif val.startswith('['):  # this will be a list
			    val = map(_s2n, val[1:-1].split(','))
			d[key] = val
		if line.startswith('\n') or idx == len(lines) - 1:
			config[param] = d
	return config

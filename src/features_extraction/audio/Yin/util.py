	#!/usr/local/bin/python2

import numpy as np
import yin
from time import time

def filterPitches ( pitches, argmins, percentileMin, percentileMax):
	pitches = np.array(pitches)
	
	for i in range( 1, len(pitches) - 1 ):
		if pitches[i] == 0 and pitches[i-1] != 0 and pitches[i+1] != 0:
			pitches[i] = pitches[i-1]
	#utiliser l'argmax et l'harmonicite ici
	
	non0Indexes = np.nonzero( pitches )[0]
	non0Values = pitches[ non0Indexes]
	minBound = np.percentile(non0Values, percentileMin )
	maxBound = np.percentile( non0Values, percentileMax)

	if non0Indexes[0] == 0 and (pitches[0] <= minBound or pitches[0] >= maxBound): pitches[0] = 0

	for i in non0Indexes[1:]:
		pitch = pitches[i]
		if pitch <= minBound or pitch >= maxBound: pitches[i] = pitches[i - 1]

	return pitches.tolist()


def runYin( result, s, sr, w_len = 1024, w_step = 512, f0_min = 70, f0_max = 450, harmo_thresh = 0.15 ):
	print ("Charger F0")
	pitches, harmonic_rates, argmins, times  =  yin.compute_yin(s, sr, None, w_len = w_len, w_step = w_step, f0_min = f0_min, f0_max = f0_max, harmo_thresh = harmo_thresh)
	pitches = filterPitches ( pitches, argmins, 2, 98)
	F0 = np.array ( zip ( times, pitches, harmonic_rates, argmins))
	f0Time = time()
	result.append(F0)
	result.append(f0Time)


if __name__ == "__main__":
	pitches = [450,0,111,112,0,130,130,0,0,100,70,70,136,145,150,111]
	print ("before filtering:")
	print (pitches)
	pitches = filterPitches(pitches, pitches, 2, 98)
	print ("After filtering:")
	print(pitches)


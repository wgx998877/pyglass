#!/usr/bin/env python
# coding=utf-8

import numppy as np
import util as u

Nband = 1
Ntheta0 = 9
Ntheta = 5
Nfi = 7
Nvis = 18
Nele = 6
Nwater = 10

class rad():
    
    theta0 = [1.0, 15.0, 30.0, 40.0, 55.0, 65.0, 75.0, 85.0, 90.0]
    theta =  [1.0, 20.0, 40.0, 60.0, 80.0]
    fi = [1.0, 30.0, 60.0, 90.0, 120.0, 150.0, 179.0]
    vis = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 10.0, 15.0, 20.0 ,30.0 ,50.0, 100.0]
    ele = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    def __init__(self):
        self.lp0 = np.zeros(
                Ntheta0, Ntheta, Nfi, Nwater, Nvis, Nele
            )
        self.rbar0 = np.zeros(
                Ntheta0, Ntheta, Nfi, Nwater, Nvis, Nele
            )
        self.fd0 = np.zeros(
                Ntheta0, Ntheta, Nfi, Nwater, Nvis, Nele
            )

    def readTOALUTbyElevation(self, filename, elevation):
        if u.file_exist(filename) == False:
            return None
        Nwat = 0
        if elevation == 0 : Nwat = 10
        elif elevation == 1 : Nwat = 9
        elif elevation == 2 : Nwat = 8
        elif elevation == 3 : Nwat = 7
        elif elevation == 4 : Nwat = 6
        elif elevation == 5 : Nwat = 7
        nele_index = int(elevation)
        fTOA = open(filename)
        for i1 in range(Ntheta0 - 1):
            for i2 in range(Ntheta):
                for i3 in range(Nfi):
                    for i4 in range(Nwat):
                        for i5 in range(Nvis):
                            line = fTOA.readline()
                            w = map(float, line.strip().split())
                            self.lp0[i1, i2, i3, i4, i5, nele_index] = w[5]
                            self.rbar0[i1, i2, i3, i4, i5, nele_index] = w[6]
                            self.fd0[i1, i2, i3, i4, i5, nele_index] = w[7]
    def readSURLUTbyElevation(self, filename, elevation):



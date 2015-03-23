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
        self.lp = np.zeros((Nvis, Nele))
        self.fd = np.zeros((Nvis, Nele))
        self.rbar = np.zeros((Nvis, Nele))
        self.lppdf = np.zeros((Nvis, Nele))
        self.lppdi = np.zeros((Nvis, Nele))
        self.rbarp = np.zeros((Nvis, Nele))
        self.fdp = np.zeros((Nvis, Nele))

        self.lp0 = np.zeros((Ntheta0, Ntheta, Nfi, Nwater, Nvis, Nele))
        self.rbar0 = np.zeros((Ntheta0, Ntheta, Nfi, Nwater, Nvis, Nele))
        self.fd0 = np.zeros((Ntheta0, Ntheta, Nfi, Nwater, Nvis, Nele))
        
        self.rbarp0 = np.zeros((Ntheta0, Nwater, Nvis, Nele))
        self.fdp0 = np.zeros((Ntheta0, Nwater, Nvis, Nele))
        self.lppdf0 = np.zeros((Ntheta0, Nwater, Nvis, Nele))

        self.lppdi0 = np.zeros((Ntheta0, Nwater, Nvis, Nele))

    def readLUTbyElevation(self, filename, elevation, Type = 'TOA'):
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
        fLUT = open(filename)
        if Type == 'TOA':
            for i1 in range(Ntheta0 - 1):
                for i2 in range(Ntheta):
                    for i3 in range(Nfi):
                        for i4 in range(Nwat):
                            for i5 in range(Nvis):
                                line = fLUT.readline()
                                w = map(float, line.strip().split())
                                self.lp0[i1, i2, i3, i4, i5, nele_index] = w[5]
                                self.rbar0[i1, i2, i3, i4, i5, nele_index] = w[6]
                                self.fd0[i1, i2, i3, i4, i5, nele_index] = w[7]
        elif Type == 'SUR':
            for i1 in range(Ntheta0 - 1):
                for i2 in range(Nwat):
                    for i3 in range(Nvis):
                        line = fLUT.readline()
                        w = map(float, line.strip().split())
                        self.rbarp0[i1, i2, i3, nele_index] = w[3]
                        self.fdp0[i1, i2, i3, nele_index] = w[4]
                        self.lppdf0[i1, i2, i3, nele_index] = w[5]
                        self.lppdi0[i1, i2, i3, nele_index] = w[6]

        return True

    def calIndex(self, angle, Nl, l):
        index = 0
        for i in range(Nl - 1):
            if angle >= l[i] and angle <= l[i+1]:
                index = i+1
        if angle > l[Nl - 1]:
            index = Nl - 1
        if angle < l[0]:
            index = 1
        return index
    
    def calSolarZenithIndex(self, angle):
        return self.calIndex(angle, Ntheta0, self.theta0)
    
    def calSensorZenithIndex(self, angle):
        return self.calIndex(angle, Ntheta, self.theta)
    
    def calSensorAzimuthIndex(self, angle):
        return self.calIndex(angle, Nfi, self.fi)

    def getWaterAmount(self, elevation):
        waterAmount = np.zeros(Nwater)
        Nwat = 0
        if elevation == 0 :
            Nwat = 10
            waterAmount = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5, 6.5]
        elif elevation == 1:
            Nwat = 9
            waterAmount = [0.2, 0.5, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.5]
        elif elevation == 2:
            Nwat = 8
            waterAmount = [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        elif elevation == 3:
            Nwat = 7
            waterAmount = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0]
        elif elevation == 4:
            Nwat = 6
            waterAmount = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
        elif elevation == 5:
            Nwat = 7
            waterAmount = [0.1, 0.2, 0.5, 0.7, 0.9, 1.5, 1.9]
        return Nwat, waterAmount

    def calWaterIndexbyElevation(self, zenith, elevation):
        Nwat, waterAmount = self.getWaterAmount(elevation)
        return self.calIndex(zenith, Nwat, waterAmount)

    def SearchLUTbyElevation_rad(self, rad, ref, elevation):#, parDi, parDif, visi):
        A = [0] * Nvis
        nele_index = int(elevation)
        for i in range(Nvis):
            A[i] = self.lp[i][nele_index] + ref * self.fd[i][nele_index] / (1 - ref * self.rbar[i][nele_index])
        Amax = max(A)
        Amax_index = A.index(Amax)
        Amin = min (A)
        Amin_index = A.index(Amin)
        flag = True if Amax > 0 else False
        if flag :
            parDif = 0
            parDi  = 0
        else:
            if rad <= Amin:
                visibility = Nvis - 1
                k = Amin_index
                parDif = self.lppdf[k][nele_index] + ref * self.fdp[k][nele_index] / (1.0 - ref * self.rbarp[k][nele_index]) * self.rbarp[k][nele_index]
                parDi = self.lppdi[k][nele_index]
            elif rad >= Amax:
                visibility = 0
                k = Amax_index
                k = Ntheta0 - 1
                parDif = self.lppdf[k][nele_index] + ref * self.fdp[k][nele_index] / (1.0 - ref * self.rbarp[k][nele_index]) * self.rbarp[k][nele_index]
                parDi = self.lppdi[k][nele_index]

            else:
                for i in range(Nvis - 1):
                    if ((rad >= A[i] and rad <= A[i+1]) or (rad <= A[i] and rad >= A[i+1])):
                        p = (rad - A[i])
                        visibility = i + p
                        k = i
                        tlp = self.lppdf[k][nele_index] + (self.lppdf[k+1][nele_index] - self.lppdf[k][nele_index]) * p
                        trbar = self.rbarp[k][nele_index] + (self.rbarp[k+1][nele_index] - self.rbarp[k][nele_index]) * p
                        tfd = self.fdp[k][nele_index] + (self.fdp[k+1][nele_index] - self.fdp[k][nele_index]) * p

                        parDif = (tlp + ref * tfd / (1.0 - ref * trbar) * trbar)
                        parDi =  (self.lppdi[k][nele_index] + (self.lppdi[k+1][nele_index] - self.lppdi[k][nele_index]) * p)
                        break
        return (parDi, parDif, visibility)

    def SearchLUTbyElevation(self, ref, visi, elevation):
        par = 0
        nele_index = int(elevation)
        if visi == Nvis:
            k = Nvis - 1
            parDif = self.lppdf[k][nele_index] + ref * self.fdp[k][nele_index] / (1.0 - ref * self.rbarp[k][nele_index]) * self.rbarp[k][nele_index]
            parDi = self.lppdi[k][nele_index]
            par = parDif + parDi
        elif visi == 0 :
            k = Nvis - 1
            parDif = self.lppdf[k][nele_index] + ref * self.fdp[k][nele_index] / (1.0 - ref * self.rbarp[k][nele_index]) * self.rbarp[k][nele_index]
            parDi = self.lppdi[k][nele_index]
            par = parDi + parDif
        else :
            k = int(visi)
            p = visi - k
            tlp = self.lppdf[k][nele_index] + (self.lppdf[k+1][nele_index] - self.lppdf[k][nele_index]) * p
            trbar = self.rbarp[k][nele_index] + (self.rbarp[k+1][nele_index] - self.rbarp[k][nele_index]) * p
            tfd = self.fdp[k][nele_index] + (self.fdp[k+1][nele_index] - self.fdp[k][nele_index]) * p

            parDif = (tlp + ref * tfd / (1 - ref * trbar) * trbar )
            parDi  = (self.lppdi[k][nele_index] + (self.lppdi[k+1][nele_index] - self.lppdi[k][nele_index]) * p)
            par = parDif + parDi
        return par, parDif, parDi

    def InterpolationParaByElevation(self, solar_zenith, view_zenith, relative_azimuth, water, elevation):
        II = self.calSolarZenithIndex(solar_zenith)
        JJ = self.calSensorZenithIndex(view_zenith)
        KK = self.calSensorAzimuthIndex(relative_azimuth)
        LL = self.calWaterIndexbyElevation(water, elevation)

        dtheta0 = (solar_zenith - self.theta0[II - 1]) / (self.theta0[II] - self.theta0[II - 1])

        if solar_zenith < self.theta0[0]:
            dtheta0 = 0.0
        elif solar_zenith > self.theta0[Ntheta0 - 1]:
            dtheta0 = 1.0

        dtheta = (view_zenith - self.theta[JJ - 1]) / (self.theta[JJ] - self.theta[JJ - 1])
        if view_zenith < self.theta[0]:
            dtheta = 0.0
        elif view_zenith > self.theta[Ntheta - 1]:
            dtheta = 1.0
        dfi = (relative_azimuth - self.fi[KK - 1]) / (self.fi[KK] - self.fi[KK - 1])
        if relative_azimuth < self.fi[0]:
            dfi = 0.0
        elif relative_azimuth > self.fi[Nfi -1 ]:
            dfi = 1.0
        Nwat, waterAmount = self.getWaterAmount(elevation)
        dwater = (water - waterAmount[LL - 1]) / (waterAmount[LL] - waterAmount[LL-1])
        if water < waterAmount[0]:
            dwater = 0.0
        elif water > waterAmount[Nwat - 1]:
            dwater = 1.0

        nele_index = int(elevation)

        for i in range(Nvis):
            self.lppdf[i][nele_index] = 0.0
            self.lppdi[i][nele_index] = 0.0
            self.rbarp[i][nele_index] = 0.0
            self.fdp[i][nele_index] = 0.0

        for j in range(Nvis):
            xxiixjj1kk1=(1.0-dtheta0)*self.lp0[II-1][JJ][KK][LL-1][j][nele_index]+dtheta0*self.lp0[II][JJ][KK][LL-1][j][nele_index];
            xxiixjjkk1=(1.0-dtheta0)*self.lp0[II-1][JJ-1][KK][LL-1][j][nele_index]+dtheta0*self.lp0[II][JJ-1][KK][LL-1][j][nele_index];
            xxiixjj1kk=(1.0-dtheta0)*self.lp0[II-1][JJ][KK-1][LL-1][j][nele_index]+dtheta0*self.lp0[II][JJ][KK-1][LL-1][j][nele_index];
            xxiixjjkk=(1.0-dtheta0)*self.lp0[II-1][JJ-1][KK-1][LL-1][j][nele_index]+dtheta0*self.lp0[II][JJ-1][KK-1][LL-1][j][nele_index];
            xxiixjjykk1=(1.0-dtheta)*xxiixjjkk1+dtheta*xxiixjj1kk1;
            xxiixjjykk=(1.0-dtheta)*xxiixjjkk+dtheta*xxiixjj1kk;
            xxiixjjykkz1=(1.0-dfi)*xxiixjjykk+dfi*xxiixjjykk1;

            xxiixjj1kk12=(1.0-dtheta0)*self.lp0[II-1][JJ][KK][LL][j][nele_index]+dtheta0*self.lp0[II][JJ][KK][LL][j][nele_index];
            xxiixjjkk12=(1.0-dtheta0)*self.lp0[II-1][JJ-1][KK][LL][j][nele_index]+dtheta0*self.lp0[II][JJ-1][KK][LL][j][nele_index];
            xxiixjj1kk2=(1.0-dtheta0)*self.lp0[II-1][JJ][KK-1][LL][j][nele_index]+dtheta0*self.lp0[II][JJ][KK-1][LL][j][nele_index];
            xxiixjjkk2=(1.0-dtheta0)*self.lp0[II-1][JJ-1][KK-1][LL][j][nele_index]+dtheta0*self.lp0[II][JJ-1][KK-1][LL][j][nele_index];
            xxiixjjykk12=(1.0-dtheta)*xxiixjjkk12+dtheta*xxiixjj1kk12;
            xxiixjjykk2=(1.0-dtheta)*xxiixjjkk2+dtheta*xxiixjj1kk2;
            xxiixjjykkz2=(1.0-dfi)*xxiixjjykk2+dfi*xxiixjjykk12;

            xxiixjjykkz=(1.0-dwater)*xxiixjjykkz1+dwater*xxiixjjykkz2;
            self.lp[j][nele_index]=xxiixjjykkz;

            xxiixjj1kk1=(1.0-dtheta0)*self.rbar0[II-1][JJ][KK][LL-1][j][nele_index]+dtheta0*self.rbar0[II][JJ][KK][LL-1][j][nele_index];
            xxiixjjkk1=(1.0-dtheta0)*self.rbar0[II-1][JJ-1][KK][LL-1][j][nele_index]+dtheta0*self.rbar0[II][JJ-1][KK][LL-1][j][nele_index];
            xxiixjj1kk=(1.0-dtheta0)*self.rbar0[II-1][JJ][KK-1][LL-1][j][nele_index]+dtheta0*self.rbar0[II][JJ][KK-1][LL-1][j][nele_index];
            xxiixjjkk=(1.0-dtheta0)*self.rbar0[II-1][JJ-1][KK-1][LL-1][j][nele_index]+dtheta0*self.rbar0[II][JJ-1][KK-1][LL-1][j][nele_index];
            xxiixjjykk1=(1.0-dtheta)*xxiixjjkk1+dtheta*xxiixjj1kk1;
            xxiixjjykk=(1.0-dtheta)*xxiixjjkk+dtheta*xxiixjj1kk;
            xxiixjjykkz1=(1.0-dfi)*xxiixjjykk+dfi*xxiixjjykk1;

            xxiixjj1kk12=(1.0-dtheta0)*self.rbar0[II-1][JJ][KK][LL][j][nele_index]+dtheta0*self.rbar0[II][JJ][KK][LL][j][nele_index];
            xxiixjjkk12=(1.0-dtheta0)*self.rbar0[II-1][JJ-1][KK][LL][j][nele_index]+dtheta0*self.rbar0[II][JJ-1][KK][LL][j][nele_index];
            xxiixjj1kk2=(1.0-dtheta0)*self.rbar0[II-1][JJ][KK-1][LL][j][nele_index]+dtheta0*self.rbar0[II][JJ][KK-1][LL][j][nele_index];
            xxiixjjkk2=(1.0-dtheta0)*self.rbar0[II-1][JJ-1][KK-1][LL][j][nele_index]+dtheta0*self.rbar0[II][JJ-1][KK-1][LL][j][nele_index];
            xxiixjjykk12=(1.0-dtheta)*xxiixjjkk12+dtheta*xxiixjj1kk12;
            xxiixjjykk2=(1.0-dtheta)*xxiixjjkk2+dtheta*xxiixjj1kk2;
            xxiixjjykkz2=(1.0-dfi)*xxiixjjykk2+dfi*xxiixjjykk12;

            xxiixjjykkz=(1.0-dwater)*xxiixjjykkz1+dwater*xxiixjjykkz2;
            self.rbar[j][nele_index]=xxiixjjykkz;

            xxiixjj1kk1=(1.0-dtheta0)*self.fd0[II-1][JJ][KK][LL-1][j][nele_index]+dtheta0*self.fd0[II][JJ][KK][LL-1][j][nele_index];
            xxiixjjkk1=(1.0-dtheta0)*self.fd0[II-1][JJ-1][KK][LL-1][j][nele_index]+dtheta0*self.fd0[II][JJ-1][KK][LL-1][j][nele_index];
            xxiixjj1kk=(1.0-dtheta0)*self.fd0[II-1][JJ][KK-1][LL-1][j][nele_index]+dtheta0*self.fd0[II][JJ][KK-1][LL-1][j][nele_index];
            xxiixjjkk=(1.0-dtheta0)*self.fd0[II-1][JJ-1][KK-1][LL-1][j][nele_index]+dtheta0*self.fd0[II][JJ-1][KK-1][LL-1][j][nele_index];
            xxiixjjykk1=(1.0-dtheta)*xxiixjjkk1+dtheta*xxiixjj1kk1;
            xxiixjjykk=(1.0-dtheta)*xxiixjjkk+dtheta*xxiixjj1kk;
            xxiixjjykkz1=(1.0-dfi)*xxiixjjykk+dfi*xxiixjjykk1;

            xxiixjj1kk12=(1.0-dtheta0)*self.fd0[II-1][JJ][KK][LL][j][nele_index]+dtheta0*self.fd0[II][JJ][KK][LL][j][nele_index];
            xxiixjjkk12=(1.0-dtheta0)*self.fd0[II-1][JJ-1][KK][LL][j][nele_index]+dtheta0*self.fd0[II][JJ-1][KK][LL][j][nele_index];
            xxiixjj1kk2=(1.0-dtheta0)*self.fd0[II-1][JJ][KK-1][LL][j][nele_index]+dtheta0*self.fd0[II][JJ][KK-1][LL][j][nele_index];
            xxiixjjkk2=(1.0-dtheta0)*self.fd0[II-1][JJ-1][KK-1][LL][j][nele_index]+dtheta0*self.fd0[II][JJ-1][KK-1][LL][j][nele_index];
            xxiixjjykk12=(1.0-dtheta)*xxiixjjkk12+dtheta*xxiixjj1kk12;
            xxiixjjykk2=(1.0-dtheta)*xxiixjjkk2+dtheta*xxiixjj1kk2;
            xxiixjjykkz2=(1.0-dfi)*xxiixjjykk2+dfi*xxiixjjykk12;

            xxiixjjykkz=(1.0-dwater)*xxiixjjykkz1+dwater*xxiixjjykkz2;
            self.fd[j][nele_index]=xxiixjjykkz;

            xx1=(1.0-dtheta0)*self.lppdf0[II-1][LL][j][nele_index]+dtheta0*self.lppdf0[II][LL][j][nele_index];
            xx12=(1.0-dtheta0)*self.lppdf0[II-1][LL-1][j][nele_index]+dtheta0*self.lppdf0[II][LL-1][j][nele_index];

            xx2=(1.0-dtheta0)*self.lppdi0[II-1][LL][j][nele_index]+dtheta0*self.lppdi0[II][LL][j][nele_index];
            xx22=(1.0-dtheta0)*self.lppdi0[II-1][LL-1][j][nele_index]+dtheta0*self.lppdi0[II][LL-1][j][nele_index];

            xx3=(1.0-dtheta0)*self.rbarp0[II-1][LL][j][nele_index]+dtheta0*self.rbarp0[II][LL][j][nele_index];
            xx32=(1.0-dtheta0)*self.rbarp0[II-1][LL-1][j][nele_index]+dtheta0*self.rbarp0[II][LL-1][j][nele_index];

            xx4=(1.0-dtheta0)*self.fdp0[II-1][LL][j][nele_index]+dtheta0*self.fdp0[II][LL][j][nele_index];
            xx42=(1.0-dtheta0)*self.fdp0[II-1][LL-1][j][nele_index]+dtheta0*self.fdp0[II][LL-1][j][nele_index];

            self.lppdf[j][nele_index]=(1.0-dwater)*xx12+dwater*xx1;
            self.lppdi[j][nele_index]=(1.0-dwater)*xx22+dwater*xx2;
            self.rbarp[j][nele_index]=(1.0-dwater)*xx32+dwater*xx3;
            self.fdp[j][nele_index]=(1.0-dwater)*xx42+dwater*xx4;

    return True


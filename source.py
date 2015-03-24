#!/usr/bin/env python
# coding=utf-8
import sys
import os
import util as u
import datetime
import numpy as np
import math

class SREF():
    def __init__(self):
        self.year = 1992
        self.month = 1
        self.day = 10
        self.jday = 10
        self.gmt = 0.0
        self.ref = 0.0

class ANGLE():
    def __init__(self):
        self.solzen = 0.0
        self.senzen = 0.0
        self.relazi = 0.0
        
class BASE():

    def __init__(self, fp = '',row = 0, col = 0):
        if u.file_exist(fp) == False:
            raise IOError('file not Exists: %s!' % fp)
        self.filepath = fp
        self.col = col
        self.row = row
        self.DN = self.initDN()
    
    def initDN(self):
        return np.zeros((self.row, self.col))


REF_SCALE = 0.0001

class Reflectance(BASE):

    def __init__(self, fp = '',row = 0, col = 0):
        BASE.__init__(fp, row, col)
        self.ref = SREF()
        self.filenum = 0
        self.filename = []
        self.refl = np.zeros(col)
        self.ref_update = self.initDN()
        self.ref_QC_MODIS = self.initDN()
        self.ref_QC = self.initDN()
        self.readFileName()

    def readFileName(self):
        self.filename = u.readTxt(self.filepath)
        self.filenum = len(self.filename)
    def setRef(self, nyear, nmonth, nday, jday, gmt, ref):
        self.ref.year = nyear
        self.ref.month = nmonth
        self.ref.day = nday
        self.ref.jday = jday
        self.ref.gmt = gmt
        self.ref = ref

    def getFileIndexYearDay(self, nyear, nday):
        pass
    
    def readRefQCMatrixFromHDFFile(self):
        index = self.getFileIndexYearDay()
        if index < 0 or index > self.filenum - 1:
            print "read reflectance file failed!"
            return None
        file = self.filename[index]
        strset = 'g2_sur_refl_b03'
        data = u.readhdf(file, strset)
        for i in self.row:
            for j in self.col:
                self.ref_update[i][j] = 0.05
                self.ref_update[i][j] = data[i*self.col + j] * REF_SCALE
        strset = 'sur_refl_qc_500m'
        data = u.readhdf(file, strset)
        for i in self.row:
            for j in self.col:
                self.ref_QC_MODIS[i][j] = data[i*self.col + j]
        strset = 'sur_refl_qc_500m+'
        data = u.readhdf(file, strset)
        for i in self.row:
            for j in self.col:
                self.ref_QC[i][j] = data[i*self.col + j]

class LatLon(BASE):

    def __init__(self, fp = '',row = 0, col = 0):
        BASE.__init__(fp, row, col)
        self.DN = u.readBinary(fp, size=row*col)
        if row*col != self.DN.size:
            raise ValueError('file size not equel row,col')
        self.DN = self.DN.reshape(row, col)

class DEM(BASE):

    def __init__(self, fp = '',row = 0, col = 0):
        BASE.__init__(fp, row, col)
        self.DN = u.readBinary(fp, size=row*col)
        if row*col != self.DN.size:
            raise ValueError('file size not equel row,col')
        self.DN = self.DN.reshape(row, col)
from spa import *
class Satellite(BASE):

    def __init__(self, fp = '',row = 0, col = 0, tag = ''):
        BASE.__init__(self, fp, row, col)
        self.filetime = u.dt() 
        self.jdays = 0
        self.fgmt = 0.0
        if fp != '' and tag != '' and  os.path.exists(fp) == True:
            self.filetime, self.jdays = self.calculate_time(tag)
        self.angle_pixel = ANGLE()
        self.angle = [ANGLE() for i in range(self.col)]

    def calibrate(self):
        for i in range(self.row):
            for j in range(self.col):
                self.DN[i, j] = self.calibrate(self.DN[i, j])

    def calibrate_pixel(self, data):
        if data < 1024:
            data = (-0.001 + (data * 1.001/1024.0)) * 435.559
        else :
            data = 435.559
        return data

    def calculate_time(self, tag=''):
        try:
            ta = self.filepath.find(tag) + len(tag) - 1
            year = int(self.filepath[ta : ta + 4])
            month = int(self.filepath[ta + 4: ta + 6])
            day = int(self.filepath[ta + 6: ta + 8])
            hour = int(self.filepath[ta + 8: ta + 10])
            minute = int(self.filepath[ta + 10: ta + 12])
            days = u.get_days(year)
            jdays = sum(days[: month - 1]) + day
            ct = u.dt(year = year, month = month, day = day, hour = hour, minute = minute)
            return ct, jdays
        except Exception, e:
            print e
    def calLineOfAngles(self, lat, lon, z, spa, sensorlon = 0.0):
        for i in range(self.col):
            if lon[i] > 0:
                spa.timezone = (lon[i] + 7.5) / 15
            else :
                spa.timezone = (lon[i] - 7.5) / 15
            ltime = self.fgmt + lon[i] / 15.0
            if ltime < 0 :
                ltime += 24
            if ltime > 24:
                ltime -= 24
            lhour = math.floor(ltime)
            lminute = (ltime - lhour) * 60.0
            spa.hour = lhour
            spa.minute = lminute
            spa.second = 0
            spa.delta_t = 67
            spa.longitude = lon[i]
            spa.latitude  = lat[i]
            spa.elevation = z[i]
            spa.pressure  = 1013
            spa.temperature = 283
            spa.slope = 0
            spa.azm_rotation = 0
            spa.atmos_refract = 0
            spa.function = SPA_ALL
            spa.year = self.year
            spa.month = self.month
            spa.day  = self.day
            spa.oorbitheight = 36000
            spa.sensorlon = sensorlon
            spa = spa_calculate(spa)
            spa = spa_calculate_angle(spa)
            self.angle[i].solzen = spa.zenith
            self.angle[i].senzen = spa.senzenith
            self.angle[i].relazi = spa.relazimuth
        return True

class MSG(Satellite):

    def __init__(self, fp = '', row = 0, col = 0, tag = ''):
        Satellite.__init__(fp, row , col,tag )
    
    def readDN(self):
        strset = 'ch1'
        self.DN = u.readhdf(self.filepath, strset)

class MTSAT(Satellite):

    def __init__(self, fp = '', row = 0, col = 0, tag = ''):
        Satellite.__init__(fp, row ,col , tag)

    def readDN(self, row = -1):
        if row == -1 :
            self.DN = u.readBinary(self.filepath, dtype = int, size = self.row * self.col)
        else :
        #this function need to be updated!
            self.DN[row] = u.readBinary(self.filepath, dtype = int, size = self.row * self.col)[row]
        return True
            

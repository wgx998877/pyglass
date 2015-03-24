#!/usr/bin/env python
# coding=utf-8
import sys
import os
import util as u
import datetime
import numpy as np

class SREF():
    def __init__(self):
        self.year = 1992
        self.month = 1
        self.day = 10
        self.jday = 10
        self.gmt = 0.0
        self.ref = 0.0

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

class Satellite(BASE):

    def __init__(self, fp = '',row = 0, col = 0, tag = ''):
        BASE.__init__(self, fp, row, col)
        self.filetime = datetime.datetime.now()
        self.jdays = 0
        if fp != '' and tag != '' and  os.path.exists(fp) == True:
            self.filetime, self.jdays = self.calculate_time(tag)


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
            ct = datetime.datetime(year = year, month = month, day = day, hour = hour, minute = minute)
            return ct, jdays
        except Exception, e:
            print e
            return datetime.datetime.now()




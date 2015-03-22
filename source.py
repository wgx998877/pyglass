#!/usr/bin/env python
# coding=utf-8
import sys
import os
import util as u
import datetime
import numpy as np

class BASE():

    def __init__(self, fp = '',row = 0, col = 0):
        self.filepath = fp
        self.col = col
        self.row = row
        self.DN = self.initDN()
    
    def initDN(self):
        return np.zeros((self.row, self.col))

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




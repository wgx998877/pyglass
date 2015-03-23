#!/usr/bin/python
# Filename: util.py
# Author: wgx
import os
from pyhdf.SD import SD, SDC
import numpy as np
import datetime


def readhdf(filename, fieldname=''):
    hdf = SD(filename, SDC.READ)
    data = hdf.select(fieldname)[:].copy()
    hdf.end()
    return data

def createhdf(filename, table=[]):
    hdf = SD(filename,SDC.CREATE|SDC.WRITE)
    for i in table:
        hdf.create(i, SDC.FLOAT32,0)
    hdf.end()
    return True

def appendhdf(filename, table='', row = 0, data = []):
    hdf = SD(filename, SDC.WRITE|SDC.CREATE)
    attr = hdf.select(table)[:]
    attr[row] = data
    return True

#def readBinary(filename, type='i',size)


def lsfiles(path=os.getcwd(),keys=''):
    result = []
    if file_exist(path) == False:
        raise IOError('Path not Exists!')
    for root,dirs,files in os.walk(path):
        for fp in files:
            if keys in fp:
                result.append(os.path.join(root,fp))
    return result

def file_exist(filepath = ''):
    if filepath == '' or os.path.exists(filepath) == False:
        return False
    return True

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

def get_days(year=2009):
    mon_num = [31,28,31,30,31,30,31,31,30,31,30,31]
    if is_r(int(year)) and year != 0:
        mon_num[1] = 29
    return mon_num

def is_r(year):
    r = False
    if year % 4==0:
        r = True
    if year%100==0 and year%400!=0:
        r = False
    return r

class dt():
    def __init__(self, year = 1990, month = 1, day = 1,hour=0,minute=0,second=0,tz=0):
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minute = minute
        self.second = second
        self.tz = tz

class GMT(datetime.tzinfo):
    def __init__(self, hours=0):
        datetime.tzinfo.__init__(self)
        self.delta = datetime.timedelta(hours = hours)
    def utc_offset(self, dt):
        return self.delta
    def tz_name(self, dt):
        return "GMT+%d" % self.delta




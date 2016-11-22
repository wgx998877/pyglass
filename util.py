#coding:utf-8
#!/usr/bin/python
# Filename: util.py
# Author: wgx
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import os
from pyhdf.SD import SD, SDC
import numpy as np
import datetime
from netCDF4 import Dataset
import math
dpath = os.path.split(os.path.realpath(__file__))[0]

def readhdf(filename, fieldname, ignoreE = True):
    try:
        hdf = SD(filename, SDC.READ)
        data = hdf.select(fieldname)[:].copy()
        hdf.end()
    except Exception, e:
        if not ignoreE:print e
        data = Dataset(filename)[fieldname][:].copy()
    return data


def createhdf(filename, table=[]):
    hdf = SD(filename, SDC.CREATE | SDC.WRITE)
    for i in table:
        hdf.create(i, SDC.FLOAT32, 0)
    hdf.end()
    return True


def appendhdf(filename, table='', row=0, data=[]):
    hdf = SD(filename, SDC.WRITE | SDC.CREATE)
    attr = hdf.select(table)[:]
    attr[row] = data
    return True


def ll2xy(lat, lon, sat='avhrr'):
    if sat == 'avhrr':
        x = 0 + (8998 - lat * 100.0)/5
        y = 0 + (lon * 100.0 + 17998)/5
    elif sat == 'gewex':
        x = 90 - int(lat)
        y = int(lon) + 180
    x = int(x)
    y = int(y)
    return x, y
# print ll2xy(89.98, -179.98)

def xy2ll(x, y, sat='avhrr'):
    if sat == 'avhrr':
        lat = (8998 - 5.0*x)/100.0
        lon = (5.0*y - 17998)/100.0
    lat = float(lat)
    lon = float(lon)
    return lat, lon
    

def readBinary(filename, dtype=float, size=-1):
    file = np.fromfile(filename, dtype=dtype, count=size)
    return file


def writeBinary(filename, data):
    data.tofile(filename)
    return True


def loadDataSet(filename, delim='\t'):
    fr = open(filename)
    strArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in strArr]
    return np.mat(datArr)


def loadCsv(filename):
    return loadDataSet(filename, ',')


def readTxt(filename):
    f = open(filename)
    data = []
    for i in f:
        data.append(i.strip())
    f.close()
    return data

def list2Txt(data, filename):
    fo = open(filename, 'w')
    for i in data:
        fo.write(str(i))
    fo.close()

def lsfiles(path=os.getcwd(), keys=''):
    result = []
    if file_exist(path) == False:
        raise IOError('Path not Exists!')
    for root, dirs, files in os.walk(path):
        for fp in files:
            if keys in fp:
                result.append(os.path.join(root, fp))
    return result


def file_exist(filepath=''):
    if filepath == '' or os.path.exists(filepath) == False:
        return False
    return True


def num_to_day(str):
    year = str[:4]
    num = int(str[4:])
    mon_num = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if is_r(int(year)):
        mon_num[1] = 29
    m = 0
    d = 0
    for i in range(12):
        if num - mon_num[i] > 0:
            num -= mon_num[i]
        else:
            m = i + 1
            d = num
            break
    return "%4s%02d%02d" % (year, m, d)


def get_days(year=2009):
    mon_num = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if is_r(int(year)) and year != 0:
        mon_num[1] = 29
    return mon_num


def is_r(year):
    r = False
    if year % 4 == 0:
        r = True
    if year % 100 == 0 and year % 400 != 0:
        r = False
    return r

zhou = np.fromfile(dpath+'/data/zhou',dtype=np.float32).reshape(720,1440)
zhou3 = np.fromfile(dpath+'/data/zhou3',dtype=np.float32).reshape(180,360)
country = np.fromfile(dpath+'/data/country',dtype=np.float32).reshape(180,360)
zret = {}
greenlandmap = np.load(dpath+'/data/greenland.npy')

def xy2zhou(x,y):
    global zret
    t = "%f,%f" % (x,y)
    if t not in zret:
        zret[t] = int(zhou3[x,y])
    return zret[t]
    
def greenland(x, y):
    return greenlandmap[x, y] == 1
    
def getgewex():
    f = '/gewexall.npy'
    r = np.load(dpath + f)[:,:,6:]
    print 'get monthly gewex, from 1984.1 to 2007.12, shape:', r.shape
    return r
    
def getceres():
    f = '/ceres.npy'
    r = np.load(dpath + f)[:,:,10:-9]
    print 'get monthly ceres, from 2001.1 to 2013.12, shape:', r.shape
    return r
    
def xy2country(x,y):
    global zret
    t = "%f,%f" % (x,y)
    if t not in zret:
        zret[t] = (country[x,y])
    return zret[t]
    
def ll2zhou(lat, lon):
    line = "%s,%s" % (lat, lon)
    global zret
    if line in zret:
        return zret[line]
    lat = int ((90 - lat) * 4)
    lon = int ((lon + 180) * 4)
    if lat not in range(720) or lon not in range(1440):
        return 0
    global zhou
    #zhou = np.fromfile('data/zhou',dtype=np.float32).reshape(720,1440)
    ret = int(zhou[lat, lon])
    zret[line] = ret
    return ret
    if ret == 0:
        try:
            sta = [
            int(zhou[lat-1, lon]),
            int(zhou[lat, lon-1]),
            int(zhou[lat-1, lon-1]),
            int(zhou[lat, lon]),
            int(zhou[lat+1, lon]),
            int(zhou[lat, lon+1]),
            int(zhou[lat-1, lon+1]),
            int(zhou[lat+1, lon-1]),
            int(zhou[lat+1, lon+1])]
            for i in sta:
               if i != 0:
                   ret = i
                   break
        except:
            pass
    return ret
    
def zhoudic():
    return {1:"North America",2:"Europe+Russia",3:"South America",4:"Africa",5:"Asia",6:"Oceania",7:"Antarctica",0:"Ocean"}
    return {1:"北美洲",2:"欧洲+俄罗斯",3:"南美洲",4:"非洲",5:"亚洲",6:"大洋洲",7:"南极洲",0:"海洋"}
#1 北美
#2 欧洲
#3 南美
#4 非洲
#5 亚洲
#6 大洋洲
#7 南极洲
x2cache = {}
def x2cos(x):
    global x2cache
    if x in x2cache:
        return x2cache[x]
    wd = 90 - x - 0.5
    if x >= 90:
        wd = x - 90 + 0.5
    wd = math.radians(wd)
    wd = np.cos(wd)
    x2cache[x] = wd
    return wd
    
#from mis import *
def rshape(data):
    #x, y, l = data.shape
    #r = np.zeros((x, y, l))
    x, y = data.shape
    r = np.zeros((x, y))
    #data = data[::-1, :]
    for x in range(180):
        for y in range(360):
            yt = y + 180
            if yt >= 360:
                yt -= 360
            r[x, yt] = data[x, y]
    return r
            
 
def readInfo():
    path = "D:\\BSRN_Processed\\BSRN.txt"
    l = readTxt(path)[1:]
    ret = {}
    for i in l:
        i = i.strip().split()
        i[1], i[2], i[3] = float(i[1]), float(i[2]), float(i[3])
        ret[i[0]] = (i[1], i[2])#, i[3])
    return ret   

    
def get_month(start, end):
    import datetime as dt
    x = []
    for i in range(start,end):
        for j in range(1,13):
            x.append(dt.datetime(i,j,15))
    return x
    
def p(x,y,n = 3):
    pn = np.poly1d(np.polyfit(x,y,n))
    return pn
    
ludic = {
1:"Evergreen Needleleaf forest",
2:"Evergreen Broadleaf forest",
3:"Deciduous Needleleaf forest",
4:"Deciduous Broadleaf forest",
5:"Mixed forest",
6:"Closed shrublands",
7:"Open shrublands",
8:"Woody savannas",
9:"Savannas",
10:"Grasslands",
11:"Permanent wetlands",
12:"Croplands",
13:"Urban and built-up",
14:"Cropland/Natural vegetation mosaic",
15:"Snow and ice",
16:"Barren or sparsely vegetated",
}
    
class dt():

    def __init__(
            self, year=1990, month=1, day=1, hour=0, minute=0, second=0, tz=0):
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
        self.delta = datetime.timedelta(hours=hours)

    def utc_offset(self, dt):
        return self.delta

    def tz_name(self, dt):
        return "GMT+%d" % self.delta

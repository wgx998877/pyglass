#!/usr/bin/env python
# coding=utf-8

from data.site_data import get_daily_data, get_month_data
from util import readhdf, ll2xy

sat = ''
filelist = ''

def main():
    files = [i.split() for i in filelist]
    sites = get_daily_data(2010)
    result = []
    for f in files:
        d = readhdf(f, 'daily_integration')
        t = f[:]
        for s in sites:
            lat, lon = s.lat, s.lon
            data = s.data
            x, y = ll2xy(lat, lon, sat)
            if t == data[0]:
                tmp = [s.id, s.name, s.lat, s.lon, s.alt, data[0], data[1], d[x, y]
                result.append(tmp)
    return result

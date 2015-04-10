#!/usr/bin/env python
# coding=utf-8

from data.site_data import get_daily_data, get_month_data
from util import readhdf, ll2xy, num_to_day, readTxt, list2Txt

sat = 'avhrr'
filelist = 'G:\\File_Gen\\daily_integrated\\filelist.txt'
field = 'DSSR_daily_integrated'

def main():
    files = readTxt(filelist)
    sites = get_daily_data(2010)
    result = []
    for f in files:
        print f
        d = readhdf(f, field)
        jDate = f[f.find("V01.A")+5:f.find("V01.A")+12]
        Date = num_to_day(jDate)
        for s in sites:
            lat, lon = s.lat, s.lon
            data = s.data
            x, y = ll2xy(lat, lon, sat)
            for i in data:
                if Date == i[0]:
                    #tmp = [s.id, s.name, s.lat, s.lon, s.alt, i[0], i[1], d[x, y]]
                    tmp = "%s,%s,%f,%f,%f,%s,%f,%f\n" % (s.id, s.name, s.lat, s.lon, s.alt, i[0], float(i[1])*10000000/3600/24/5000/5000, d[x, y])
                    result.append(tmp)
                    break
    print "done"
    return result

r = main()
print len(r)
list2Txt(r, "out2")
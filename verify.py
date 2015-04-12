#!/usr/bin/env python
# coding=utf-8

from data.site_data import get_daily_data, get_month_data
from util import readhdf, ll2xy, num_to_day, readTxt, list2Txt

sat = 'avhrr'
def daily():
    filelist = 'G:\\File_Gen\\daily_integrated\\filelist.txt'
    field = 'DSSR_daily_integrated'
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
                    tmp = "%s,%s,%f,%f,%f,%s,%f,%f\n" % (s.id, s.name, s.lat, s.lon, s.alt, i[0], float(i[1])*10000/3600/24, d[x, y])
                    result.append(tmp)
                    break
    print "done"
    return result

def monthly():
    filelist = "G:\\File_Gen\\monthly_integrated\\filelist.txt"
    field = "DSSR_monthly_integrated"
    files = readTxt(filelist)
    sites = get_month_data(2010)
    result = []
    for f in files:
        print f
        d = readhdf(f, field)
        Date = f[f.find("V01.A")+5:f.find("V01.A")+11]
        #Date = num_to_day(jDate)
        for s in sites:
            id = s[0]
            t = "%d%02d" % (s[4] ,s[5])
            
            if Date != t:
                continue
            lat, lon = s[1], s[2]
            r = s[6]
            alt = s[3]
            x, y = ll2xy(lat, lon, sat)
            tmp = "%s,%f,%f,%f,%s,%f,%f\n" % (id, lat, lon, alt, Date, float(r), d[x, y])
            result.append(tmp)
    return result
    
if __name__ == "__main__":
    r = monthly()
    print len(r)
    list2Txt(r, "out4")
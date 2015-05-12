#!/usr/bin/env python
# coding=utf-8

from data.site_data import *
from util import *
from cal import *

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
                    #tmp = [s.id, s.name, s.lat, s.lon, s.alt, i[0], float(i[1])*10, d[x, y]]
                    #tmp = "%s,%s,%f,%f,%f,%s,%f,%f\n" % (s.id, s.name, s.lat, s.lon, s.alt, i[0], float(i[1])*10000/3600/24, d[x, y])
                    tmp = "%s,%s,%f,%f,%f,%s,%f,%f\n" % (s.id, s.name, s.lat, s.lon, s.alt, i[0], float(i[1])*10000/3600/24, float(d[x, y])*100000/3600/24)
                    result.append(tmp)
                    break
    list2Txt(result, "out_daily")
    print "done"
    return result

def daily_ex(filelist = "..\\station_data"):
    files = lsfiles(filelist)
    sites = {}
    site_info = get_site_info_ex()
    for f in files:
        fr = open(f)
        siteid = f[-9:-4]
        sites[siteid] = {}
        for i in fr:
            i = i.strip().split()
            t = "%s%s%s" % (i[0], i[1], i[2])
            sites[siteid][t] = float(i[3])
        fr.close()
    print 'site_info_ex done!'
    filelist = 'G:\\File_Gen\\daily_integrated\\filelist.txt'
    field = 'DSSR_daily_integrated'
    files = readTxt(filelist)
    #sites = get_daily_data(2010)
    result = []
    for f in files:
        print f
        d = readhdf(f, field)
        jDate = f[f.find("V01.A")+5:f.find("V01.A")+12]
        Date = num_to_day(jDate)
        for s in sites:
            if s not in site_info:
                continue
            data = sites[s]
            lat, lon = float(site_info[s][0]), float(site_info[s][1])
            alt = float(site_info[s][2])
            x, y = ll2xy(lat, lon, sat)
            if Date in sites[s]:
                #tmp = "%s,unknown,%f,%f,%f,%s,%f,%f\n" % (s, lat, lon, alt, Date, float(data[Date])*100000/3600/24, d[x, y])
                tmp = "%s,unknown,%f,%f,%f,%s,%f,%f\n" % (s, lat, lon, alt, Date, float(data[Date]), float(d[x, y])/3600/24)
                result.append(tmp)
                
    list2Txt(result, "out_daily_ex")
    print "done:" + str(len(result))
    return result
    
def check_daily_ex(filelist = "..\\station_data"):
    files = lsfiles(filelist)
    sites = {}
    site_info = get_site_info_ex()
    for f in files:
        fr = open(f)
        siteid = f[-9:-4]
        sites[siteid] = {}
        for i in fr:
            i = i.strip().split()
            t = "%s%s%s" % (i[0], i[1], i[2])
            sites[siteid][t] = float(i[3])
        fr.close()
    print 'site_info_ex done!'
    filelist = 'G:\\File_Gen\\daily_integrated\\filelist.txt'
    field = 'DSSR_daily_integrated'
    files = readTxt(filelist)
    #sites = get_daily_data(2010)
    result = []
    for f in files:
        print f
        d = readhdf(f, field)
        jDate = f[f.find("V01.A")+5:f.find("V01.A")+12]
        Date = num_to_day(jDate)
        for s in sites:
            if s not in site_info:
                continue
            data = sites[s]
            lat, lon = float(site_info[s][0]), float(site_info[s][1])
            alt = float(site_info[s][2])
            x, y = ll2xy(lat, lon, sat)
            if Date in sites[s]:
                sdata = float(data[Date])
                rdata = float(d[x, y])*100000/3600/24
                if rdata>200 and sdata in range(0,120):
                    #tmp = "%s,unknown,%f,%f,%f,%s,%f,%f\n" % (s, lat, lon, alt, Date, float(data[Date])*100000/3600/24, d[x, y])
                    tmp = "%s,unknown,%f,%f,%f,%s,%f,%f\n" % (s, lat, lon, alt, Date, sdata, rdata)
                    result.append(tmp)
                
    list2Txt(result, "out_daily_ex_check")
    print "done:" + str(len(result))
    return result
    
def monthly():
    filelist = "G:\\File_Gen\\monthly_integrated\\filelist.txt"
    field = "DSSR_monthly_integrated"
    files = readTxt(filelist)
    sites = get_month_data(2010)
    result = []
    days = get_days(2010)
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
            tmp = "%s,unknonwn,%f,%f,%f,%s,%f,%f,%f\n" % (id, lat, lon, alt, Date, float(r), float(d[x, y])*100000/24/3600/days[s[5]-1], float(d[x,y]))
            result.append(tmp)
    list2Txt(result, "out_monthly")
    return result
    
def rstest():
    filelist = "G:\\File_Gen\\monthly_integrated\\filelist.txt"
    field = "DSSR_monthly_integrated"
    files = readTxt(filelist)
    cnt = 0
    data = []
    r = np.ones((3600,7200))
    for f in files:
        d = readhdf(f, field)
        Date = f[f.find("V01.A")+5:f.find("V01.A")+11]
        data.append(d)
    for i in range(1500,1700):
        print i
        for j in range(7200):
            r[i,j]=0
            x = []
            
            for k in data:
                x.append(k[i, j])
            t=RS(x)[0]
            r[i,j]= t
    r.tofile("prs.bin")
    
def mktest():
    filelist = "G:\\File_Gen\\monthly_integrated\\filelist.txt"
    field = "DSSR_monthly_integrated"
    files = readTxt(filelist)
    cnt = 0
    data = []
    r = np.ones((3600,7200))
    for f in files:
        d = readhdf(f, field)
        Date = f[f.find("V01.A")+5:f.find("V01.A")+11]
        data.append(d)
    for i in range(0,3600):
        print i
        for j in range(7200):
            r[i,j]=0
            x = []
            
            for k in data:
                x.append(float(k[i, j]))
            t=mk(x)
            r[i,j]= t
    r.tofile("t.bin")
        
    
if __name__ == "__main__":
    rstest()
    #daily()
    #daily_ex()
    #monthly()
    #check_daily_ex()
    #mktest()
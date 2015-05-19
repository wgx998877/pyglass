#!/usr/bin/env python
# coding=utf-8

from data.site_data import *
from util import *
from cal import *

sat = 'avhrr'
def daily():
    filelist = 'G:\\File_Gen2\\daily_integrated\\filelist.txt'
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
    filelist = 'G:\\File_Gen1\\daily_integrated\\filelist.txt'
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
                rdata = float(d[x, y])*100000.0/3600.0/24.0
                #tmp = "%s,unknown,%f,%f,%f,%s,%f,%f\n" % (s, lat, lon, alt, Date, float(data[Date])*100000/3600/24, d[x, y])
                #tmp = "%s,unknown,%f,%f,%f,%s,%f,%f\n" % (s, lat, lon, alt, Date, float(data[Date]), float(d[x, y])/3600/24)
                tmp = "%s,unknown,%f,%f,%f,%s,%f,%f\n" % (s, lat, lon, alt, Date, sdata, rdata)
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
    filelist = 'G:\\File_Gen2\\daily_integrated\\filelist.txt'
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
    filelist = "G:\\File_Gen2\\monthly_integrated\\filelist.txt"
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
    
def rstest(row=3600, col=7200):
    filelist = "G:\\File_Gen\\monthly_integrated\\filelist.txt"
    field = "DSSR_monthly_integrated"
    filelist = "I:\\GEWEX_SW_Monthly\\filelist.txt"
    field = "par"
    files = readTxt(filelist)
    cnt = 0
    data = []
    r = np.ones((row,col))
    for f in files:
        d = readhdf(f, field)
        #Date = f[f.find("V01.A")+5:f.find("V01.A")+11]
        data.append(d[0])
    for i in range(0,row):
        print i
        for j in range(col):
            r[i,j]=0
            x = []
            
            for k in data:
                x.append(k[i, j])
            t=RS(x)[0]
            r[i,j]= t
    r.tofile("rs.bin")
    
def mktest(row=3600, col=7200):
    filelist = "G:\\File_Gen\\monthly_integrated\\filelist.txt"
    field = "DSSR_monthly_integrated"
    filelist = "I:\\GEWEX_SW_Monthly\\filelist.txt"
    field = "par"
    files = readTxt(filelist)
    cnt = 0
    data = []
    r = np.ones((row,col))
    for f in files:
        d = readhdf(f, field)
        #Date = f[f.find("V01.A")+5:f.find("V01.A")+11]
        data.append(d[0])
    for i in range(0,row):
        print i
        for j in range(col):
            r[i,j]=0
            x = []
            
            for k in data:
                x.append(float(k[i, j]))
            t=mk(x)
            r[i,j]= t
    r.tofile("mk.bin")

def get_site_year_data():
    s = get_year_data()
    sx = []
    for i in s:
        if i[4]['range'][0]>1961 or i[4]['range'][1]<2010:
            continue
        del i[4]['range']
        sx.append(i)
    print "get site num : %d" % len(sx)
    r = {}
    result = []
    for i in sx:
        i = i[4]
        for j in i:
            if j not in range(1961,2011):
                continue
            if j not in r:
                r[j] = 0
            r[j] += i[j]['ave']
    for i in r:
        result.append(r[i])
    ave = sum(result)*1.0/len(result)
    for i in range(len(result)):
        result[i] -= ave
    years = r.keys()
    return years, result

def site_emd():
    years, result = get_site_year_data()
    n = len(result)
    c = []
    cn = 7
    sd_max = 300
    alp = 0.3
    xdel = 0.05 
    xnew = np.arange(min(years), max(years)+1, xdel)
    xnew = xnew[:-1/xdel+1]
    Y = result[:]
    c.append(Y[:])
    for i in range(cn):
        h0 ,h1 = Y[:], []
        print i, len(h0), len(h1)
        for j in range(sd_max): 
            xmin, ymin, xmax, ymax = fm(years, h0)
            h1 = []
            #print len(xmin),len(ymin),len(xmax),len(ymax)
            #y = cubic_spline(x = years, y = h0,xnew = xnew)
            #print len(xmin),len(ymin), len(xmax), len(ymax)
            #print (xmin),(ymin), (xmax), (ymax)
            if len(xmin) < 4 or len(xmax) < 4:
                break
            symin = cubic_spline_new(x = xmin, y = ymin,xnew = xnew)
            symax = cubic_spline_new(x = xmax, y = ymax,xnew = xnew)
            smean = []
            nn = len(xnew)
            for k in range(nn):
                smean.append( (symin[k] + symax[k])/2.0)
            for k in range(n):
                h1.append(h0[k] - smean[int(k/xdel)])
            SD = sd(years, h0, h1)
            print i, j, SD
            if SD <= alp: 
                break
            h0 = h1[:]
        if len(h1) == 0:
            c.append(Y[:])
            break
        c.append(h1[:])
        print len(h1)
        #break
        for k in range(n):
            Y[k] -= h1[k]
    color = ['r', 'b', 'g', 'k', 'm', 'c', 'y', '--']
    #return
    import matplotlib.pyplot as plt
    #plt.plot(years,result,'r')
    plt.figure(1)
    plt.suptitle("CMA 54 stations SW data EMD-IMF")
    for i in range(len(c)):
        print len(c[i])
        num = int ("%d1%d" % (len(c), i+1))
        plt.subplot(num)
        print num
        plt.plot(years,c[i],color[i],label=str(i))
    #plt.legend()
    plt.show()
    return
    
    import matplotlib.pyplot as plt
    plt.plot(years,result,'r')
    plt.plot(xmin,ymin,'go')
    plt.plot(xmax,ymax,'bo')
    #plt.plot(xnew,y,'b-')
    plt.plot(xnew,symin,'y-')
    plt.plot(xnew,symax,'g-')
    plt.plot(xnew,smean,'b')
    plt.show()
    
def site_wavelet():
    years, result = get_site_year_data()
    n = len(result)
    y = wavelet(t = 'haar', x = result,a = 32)
    print n, len(y)
    #print y
    import matplotlib.pyplot as plt
    #plt.plot(years,result,'r')
    plt.plot(years,y,'b')
    plt.show()
    
def site_mk_check():
    years, result = get_site_year_data()
    n = len(result)
    ua, ub = mka(result)
    k,b = leastsq(range(n), result)
    print k,b
    ky = [i*1.0*k+b for i in range(n)]
    print len(ua),len(ub),len(years)
    #return 
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    ax1.plot(years, result,'or--')
    ax1.plot(years,ky,':',color='black')
    ax1.set_xlabel('years')
    ax1.set_ylabel('Anomaly W/m2', color='b')
    ax2 = ax1.twinx()
    ax2.plot(years[1:],ua,'b',years[1:],ub,'g')
    ax2.set_ylabel('Mann-Kendall Test', color='r')
    ax1.legend(['Anomaly','Line trend'],loc='upper right')
    ax2.legend(['MK_UA','MK_UB'],loc='lower right')
    plt.annotate(r'Turning point',
             xy=(1986, ua[25]), xycoords='data',
             xytext=(-40, -60), textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    #plt.plot(years, result,'r',years[1:],ua,'b',years[1:],ub,'g')
    plt.title('CMA 54 stations SW data')
    plt.show()
    
    #print sx[0]
    
if __name__ == "__main__":
    #site_emd()
    #site_wavelet()
    #site_mk_check()
    #daily()
    daily_ex()
    #monthly()
    #check_daily_ex()
    #mktest(180,360)
    #rstest(180,360)
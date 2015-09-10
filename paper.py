import os
from util import *
from cal import *
from plot import *
import random
import numpy as np
import math
dp = 'E:\\forpaper\\'

def ceres_check():
    gewex_path = dp + 'ceres.npy'
    #gewex = np.load(gewex_path)
    ceres_kb_path = dp + 'ceresKb.npy'
    ceres_kb = np.load(ceres_kb_path)
    data_name = '1961to2006.npy'
    data = np.load(dp + data_name)
    #data = np.load(dp+'1950to1999.npy')
    l = len(data[0,0])
    ret = np.zeros((180,360,l))
    mseason = [0,1,1,2,2,2,3,3,3,4,4,4,1]
    for i in range(l):
        print i
        month = i % 12 +1 
        for la in range(180):
            for lo in range(360):
                season = mseason[month] - 1
                m = season
                kb = ceres_kb[la,lo,m]
                d = data[la,lo,i]
                k,b = kb[:41], kb[41]
                rad = np.dot(k.reshape(1,41), d.reshape(41,1))[0][0]+b
                ret[la,lo,i] = rad
    np.save(dp+data_name+'.result', ret)

def ceres_learn():
    gewex_path = dp + 'gewex.npy'
    #gewex = np.load(gewex_path)
    ceres_path = dp + 'ceres.npy'
    ceres = np.load(ceres_path)
    data = np.load(dp+'1984to2005.npy')
    print data.shape#198401 -- 200412
    #print gewex.shape#198401 -- 200412
    print ceres.shape#200003 -- 201409
    la, lo = 180,360
    mode_num = 41
    K = np.zeros((la, lo, 4, mode_num + 1))
    mseason = [0,1,1,2,2,2,3,3,3,4,4,4,1]
    for laa in range(la):
        print laa
        for loo in range(lo):
            l = 58
            x = [[], [], [], []]
            y = [[], [], [], []]
            month = 3
            for i in range(l):
                dindex = i + 16 * 12 + 2
                cindex = i
                season = mseason[month]
                month += 1
                if month == 13:
                    month = 1
                xtmp = data[laa,loo,dindex]
                ytmp = ceres[laa,loo,cindex]
                x[season-1].append(xtmp)
                y[season-1].append(ytmp)
                
            for i in range(4):
                #print xsum[i].sum()
                #print y[i].sum()
                k,b = line_regr(x[i],y[i],False)
                #print k,b
                #print np.hstack((k,b))
                K[laa,loo,i] = np.hstack((k,b))
                #print k,b,k.sum()
                #print 'len', len(y[i])
    np.save(dp+'ceresKb', K)
            
    
def forpaper():
    gewex_path = dp + 'gewex.npy'
    gewex = np.load(gewex_path)
    ceres_path = dp + 'ceres.npy'
    ceres = np.load(ceres_path)
    data_84to05 = np.load(dp+'1984to2005.npy')
    print data_84to05.shape#198401 -- 200412
    print gewex.shape#198401 -- 200412
    print ceres.shape#200003 -- 201409
    la, lo = 180,360
    mode_num = 41
    months = 12
    data = data_84to05
    ret = np.zeros((la, lo, months, mode_num+1))
    
    for laa in range(la):
        print laa
        for loo in range(lo):
        
            l = len(data[laa,loo])
            t = l/months
            x = np.zeros((months,t,mode_num))
            y = np.zeros((months,t))
            xsum = np.zeros((months,t))
            #print t,laa,loo
            for i in range(l):
                c = i%months
                m = i/months
                #print i,c,m
                x[c,m] = data[laa,loo,i]
                y[c,m] = gewex[laa,loo,i]
                xsum[c,m] = data[laa,loo].sum()/mode_num
            #print y
            for i in range(months):
                #print xsum[i].sum()
                #print y[i].sum()
                k,b = line_regr(x[i],y[i],False)
                print k,b
                print np.hstack((k,b))
                ret[laa,loo,i] = np.hstack((k,b))
                #print k,b,k.sum()
                return
                
            #return
    #np.save(dp+'gewex_kb', ret)
    
def forcg():
    gewex_path = dp + 'gewexall.npy'
    gewex = np.load(gewex_path)
    ceres_path = dp + 'ceres.npy'
    ceres = np.load(ceres_path)
    print gewex.shape#198307 -- 200712 24*12+6 294
    print ceres.shape#200003 -- 201409 14*12+7 175
    la, lo = 180,360
    months = 12
    ret = np.zeros((la, lo, 2))
    #ret = np.zeros((la, lo, months))
    #2000 03 
    #2007 12  94
    l = 12 * 7
    X,Y = [], []
    for laa in range(la):
        print laa
        for loo in range(lo):
            x = np.zeros((l))
            y = np.zeros((l))
            '''
            l = len(gewex[laa,loo])
            t = l/months
            for i in range(l):
                c = i%months
                m = i/months
                x[c,m] = data[laa,loo,i]
                y[c,m] = gewex[laa,loo,i]
            2001 1
            3 4 5 6 7 8 9 10 11 12 1
            0 1 2 3 4 5 6 7  8  9  10
            '''
            for i in range(l):
                i1, i2 = i+6+17*12, i+10
                #print i1, i2
                #print i1/12+1984,i1%12,(i2)/12+2000,i2%12 
                x[i] = gewex[laa,loo,i1]
                y[i] = ceres[laa,loo,i2]
                X.append(x[i])
                Y.append(y[i])
            k,b = leastsq(x,y)#line_regr(x,y,False)
            ret[laa,loo][0] = k
            ret[laa,loo][1] = b
        print k, b
    #np.save(dp+'gewex_ceres_kb', ret)
    l = len(gewex[0,0])
    result = np.zeros((la, lo, l))
    s = 0
    x, y = [], []
    for laa in range(la):
        for loo in range(lo):
            for i in range(l):
                k, b = ret[laa, loo][0], ret[laa, loo][1]
                result[laa, loo, i] = gewex[laa, loo, i] * k + b
                s += (result[laa, loo, i] - gewex[laa, loo, i]) ** 2
                x.append(gewex[laa, loo, i])
                y.append(result[laa,loo, i])
    #np.save(dp+'gewex_ceres_result', result)
    #s /= (laa*loo*l)
    #print s
    import plot
    plot.point(X,Y)
    print '-'*10
    print r2(X,Y)
    print rmse(X,Y)
    print bias(X,Y)
    print len(X)
    print '-'*10
    print r2(x,y)
    print rmse(x,y)
    print bias(x,y)
    print len(x)
    plot.point(X,Y)
    
    
def getdata(N = 1):
    gewex_path = dp + 'gewex.npy'
    #gewex = np.load(gewex_path)
    gewex_kb_path = dp + 'gewex_kb.npy'
    gewex_kb = np.load(gewex_kb_path)
    #data = np.load(dp+'1984to2005.npy')
    data = np.load(dp+'1950to1999.npy')
    l = len(data[0,0])
    ret = np.zeros((180,360,l))
    for i in range(l):
        print i
        for la in range(180):
            for lo in range(360):
                m = i % 12
                kb = gewex_kb[la,lo,m]
                d = data[la,lo,i]
                k,b = kb[:41], kb[41]
                rad = np.dot(k.reshape(1,41), d.reshape(41,1))[0][0]+b
                ret[la,lo,i] = rad
    np.save(dp+'1950to1999_result', ret)
    return 
    xy = []
    
    import math
    for i in range(N):
        s = 0
        x, y = random.randint(0,180), random.randint(0,360)
        print x,y
        l = len(data[x,y])
        for j in range(l):
            m = j % 12
            if m != 0:continue
            #g = gewex[x,y,j]
            kb = gewex_kb[x,y,m]
            k,b = kb[:41], kb[41]
            d = data[x,y,j]
            rad = np.dot(k.reshape(1,41), d.reshape(41,1))[0][0]+b
            #s += math.fabs(g - rad)
            print j,'k*d:',np.dot(k.reshape(1,41), d.reshape(41,1))[0][0]
            print 'b:',b
            print 'rad:',rad
            #print 'g:',g
            print '--'*12
            #print kb.shape
            #print d.shape
            #print k.shape,b
        print '--'*5, s

def test1():
    gewex = np.load(dp+'gewexall.npy')
    data = np.load(dp+'1980to2006.npy')
    gewex_kb = np.load(dp+'gewex_kb.npy')
    l = len(data[0,0])
    ret = []
    for i in range(l):
        print i,i/12+1980,i%12+1
        for la in range(180):
            for lo in range(360):
                m = i % 12
                #print i/12+1980,m,i
                kb = gewex_kb[la,lo,m]
                d = data[la,lo,i]
                k,b = kb[:41], kb[41]
                rad = np.dot(k.reshape(1,41), d.reshape(41,1))[0][0]+b
                #if rad < 0:
                    #print i,i/12+1980,i%12+1
                    #print rad
                if (i>=42 and i <= 47) or (i>=300 and i<=310):
                    gew = gewex[la,lo,i-42]
                    ret.append([la,lo,i,rad,gew])
    print len(ret)
    x, y = [], []
    d = []
    import matplotlib.pyplot as plt
    for i in ret:
        x.append(i[3])
        y.append(i[4])
        d.append([int(i[3]),int(i[4])])
    print 'r2', r2(x,y)
    print 'rmse', rmse(x,y)
    print 'bias', bias(x,y)
    #plt.plot(x,y,'o')
    #plt.show()
    from pyheatmap.heatmap import HeatMap
    hm = HeatMap(d)
    hm.heatmap(save_as="h.png")
    hmimg = plt.imread("h.png")
    plt.imshow(hmimg, aspect='auto',origin='lower')
    plt.show()
        
def test2():
    X = 130
    Y = 116
    ceres = np.load(dp+'ceres.npy')
    cs = 0
    cx, cy = range(2001,2005), []
    for i in range(4*12):
        cs += ceres[X,Y,i+10]
        if (i+1)%12 == 0:
            cy .append(cs/12.0)
            #print i, cs/12.0
            cs = 0
    gewex = np.load(dp+'gewex.npy')
    
    g, gy = [], []
    gx = range(1984,2005)
    for i in range(len(gx)*12):
        d = gewex[X,Y,i]
        g.append(d)
    gs = 0
    for i in range(len(g)):
        gs += g[i]
        if (i+1)%12 == 0:
            gy.append(gs/12)
            gs = 0
    import data.site_data as sd
    r = sd.get_month_data()
    site = []
    x, y = [], []
    for i in r:
        if i[0] == '54511':
            site.append(i)
            if i[6] < 2000 and i[6] > -10:
                y.append(i[6])
            else:
                print 'xxxxxxx'
    #data = np.load( dp + '1984to2005.npy.result.npy')
    data = np.load( dp + '1961to2006.npy.result.npy')
    #data = np.load( dp + '1950to1999_result.npy')
    xa = range(1961, 2006)
    #print len(data[])
    for i in range(540):
        wd = 90 - X - 0.5
        if X > 90:
            wd = 180 - X - 0.5
        wd = math.radians(wd)
        #d = data[X, Y, i] * np.cos(wd)
        d = data[X, Y, i]
        x.append(d)
    import matplotlib.pyplot as plt
    print 'xy:', len(x), len(y)
    ya ,yb = [], []
    sa ,sb = 0, 0
    for i in range(len(x)):
        sa += x[i]
        if (i+1)%12 == 0:
            ya.append(sa/12.0)
            #print i,(i+1)/12 + 1960, sa/12.0
            sa = 0
    for i in range(len(y)):
        sb += y[i]
        if (i+1)%12 == 0:
            yb.append(sb/12.0)
            sb = 0
    plt.clf()
    year = range(1958,1958+54)
    print len(ya), len(yb), len(gx), len(year)
    tx, ty = [], []
    for i in range(45):
        tx.append(yb[i+4])
        ty.append(ya[i])
        #print yb[i+4], ya[i], yb[i+4]- ya[i]
    print 'len', len(tx), len(ty)
    print 'r2', r2(tx,ty)
    print 'rmse', rmse(tx, ty)
    print 'bias', bias(tx, ty)
    print leastsq(tx, ty)
    point (tx, ty)
    return
    plt.plot(xa,ya,'r', label='ceres_result')
    plt.plot(year,yb,'b',label='cma')
    plt.plot(gx, gy, 'g', label='gewex')
    plt.plot(cx, cy, 'y', label='ceres')
    plt.legend()
    plt.show()
        
def test3():
    X = 115
    Y = 102
    gewex = np.load(dp+'gewex.npy')
    g, gy = [], []
    gx = range(1984,2000)
    for i in range(len(gx)*12):
        d = gewex[X,Y,i]
        g.append(d)
    gs = 0
    for i in range(len(g)):
        gs += g[i]
        if (i+1)%12 == 0:
            gy.append(gs/12)
            gs = 0
    import data.site_data as sd
    r = sd.get_year_data()
    site = []
    x, y = [], []
    xx = range(600)
    xy = range(88, 688+51)
    yb = []
    for i in r:
        if i[0] == '54511':
            d = i[4]
            for i1 in range(1958,2000):
                for i2 in range(1,13):
                    if i2 not in d[i1]:
                        y.append(0)
                    else:
                        y.append(d[i1][i2])
                yb.append(d[i1]['ave'])
                
    print len(y), len(yb)
    data = np.load( dp + '1950to1999_result.npy')
    for i in range(600):
        wd = 90 - X - 0.5
        if X > 90:
            wd = 180 - X - 0.5
        wd = math.radians(wd)
        d = data[X, Y, i]
        #d = data[X, Y, i] * np.cos(wd)
        x.append(d)
    import matplotlib.pyplot as plt
    lx = x[96:]
    ly = y[:]
    ya = []
    sa ,sb = 0, 0
    for i in range(len(lx)):
        sa += lx[i]
        if (i+1)%12 == 0:
            ya.append(sa/12)
            sa, sb = 0, 0
    print len(lx), len(ly)
    print r2(lx,ly)
    print rmse(lx, ly)
    print bias(lx,ly)
    print leastsq(lx,ly)
    plt.plot(lx, ly, 'o')
    #plt.plot(xx,x,'r')
    #plt.plot(xy,y,'b')
    #plt.show()
    plt.clf()
    year = range(1958,1958+42)
    print len(ya), len(yb)
    print r2(ya,yb)
    print rmse(ya, yb)
    print bias(ya,yb)
    print leastsq(ya,yb)
    plt.plot(year,ya,'r')
    plt.plot(year,yb,'b')
    plt.plot(gx, gy, 'g')
    plt.show()
    
    
def test4():
    X = 130
    Y = 116
    gewex = np.load(dp+'gewex.npy')
    g, gy = [], []
    gx = np.arange(1984,2005,1.0/12)
    for i in range(len(gx)):
        d = gewex[X,Y,i]
        g.append(d)
    for i in range(len(g)):
        gy.append(g[i])
    import data.site_data as sd
    r = sd.get_month_data()
    site = []
    x, y = [], []
    for i in r:
        if i[0] == '54511':
            site.append(i)
            #print i
            if i[4] not in range(1958,2011):
                continue
            if i[6] < 2000 and i[6] > -10:
                y.append(i[6])
            else:
                print 'xxxxxxx'
    data = np.load( dp + '1961to2006.npy.result.npy')
    xa = np.arange(1961, 2006,1.0/12)
    for i in range(540):
        wd = 90 - X - 0.5
        if X > 90:
            wd = 180 - X - 0.5
        wd = math.radians(wd)
        #d = data[X, Y, i] * np.cos(wd)
        d = data[X, Y, i]
        x.append(d)
    import matplotlib.pyplot as plt
    print 'xy:', len(x), len(y)
    ya ,yb = [], []
    sa ,sb = 0, 0
    for i in range(len(x)):
        ya.append(x[i])
    for i in range(len(y)):
        yb.append(y[i])
    plt.clf()
    year = np.arange(1958,1958+53, 1.0/12)
    
    mseason = [0,1,1,2,2,2,3,3,3,4,4,4,1]
    print len(ya), len(yb), len(gx), len(year)
    tx, ty = [], []
    for i in range(45*12):
        m = i%12 + 1
        s = mseason[m]
        if s != 1:
            pass
            #continue
        tx.append(yb[i+4*12])
        ty.append(ya[i])
        print yb[i+4*12], ya[i], yb[i+4*12]- ya[i]
    point(tx, ty, title='CMA v.s. CERES_RESULT Monthly')
    return
    print 'len', len(tx), len(ty)
    print 'r2', r2(tx,ty)
    print 'rmse', rmse(tx, ty)
    print 'bias', bias(tx, ty)
    print leastsq(tx, ty)
    
    print len(xa), len(ya)
    plt.plot(xa,ya,'r', label='ceres_result')
    print len(year), len(yb)
    plt.plot(year,yb,'b',label='cma')
    plt.plot(gx, gy, 'g', label='gewex')
    plt.legend()
    plt.show()
          
def test5():
    print 'ok'
    ceres = np.load(dp+'ceres.npy')
    gewex = np.load(dp+'gewex.npy')
    print gewex.shape#198401 -- 200412 24*12+6 294
    print ceres.shape#200003 -- 201409 14*12+7 175
    data = np.load( dp + '1961to2006.npy.result.npy')
    print ceres.shape, gewex.shape, data.shape
    import data.site_data as sd
    r = sd.get_month_data()
    x = []
    for i in r:
        if i[6] < 2000 and i[6] > -10:
            x.append(i)
    xa = range(1961, 2006)
    y1, y2, y3 = [], [], []
    x1, x2, x3 = [], [], []
    for i in range(0,540):
        year ,month = 1961 + (i+1)/12, (i)%12+1
        for j in x:
            if j[4] == year and j[5] == month:
                lat, lon = j[1], j[2]
                X, Y = int(lat + 90.5), int(lon)
                wd = 90 - X - 0.5
                wd = math.radians(wd)
                ytmp = data[X, Y, i]
                if ytmp < 0:continue
                x1.append(j[6])
                y1.append(ytmp)
    pointheat(x1, y1, title='CMA v.s. CERES_RESULT')
    for i in range(175):
        year ,month = 2000 + (i+3)/12, (i)%12+3 
        for j in x:
            if j[4] == year and j[5] == month:
                lat, lon = j[1], j[2]
                X, Y = int(lat + 90.5), int(lon)
                wd = 90 - X - 0.5
                wd = math.radians(wd)
                ytmp = ceres[X, Y, i]
                if ytmp < 0:continue
                x2.append(j[6])
                y2.append(ytmp)
    pointheat(x2, y2, title='CMA v.s. CERES')
        
    for i in range(252):
        year ,month = 1984 + (i+1)/12, (i)%12 +1
        for j in x:
            if j[4] == year and j[5] == month:
                lat, lon = j[1], j[2]
                X, Y = int(lat + 90.5), int(lon)
                wd = 90 - X - 0.5
                wd = math.radians(wd)
                ytmp = gewex[X, Y, i]
                if ytmp < 0:continue
                x3.append(j[6])
                y3.append(ytmp)
    pointheat(x3, y3, title='CMA v.s. GEWEX')
    return
        
def main():
    data = np.load( dp + '1961to2006.npy.result.npy')
    y = []
    ret = [[],[],[],[],[],[],[],[]]
    zz = np.zeros((180,360),dtype = int)
    for i in range(180):
        for j in range(360):
            zz[i ,j] = -1
    for i in range(540):
        print i
        s = [0,0,0,0,0,0,0,0]
        c = [0,0,0,0,0,0,0,0]
        for la in range(180):
            wd = 90 - la - 0.5
            if la > 90:
                wd = 180 - la + 0.5
            wd = math.radians(wd)
            #rad = k[i, j] * np.cos(wd)
            for lo in range(360):
                ss = data[la,lo,i]
                if la > 90:
                    lat = la - 90
                else :
                    lat = la - 90
                lon = lo - 180
                #print la ,lo ,lat, lon
                if zz[la, lo] == -1:
                    z = ll2zhou(lat, lon)
                    zz[la, lo] = int(z)
                else :
                    z = zz[la, lo]
                #z = lat % 8
                #print z,
                if ss > 2000 or ss < -10:
                    continue
                s[z] += ss 
                c[z] += 1
        for j in range(8):
            #print j
            ret[j].append(s[j]*1.0/c[j])
    x = range(1961, 2006)
    import matplotlib.pyplot as plt
    import copy
    plt.clf()
    dic = zhoudic()
    for i in range(8):
        y = []
        su = 0
        lines = []
        for j in range(540):
            su += ret[i][j]
            if (j+1) % 12 == 0:
                line = "%d,%f\n" % (1961+(j+1)/12, su/12.0)
                lines.append(line)
                y.append(su/12.0)
                su = 0
        list2Txt(lines, dic[i])
        print len(x), len(y)
        plt.clf()
        plt.plot(x, y, label = dic[i])
        plt.legend()
        #plt.show()
    plt.legend()
    plt.show()
        
    
    
    
if __name__ == "__main__":
    #ceres_check()
    #ceres_learn()
    #test3()
    #forcg()
    #test5()
    #test1()
    main()
    #forpaper()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
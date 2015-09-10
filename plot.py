#!/usr/bin/env python
# coding=utf-8
import matplotlib.pyplot as plt
import cal
import numpy as np

def image(f ,height, width, save=False):
    try:
        d = np.fromfile('gd.dem', dtype=np.int16)
        r = np.fromfile(f)
        for i in range(len(r)):
            if d[i] == 0:
                r[i] = 0
        r = r.reshape(height, width)
        plt.clf()
        plt.imshow(r)
        plt.colorbar()
        if save:
            plt.savefig(f+'.png')
        else:
            plt.show()
        return True
    except Exception, e:
        print e
        return False
    

def point_daily(heat=False, ex=False):
    f = 'out_daily'
    if ex :
        f = 'out_daily_ex'
    fr = open(f)
    x,y = [], []
    data = []
    for i in fr:
        i = i.strip().split(',')
        if len(i) < 6 or float(i[6]) > 2000 or float(i[6]) < 0.01 or float(i[7])>2000 or float(i[7]) < 0.01:
            continue
        x.append(float(i[6]))
        y.append(float(i[7]))
        data.append([int(float(i[6])), int(float(i[7]))])
    plt.plot([0,450],[0,450], 'black')
    if heat:
        from pyheatmap.heatmap import HeatMap
        hm = HeatMap(data)
        hm.heatmap(save_as="h.png")
        hmimg = plt.imread("h.png")
        plt.imshow(hmimg, aspect='auto',origin='lower')
    else:
        plt.plot(x, y, '.',alpha = 0.5)
    #print cal.leastsq(x,y)
    plt.text(280,120, "num = " + str(len(x)))
    plt.text(280,90, "bias = "+str(cal.bias(x,y)))
    plt.text(280,60, "rmse = "+str(cal.rmse(x,y)))
    plt.text(280,30, "r2 = "+str(cal.r2(x,y)))
    plt.xlabel("Observation")
    plt.ylabel("AVHRR-Production")
    plt.title("GLASS-AVHRR v.s. CMA Daily-Valid")

    plt.show()

def point_monthly(heat=False):
    f = 'out_monthly'
    fr = open(f)
    x,y = [], []
    data = []
    for i in fr:
        i = i.strip().split(',')
        if len(i) < 6 or float(i[6]) > 2000 or float(i[6]) < 0.01 or float(i[7])>2000 or float(i[7]) < 0.01:
            continue
        x.append(float(i[6]))
        y.append(float(i[7]))
        data.append([int(float(i[6])), int(float(i[7]))])
    print len(x)
    plt.plot([0,450],[0,450], 'black')
    if heat:
        from pyheatmap.heatmap import HeatMap
        hm = HeatMap(data)
        hm.heatmap(save_as="h.png")
        hmimg = plt.imread("h.png")
        plt.imshow(hmimg, aspect='auto',origin='lower')
    else:
        plt.plot(x, y, 'ob')
    print cal.leastsq(x,y)
    plt.text(280,120, "num = " + str(len(x)))
    plt.text(280,90, "bias = "+str(cal.bias(x,y)))
    plt.text(280,60, "rmse = "+str(cal.rmse(x,y)))
    plt.text(280,30, "r2 = "+str(cal.r2(x,y)))
    plt.xlabel("Observation")
    plt.ylabel("AVHRR-Production")
    plt.title("GLASS-AVHRR v.s. CMA Monthly-Valid")

    plt.show()

def point(x,y,t='o',title=''):
    plt.clf()
    plt.plot(x,y,t)
    print 'r2', cal.r2(x, y)
    print 'rmse', cal.rmse(x, y)
    print 'bias', cal.bias(x, y)
    print 'k, b', cal.leastsq(x, y)
    plt.text(270,120, "num = " + str(len(x)))
    plt.text(270,90, "bias = "+str(cal.bias(x,y)))
    plt.text(270,60, "rmse = "+str(cal.rmse(x,y)))
    plt.text(270,30, "r2 = "+str(cal.r2(x,y)))
    plt.plot([0,450],[0,450], 'black')
    plt.title(title)
    
    plt.show()
    

def pointsave(x,y,t='o',title='',path=''):
    plt.clf()
    plt.plot(x,y,t)
    print 'r2', cal.r2(x, y)
    print 'rmse', cal.rmse(x, y)
    print 'bias', cal.bias(x, y)
    print 'k, b', cal.leastsq(x, y)
    plt.title(title)
    plt.text(270,120, "num = " + str(len(x)))
    plt.text(270,90, "bias = "+str(cal.bias(x,y)))
    plt.text(270,60, "rmse = "+str(cal.rmse(x,y)))
    plt.text(270,30, "r2 = "+str(cal.r2(x,y)))
    plt.plot([0,450],[0,450], 'black')
    plt.savefig(path)
    
def pointheat(x,y,t='o',title=''):
    print 'r2', cal.r2(x, y)
    print 'rmse', cal.rmse(x, y)
    print 'bias', cal.bias(x, y)
    print 'k, b', cal.leastsq(x, y)
    data = []
    plt.clf()
    t = 8
    for i in range(len(x)):
        data.append([int(float(x[i])+t), int(float(y[i])+t)])
    from pyheatmap.heatmap import HeatMap
    hm = HeatMap(data)
    hm.heatmap(save_as="h.png")
    hmimg = plt.imread("h.png")
    plt.imshow(hmimg, aspect='auto',origin='lower')
    plt.title(title)
    plt.text(270,120, "num = " + str(len(x)))
    plt.text(270,90, "bias = "+str(cal.bias(x,y)))
    plt.text(270,60, "rmse = "+str(cal.rmse(x,y)))
    plt.text(270,30, "r2 = "+str(cal.r2(x,y)))
    plt.plot([0,450],[0,450], 'black')
    plt.xlim(0,450)
    plt.ylim(0,450)
    plt.xlabel('BSRN')
    plt.ylabel('Ceres_result')
    plt.show()
    
def pointheatsave(x,y,t='o',title='',path=''):
    print 'r2', cal.r2(x, y)
    print 'rmse', cal.rmse(x, y)
    print 'bias', cal.bias(x, y)
    print 'k, b', cal.leastsq(x, y)
    data = []
    plt.clf()
    for i in range(len(x)):
        data.append([int(float(x[i])+9), int(float(y[i])+9)])
    from pyheatmap.heatmap import HeatMap
    hm = HeatMap(data)
    hm.heatmap(save_as="h.png")
    hmimg = plt.imread("h.png")
    plt.imshow(hmimg, aspect='auto',origin='lower')
    plt.title(title)
    plt.text(270,120, "num = " + str(len(x)))
    plt.text(270,90, "bias = "+str(cal.bias(x,y)))
    plt.text(270,60, "rmse = "+str(cal.rmse(x,y)))
    plt.text(270,30, "r2 = "+str(cal.r2(x,y)))
    plt.savefig(path)
    
if __name__ == "__main__":

    point_daily(heat=True, ex=True)

    #point_monthly(heat=True)

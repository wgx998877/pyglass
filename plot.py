#!/usr/bin/env python
# coding=utf-8
import matplotlib.pyplot as plt
import pyglass.cal as cal
from pyheatmap.heatmap import HeatMap

f = 'out3'
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
plt.plot(x, y, 'ob')
plt.plot([0,450],[0,450], 'black')
#hm = HeatMap(data)
#hm.heatmap(save_as="h.png")
#hmimg = hmimg[::-1]

#hmimg = plt.imread("h.png")
print cal.leastsq(x,y)
plt.text(280,120, "num = " + str(len(x)))
plt.text(280,90, "bias = "+str(cal.bias(x,y)))
plt.text(280,60, "rmse = "+str(cal.rmse(x,y)))
plt.text(280,30, "r2 = "+str(cal.r2(x,y)))
plt.xlabel("Observation")
plt.ylabel("AVHRR-Production")
plt.title("GLASS-AVHRR v.s. CMA Daily-Valid")

#plt.imshow(hmimg, aspect='auto',origin='lower')
plt.show()

def point(x,y):
    pass

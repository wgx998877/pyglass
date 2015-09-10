#!/usr/bin/python
# Filename: mis.py
# Author: wgx
import os
import numpy as np
import datetime

from math import sin, cos, tan, pi, asin, acos, atan, atan2
#pi = 3.14159
tpi = 2*pi
degs = 180.0/pi
rads = pi/180.0

SunDia = 0.53
AirRefr = 34.0/ 60.0
L,g = 0, 0
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

def f0(lat, declin):
    df0 = rads * (0.5*SunDia + AirRefr)
    if lat < 0.0:
        df0 = -df0
    fo = tan(declin + df0) * tan(lat * rads)
    if fo > 0.999999999 or fo < -0.999999999:
        fo = 1.0
    fo = asin(fo) + pi/2.0
    return fo

def f1(lat, declin):
    df1 = rads * 6.0
    if lat < 0.0:
        df1 = -df1
    fo = tan(declin + df1) * tan(lat * rads)
    if fo > 0.999999999 or fo < -0.999999999 :
        fo = 1.0
    fo = asin(fo) + pi/2.0
    return fo
    
def FNrange (x):
    b = x / tpi;
    a = tpi * (b - int(b));
    if a < 0:
        a = tpi + a
    return a
    
def FNday (y, m, d, h):
    luku = int(- 7 * (y + (m + 9)/12)/4 + 275*m/9 + d)
    luku+= y*367
    return luku - 730531.5 + h/24.0

def FNsun (d):
    global L,g
    L = FNrange(280.461 * rads + .9856474 * rads * d)
    g = FNrange(357.528 * rads + .9856003 * rads * d)
    return FNrange(L + 1.915 * rads * sin(g) + .02 * rads * sin(2 * g))

def jday2monthDay(year, jday):
    days = get_days(year)
    month = 1
    for i in days:
        if jday - i <= 0 :
            break
        jday -= i
        month += 1
    return month , jday

def monthDay2jday(year, month, day):
    days = get_days(year)
    jday = day
    jday += sum(days[:month-1])
    return jday

def calSunTime(iYear,  iYearDay,  latitude,  longitude):
    tzone = int(longitude/15.0)
    latit = latitude
    longit = longitude
    month ,day = jday2monthDay(iYear, iYearDay)
    d=FNday(iYear,month,day,0)
    lamb = FNsun(d)
    obliq = 23.439 * rads - .0000004 * rads * d
    alpha = atan2(cos(obliq) * sin(lamb), cos(lamb))
    delta = asin(sin(obliq) * sin(lamb))
    LL = L - alpha
    if L < pi: 
        LL += tpi
    equation = 1440.0 * (1.0 - LL / tpi)
    ha = f0(latit,delta)
    hb = f1(latit,delta)
    twx = hb - ha
    twx = 12.0*twx/pi		
    daylen = degs*ha/7.5
    if daylen<0.0001: 
        daylen = 0.0
    riset = 12.0 - 12.0 * ha/pi + tzone - longit/15.0 + equation/60.0
    settm = 12.0 + 12.0 * ha/pi + tzone - longit/15.0 + equation/60.0

    if daylen > 24:
        fSunrise= 0
        fSunset = 24
	
    elif daylen<=0.0:
        fSunrise = 0
        fSunset = 0
    if riset > 24.0:
        riset-= 24.0
    if settm > 24.0: 
        settm-= 24.0

    fSunrise= riset
    fSunset = settm
		
    return fSunrise, fSunset
    
def Integrate_Daily_Par(data):
    num = len(data)
    fSunrise,fSunset = calSunTime(year, day, lat, lon)
    if num==0:
        daily_par=0
    elif num==1:
        daily_par=integrateSin0(fSunrise,fSunset,local_time_2[0],instan_par_2[0])
        daily_par=daily_par*3600/1000
    else :
        ff1=integrateSin1(fSunrise,fSunset,local_time_2[0],instan_par_2[0]);
        ff2=.0;
        for j in range(len(num-1)):
            ff2+=integrateXSin(fSunrise,fSunset,local_time_2[j],local_time_2[j+1], 
				               instan_par_2[j],instan_par_2[j+1])
        ff3=integrateSin2(fSunrise,fSunset,local_time_2[num-1],instan_par_2[num-1])
        daily_par = ff1+ff2+ff3
        daily_par=daily_par*3600/1000
    return daily_par
def Integrate_Daily_PAR_Direct(data):
    num = len(data)
    if num==0:
        daily_par_direct=0
    else :
        daily_par_direct=0
        for i in range(len(num - 1)):
            IntegratePAR += (this->instan_par_2[i+1]+this->instan_par_2[i])*(this->local_time_2[i+1] - this->local_time_2[i])*3600/2
        ff1 = this->instan_par_2[0]*(this->local_time_2[0] - fSunrise)*3600/2
        ff2 = this->instan_par_2[num-1]*(fSunset - this->local_time_2[num-1])*3600/2
        IntegratePAR +=ff1
        IntegratePAR +=ff2
        daily_par_direct = IntegratePAR/1000
    return daily_par_direct
    
def integrateSin0(trise, tset, t1, PAR1):
	temp = PAR1 * (tset-trise) / pi / sin( (t1-trise) * pi / (tset-trise) )
    result = temp *2
	if result < 0:
		return 0.0
	else:
        return result


def integrateSin1(trise, tset, t1, PAR1):
	temp = PAR1 * (tset-trise) / pi / sin( (t1-trise) * pi / (tset-trise) )
	result = temp * ( 1 - cos( (t1-trise) * pi / (tset-trise) ) )
	if result < 0.:
		return 0.0
	else :
        return result

def integrateSin2(trise, tset, t2, PAR2):
	double temp, result
	temp = PAR2 * (tset-trise) / pi / sin( (t2-trise) * pi / (tset-trise) )
	result = temp * ( cos( (t2-trise) * pi / (tset-trise)) + 1 )
	if result < 0.:
		return 0.0
	else :
        return result

def integrateXSin(tr, ts, t1, t2, R1, R2):
	term1 = cos(pi*(t1-tr)/(tr-ts))
	term2 = cos(pi*(t2-tr)/(tr-ts))
	term3 = 1./ sin(pi*(t2-tr)/(tr-ts))
	term4 = 1./sin(pi*(t1-tr)/(-tr+ts))
	term5 = sin(pi*(t1-tr)/(tr-ts)
	term6 = sin(pi*(t2-tr)/(tr-ts))
	
	temp1 = R2 * t1 * (tr-ts) * (term1-term2) * term3/pi/(t1-t2)
	temp2 = R1 * t2 * (tr-ts) * (term1 - term2)* term4/pi/(t1-t2)
	temp3 = (R1*(tr-ts)*term4*(-pi*t1*term1+pi*t2*term2+(tr-ts)*(term5-term6)))/(pi*pi*(t1-t2))
	temp4 = (R2*(tr-ts)*(-term3)*(-pi*t1*term1+pi*t2*term2+(tr-ts)*(term5-term6)))/(pi*pi*(t1-t2))
	
	result = temp1+temp2+temp3-temp4
	if result < 0.:
		return 0.0
	else :
        return result;


if __name__ == '__main__':
    print calSunTime(2010,130,10,10)
    print calSunTime(2010,230,50,50)
    
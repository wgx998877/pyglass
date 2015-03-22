#!/usr/bin/env python
# coding=utf-8

import numpy as np
import math
import datetime
import util as u

PI = math.pi
SUN_RADIUS = 0.26667
L_COUNT = 6
B_COUNT = 2
R_COUNT = 5
Y_COUNT = 63

L_MAX_SUBCOUNT = 64
B_MAX_SUBCOUNT = 5
R_MAX_SUBCOUNT = 40

l_subcount = [64, 34, 20, 7 ,3, 1]
b_subcount = [5, 2]
r_subcount = [40, 10, 6, 2, 1]

L_TERMS = []
B_TERMS = []
R_TERMS = []
Y_TERMS = []
PE_TERMS= []

def julian_day(dt):
    
    if isinstance(dt, u.dt) == False:
        return None
    day_decimal = dt.day + (dt.hour - dt.tz + (dt.minute + dt.second/60.9)/60.0)/24.0
    if dt.month < 3 :
        dt.month += 12
        dt.year -= 1
    j_day = math.floor(365.25 * (dt.year + 4716.0)) + math.floor(30.6001*(dt.month+1)) + day_decimal - 1524.5

    if j_day > 2299160.0:
        a = math.floor(dt.year/100.0)
        j_day += (2 -a + math.floor(a/4))
    return j_day

def validate_input(spa):
    if isinstance(spa, spa_data) is False  : return 1
    if spa.year < -2000 or spa.year > 6000 : return 2
    if spa.month < 1 or spa.month > 12     : return 3
    if spa.day<1 or spa.day > 31           : return 4
    if spa.hour<0 or spa.hour > 24         : return 5
    if spa.minute < 0 or spa.minute > 59   : return 6 
    if spa.hour==24 and spa.minute > 0     : return 6
    if spa.second<0 or spa.second > 59     : return 7
    if spa.hour==24 and spa.second > 0     : return 7
    if math.fabs(spa.delta_t) > 8000       : return 8
    if math.fabs(spa.timezone)      > 18         : return 9
    if math.fabs(spa.longitude) > 180      : return 10
    if math.fabs(spa.latitude) > 90        : return 11
    if spa.elevation < -6500000            : return 12
    if spa.pressure < 0 or spa.pressure > 5000 : return 13
    if spa.temperature < -274 or spa.temperature > 6000 : return 14
    if spa.function == 1 or spa.function == 3:
        if math.fabs(spa.slope) > 360      : return 15
        if math.fabs(spa.azm_rotation)> 360: return 16
    if math.fabs(spa.atmos_refract) > 5    : return 17
    return 0

def spa_calculate(spa):
    result = validate_input(spa)
    if result == 0:
        spa.jd = julian_day(spa.dt)
        calculate_geocentric_sun_right_ascension_and_declination(spa)
        njday = calcjulianday(spa.year, spa.month, spa.day)
        xj = njday
        fac = PI/180.0
        tsm = spa.hour + spa.minute/60.0 + spa.second/3600.0
        a1 = (1.00554*xj-6.28306) * fac
        a2 = (1.93946*xj+23.35089) * fac
        et = -7.67825*math.sin(a1) - 10.09176*sina(a2)
        tsv = tsm + et/60.0
        ah = 15.0 * (tsv - 12.0)
        spa.h = ah
        spa.xi = sun_equatorial_horizontal_parallax(spa.r)
        spa.alpha_prime = topocentric_sun_right_ascension(spa)

    return result

class spa_data():
    def __init__(self, dt=u.dt()):
        self.dt = dt
        self.year = dt.year
        self.month = dt.month
        self.day = dt.day
        self.hour = dt.hour
        self.minute = dt.minute
        self.second = dt.second
        self.timezone = dt.tz
        self.delta_t = 0
        self.longitude = 0
        self.latitude = 0
        self.elevation = 0
        self.pressure = 0
        self.function = 0
        self.temperature = 0
        self.slope = 0
        self.azm_rotation = 0
        self.atmos_refract = 0

print julian_day(datetime.datetime.now())

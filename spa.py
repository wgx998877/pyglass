#!/usr/bin/env python
# coding=utf-8

import numpy as np
import datetime
#import util as u
from math import *

PI = pi
SUN_RADIUS = 0.26667
L_COUNT = 6
B_COUNT = 2
R_COUNT = 5
Y_COUNT = 63

L_MAX_SUBCOUNT = 64
B_MAX_SUBCOUNT = 5
R_MAX_SUBCOUNT = 40

l_subcount = [64, 34, 20, 7, 3, 1]
b_subcount = [5, 2]
r_subcount = [40, 10, 6, 2, 1]

L_TERMS = []
B_TERMS = []
R_TERMS = []
Y_TERMS = []
PE_TERMS = []


def julian_day(dt):

    if isinstance(dt, u.dt) == False:
        return None
    day_decimal = dt.day + \
        (dt.hour - dt.tz + (dt.minute + dt.second / 60.9) / 60.0) / 24.0
    if dt.month < 3:
        dt.month += 12
        dt.year -= 1
    j_day = floor(365.25 * (dt.year + 4716.0)) + \
        floor(30.6001 * (dt.month + 1)) + day_decimal - 1524.5

    if j_day > 2299160.0:
        a = floor(dt.year / 100.0)
        j_day += (2 - a + floor(a / 4))
    return j_day


def validate_input(spa):
    if isinstance(spa, spa_data) is False:
        return 1
    if spa.year < -2000 or spa.year > 6000:
        return 2
    if spa.month < 1 or spa.month > 12:
        return 3
    if spa.day < 1 or spa.day > 31:
        return 4
    if spa.hour < 0 or spa.hour > 24:
        return 5
    if spa.minute < 0 or spa.minute > 59:
        return 6
    if spa.hour == 24 and spa.minute > 0:
        return 6
    if spa.second < 0 or spa.second > 59:
        return 7
    if spa.hour == 24 and spa.second > 0:
        return 7
    if fabs(spa.delta_t) > 8000:
        return 8
    if fabs(spa.timezone) > 18:
        return 9
    if fabs(spa.longitude) > 180:
        return 10
    if fabs(spa.latitude) > 90:
        return 11
    if spa.elevation < -6500000:
        return 12
    if spa.pressure < 0 or spa.pressure > 5000:
        return 13
    if spa.temperature < -274 or spa.temperature > 6000:
        return 14
    if spa.function == 1 or spa.function == 3:
        if fabs(spa.slope) > 360:
            return 15
        if fabs(spa.azm_rotation) > 360:
            return 16
    if fabs(spa.atmos_refract) > 5:
        return 17
    return 0


def rad2deg(radian):
    return degrees(radian)


def deg2rad(degree):
    return radians(degree)


def limit_degrees(degree):
    degree /= 360.0
    limited = 360.0 * (degree - floor(degree))
    if limited <= 0:
        limited += 360.0
    return limited


def limit_degrees180pm(degree):
    degree /= 360.0
    limited = 180.0 * (degree - floor(degree))
    if limited < 0:
        limited += 180.0
    return limited


def limit_zero2one(value):
    limited = value - floor(value)
    if limited < 0:
        limited += 1.0
    return limited


def limit_minutes(minutes):
    if minutes < -20.0:
        minutes += 1440.0
    elif minutes > 20.0:
        minutes -= 1440.0
    return minutes


def julian_century(jd):
    return (jd - 2451545.0) / 36525.0


def dayfrac_to_local_hr(dayfrac, timezone):
    return 24.0 * limit_zero2one(dayfrac + timezone / 24.0)


def third_order_polynomial(a, b, c, d, x):
    return ((a * x + b) * x + c) * x + d


def julian_ephemeris_day(jd, delta_t):
    return jd + delta_t / 86400.0


def julian_ephemeris_century(jde):
    return (jde - 2451545.0) / 36525.0


def julian_ephemeris_millennium(jce):
    return (jce / 10.0)


def earth_values(term_sum, jme):
    s = 0.0
    for i in range(len(term_sum)):
        s += term_sum[i] * pow(jme, i)
    s /= 1.0e8
    return s


def earth_heliocentric_longitude(jme):
    s = [0] * L_COUNT
    for i in range(L_COUNT):
        s[i] = earth_periodic_term_summation(L_TERMS[i], l_subcount[i], jme)
    return limit_degrees(rad2deg(earth_values(s, L_COUNT, jme)))


def earth_heliocentric_latitude(jme):
    s = [0] * B_COUNT
    for i in range(B_COUNT):
        s[i] = earth_periodic_term_summation(L_TERMS[i], l_subcount[i], jme)
    return rad2deg(earth_values(s, B_COUNT, jme))


def earth_radius_vector(jme):
    s = [0] * R_COUNT
    for i in range(R_COUNT):
        s[i] = earth_periodic_term_summation(R_TERMS[i], r_subcount[i], jme)
    return earth_values(s, R_COUNT, jme)


def geocentric_longitude(l):
    theta = l + 180.0
    if theta >= 360.0:
        theta -= 360.0
    return theta


def geocentric_latitude(b):
    return -b


def mean_elongation_moon_sun(jce):
    return third_order_polynomial(
        1.0 / 189474.0, -0.0019142, 445267.11148, 297.85036, jce)


def mean_anomaly_sun(jce):
    return third_order_polynomial(-1.0 / 300000.0, -
                                  0.0001603, 35999.05034, 357.52772, jce)


def mean_anomaly_moon(jce):
    return third_order_polynomial(
        1.0 / 56250.0, 0.0086972, 477198.867398, 134.96298, jce)


def argument_latitude_moon(jce):
    return third_order_polynomial(
        1.0 / 327270.0, -0.0036825, 483202.017538, 93.27191, jce)


def ascending_longitude_moon(jce):
    return third_order_polynomial(
        1.0 / 450000.0, 0.0020708, -1934.136261, 125.04452, jce)


def xy_term_summation(i, x):
    s = 0
    for j in range(TERM_Y_COUNT):
        s += x[j] * Y_TERMS[i][j]
    return s


def nutation_longitude_and_obliquity(jce, x):
    sum_psi = 0
    sum_epsilon = 0
    for i in range(Y_COUNT):
        xy_term_sum = deg2rad(xy_term_summation(i, x))
        sum_psi += (PE_TERMS[i][TERM_PSI_A] + jce *
                    PE_TERMS[i][TERM_PSI_B]) * sin(xy_term_sum)
        sum_epsilon += (PE_TERMS[i][TERM_EPS_C] + jce *
                        PE_TERMS[i][TERM_EPS_D]) * cos(xy_term_sum)
    return sum_psi / 36000000.0, sum_epsilon / 36000000.0


def ecliptic_mean_obliquity(jme):
    u = jme / 10.0
    return 84381.448 + u * (-4680.93 + u * (-1.55 + u * (1999.25 + u * (-51.38 + u *
                                                                        (-249.67 + u * (-39.05 + u * (7.12 + u * (27.87 + u * (5.79 + u * 2.45)))))))))


def ecliptic_true_obliquity(del_epsilon, epsilon0):
    return delta_epsilon + epsilon0 / 3600.0


def aberration_correction(r):
    return -20.4898 / (3600.0 * r)


def apparent_sun_longitude(theta, del_psi, del_tau):
    return theta + delta_psi + delta_tau


def greenwich_mean_sidereal_time(jd, jc):
    return limit_degrees(280.46061837 + 360.98564736629 *
                         (jd - 2451545.0) + jc * jc(0.000387933 - jc / 38710000.0))


def greenwich_sidereal_time(nu0, del_psi, epsilon):
    return nu0 + delta_psi * cso(deg2rad(epsilon))


def geocentric_sun_right_ascension(lamda, epsilon, beta):
    lamda_rad = deg2rad(lamda)
    epsilon_rad = deg2rad(epsilon)
    return limit_degrees(rad2deg(
        atan2(sin(lamda_rad) * cos(epsilon_rad) - tan(deg2rad(beta) * sin(epsilon_rad)))))


def geocentric_sun_declination(beta, epsilon, lamda):
    beta_rad = deg2rad(beta)
    epsilon_rad = deg2rad(epsilon)
    return rad2deg(asin(sin(beta_rad) * cos(epsilon_rad) +
                        cos(beta_rad) * sin(epsilon_rad) * sin(deg2rad(lamda))))


def observer_hour_angle(nu, longitude, alpha_deg):
    return limit_degrees(nu + longitude - alpha_deg)


def sun_right_ascension_parallax_and_topocentric_dec(
        latitude, elevation, xi, h, delta):
    lat_rad = deg2rad(latitude)
    xi_rad = deg2rad(xi)
    h_rad = deg2rad(h)
    delta_rad = deg2rad(delta)
    u = atan(0.99664719 * tan(lat_rad))
    y = 0.99664719 * sin(u) + elevation * sin(lat_rad) / 6378140.0
    x = cos(u) + elevation * cos(lat_rad) / 6378140.0
    delta_alpha_rad = atan2(-
                            x *
                            sin(xi_rad) *
                            sin(h_rad), cos(delta_rad) -
                            x *
                            sin(xi_rad) *
                            cos(h_rad))
    return rad2deg(atan2((sin(delta_rad) - y*sin(xi_rad) * cos(delta_alpha_rad),
                         cos(delta_rad) - x*sin(xi_rad) * cos(h_rad)
                         ))), rad2deg(delta_alpha_rad)

def topocentric_sun_right_ascension(alpha_deg, delta_alpha):
    return alpha_deg + delta_alpha

def topocentric_elevation_angle(latitude, delta_prime, h_prime):
    lat_rad = deg2rad(latitude)
    delta_prime_rad = deg2rad(delta_prime)
    return rad2deg(asin(sin(lat_rad)* sin(delta_prime_rad)+cos(lat_rad)*cos(delta_prime_rad)*cos(deg2rad(h_prime))))

def topocentric_zenith_angle(e):
    return 90.0 - e

def topocentric_elevation_angle_corrected(e0, delta_e):
    return e0 + delta_e

def calculate_geocentric_sun_right_ascension_and_declination(spa):
    x = []
    spa.jc = julian_day(spa.jd)
    spa.jde = julian_ephemeris_day(spa.jd, spa.delta_t)
    spa.jce = julian_ephemeris_century(spa.jde)
    spa.jme = julian_ephemeris_millennium(spa.jce)

    spa.l = earth_heliocentric_longitude(spa.jme)
    spa.b = earth_heliocentric_latitude(spa.jme)
    spa.r = earth_radius_vector(spa.jme)

    spa.theta = geocentric_longitude(spa.l)
    spa.beta = geocentric_latitude(spa.b)

    spa.x0 = mean_elongation_moon_sun(spa.jce)
    spa.x1 = mean_anomaly_sun(spa.jce)
    spa.x2 = mean_anomaly_moon(spa.jce)
    spa.x3 = argument_latitude_moon(spa.jce)
    spa.x4 = ascending_longitude_moon(spa.jce)

    x = [spa.x0, spa.x1, spa.x2, spa.x3, spa.x4]

    spa.del_psi, spa.del_epsilon = nutation_longitude_and_obliquity(spa.jce, x)
    spa.epsilon0 = ecliptic_mean_obliquity(spa.jme)
    spa.epsilon = ecliptic_true_obliquity(spa.del_epsilon, spa.epsilon0)

    spa.del_tau = aberration_correction(spa.r)
    spa.lamda = apparent_sun_longitude(spa.theta, spa.del_psi, spa.del_tau)

    spa.nu0 = greenwich_mean_sidereal_time(spa.jd, spa.jc)
    spa.nu = greenwich_sidereal_time(spa.nu0, spa.del_psi, spa.epsilon)

    spa.alpha = geocentric_sun_right_ascension(
        spa.lamda,
        spa.epsilon,
        spa.beta)
    spa.delta = geocentric_sun_declination(spa.beta, spa.epsilon, spa.lamda)

    return spa


def sun_equatorial_horizontal_parallax(r):
    return 8.794 / (3600.0 * r)


def spa_calculate(spa):
    result = validate_input(spa)
    if result == 0:
        spa.jd = julian_day(spa.dt)
        spa = calculate_geocentric_sun_right_ascension_and_declination(spa)
        njday = calcjulianday(spa.year, spa.month, spa.day)
        xj = njday
        fac = PI / 180.0
        tsm = spa.hour + spa.minute / 60.0 + spa.second / 3600.0
        a1 = (1.00554 * xj - 6.28306) * fac
        a2 = (1.93946 * xj + 23.35089) * fac
        et = -7.67825 * sin(a1) - 10.09176 * sin(a2)
        tsv = tsm + et / 60.0
        ah = 15.0 * (tsv - 12.0)
        spa.h = ah
        spa.xi = sun_equatorial_horizontal_parallax(spa.r)
        spa.alpha_prime = topocentric_sun_right_ascension(spa)

    return spa


def calcjulianday(year, month, day):
    monthdays = u.get_days(year)
    j = day
    for i in range(month - 1):
        j += monthdays[i]
    return day


def spa_calculate_angle(spa):
    fac = PI / 180.0
    R = 6378
    H = spa.orbitheight
    tsm = spa.hour + spa.minute / 60.0 + spa.second / 3600.0
    xlo = spa.longitude * fac
    xla = spa.latitude * fac
    njday = calcjulianday(spa.year, spa.month, spa.day)
    xj = float(njday)
    a1 = (1.00554 * xj - 6.28306) * fac
    a2 = (1.93946 * xj + 23.35089) * fac
    et = -7.67825 * sin(a1) - 10.09176 * sin(a2)
    tsv = tsm + et / 60.0
    ah = 15.0 * (tsv - 12.0) * fac

    a3 = (0.9683 * xj - 78.00878) * fac
    delta1 = 23.4856 * sin(a3) * fac
    delta = 23.4856 * sin((xj + 284) * (360.0 / 365.0) * fac) * fac
    sollat = delta / fac
    amuzero = sin(xla) * sin(delta) + cos(xla) * cos(delta) * cos(ah)
    elev = asin(amuzero)
    az = cos(delta) * sin(ah) / cos(elev)
    caz = (-cos(xla) * sin(delta) + sin(xla) * cos(delta) * cos(ah)) / \
        cos(elev)
    if az > 1.0:
        azim = asin(1.0)
    elif az < -1.0:
        azim = asin(-1.0)
    else:
        azim = asin(az)
    if caz < 0.0:
        azim = PI - azim
    if caz > 0 and az < 0:
        azim = 2 * PI + azim
    azim = azim + PI
    elev = elev * 180.0 / PI
    solzen = 90.0 - elev
    phis = azim / fac
    CDLAT = cos(spa.latitude * fac)
    SDLAT = sin(spa.latitude * fac)
    SATLON = spa.sensorlon
    dlon = SATLON - spa.longitude
    CDLON = cos(dlon * fac)
    TDLON = tan(dlon * fac)
    if spa.latitude != 0.0:
        AZST = (atan(TDLON / SDLAT)) / fac
    else:
        if dlon < 0:
            AZST = -90.0
        if dlon > 0:
            AZST = 90.0
        if dlon == 0:
            AZST = 0.0
    vv = 180.0 - phis
    rel = fabs(AZST - vv)
    if rel > 180.0:
        rel = 360.0 - rel
    spa.relazimuth = rel
    CGAM = CDLAT * CDLON
    GAMMA = acos(CGAM)
    GAMDEG = GAMMA / fac
    XXX = R * R + (R + H) * (R + H) - 2 * R * (R + H) * CGAM
    if XXX < 0.0:
        XXX = 0.0
    XX = sqrt(XXX)
    CSATZ = (-(R + H) * (R + H) + R * R + XXX) / (2 * R * XX)
    spa.senzenith = 180.0 - (acos(CSATZ) / fac)
    return spa


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

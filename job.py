#!/usr/bin/env python
# coding=utf-8

import util as u
import source
import production
from spa import *


class job():

    def __init__(self):
        pass

    def readConfig(self, filepath):
        if u.file_exist(filepath) == False:
            return False
        f = open(filepath)
        config = {}
        for i in f:
            i = i.strip().split('=')
            config[i[0].strip()] = i[1].strip()
        return config


class RadEstimation(job):

    def __init__(self, configfp):
        self.configfp = configfp
        self.config = self.readConfig(configfp)

    def checkConfig(self):
        para = []
        for i in para:
            if i not in self.config:
                return False
        return True

    def init(self):
        self.DNlist = u.readTxt(self.config['DNlist'])
        self.row = self.config['row']
        self.col = self.config['col']
        self.name = self.config['name']
        self.lat = source.LatLon(self.config['lat'], self.row, self.col)
        self.lon = source.LatLon(self.config['lon'], self.row, self.col)
        self.dem = source.DEM(self.config['dem'], self.row, self.col)

        self.ref = source.Reflectance(self.config['ref'], self.row, self.col)
        self.inso = production.INSO(
            self.config['INSO_TOA'],
            self.config['INSO_SUR'],
            self.config['sensor'])
        self.inso.readLUT()
        self.par = production.PAR(
            self.config['PAR_TOA'],
            self.config['PAR_SUR'],
            self.config['sensor'])
        self.wv = source.Watervapor(self.config['water'])
        self.spa = spa_data()
        self.jobType = 'production'
        self.sensor_lon = self.config['sensor_lon']

    def run(self):
        SF = source.SatelliteFactory()
        for i in self.DNlist:
            satellite = SF.create(
                name=self.name,
                fp=i,
                row=self.row,
                col=self.col)
            satellite.readDN()
            self.ref.setRef(
                satellite.filetime.year,
                satellite.filetime.month,
                satellite.jdays,
                satellite.fgmt,
                0.05)
            self.ref.readRefQCMatrixFromHDFFile()
            self.wv.setWV()
            self.wv.readMERRAWV()
            inso_path, par_path = self.generateOutputFilePaht(
                i, satellite.filetime.year, satellite.jdays)
            u.createhdf(inso_path)
            u.createhdf(par_path)

            for j in range(self.row):
                print "line %d: %d of %d files" % (j, i, len(self.DNlist))
                QC_MODIS = np.zeros((self.col, 40))
                QC = np.zeros((self.col, 40))
                QC_Com = np.zeros((self.col, 17))

                satellite.calibrate()
                self.spa = satellite.calLineOfAngles(
                    self.lat.DN[j],
                    self.lon.DN[j],
                    self.dem.DN[j],
                    self.spa,
                    self.sensor_lon)
                # self.ref.getOneLineOfRefQC()
                # self.wv.getOneLineOfWV()
                insoDi = np.zeros(self.col)
                insoDif = np.zeros(self.col)
                insoGlobal = np.zeros(self.col)
                parDi = np.zeros(self.col)
                parDif = np.zeros(self.col)
                parGlobal = np.zeros(self.col)
                for k in range(self.col):
                    if self.dem[j][k] < 0:
                        self.dem[j][k] = 1
                    if self.dem[j][k] < 0 or self.lat.DN[j][k] > 90 or self.lon.DN[j][k] > 180 or self.lat.DN[j][k] < -90 or self.lon[j][k] < -180 or satellite.DN[j][k] < 0 or self.ref.DN[j][k] <= 0:
                        continue
                    self.inso.Interpolation(satellite.angle[k].solzen, satellite.angle[k].senzen, satellite.angle[k].relazi, wv.WV[k])
                    self.inso.SearchLUT(satellite.DN[j][k], ref.DN[j][k])
                    self.par.Interpolation(satellite.angle[k].solzen, satellite.angle[k].senzen, satellite.angle[k].relazi, wv.WV[k])
                    insoDi[k], insoDif[k] = self.InterpolationRAD(self.inso.Di, self.inso.Dif, self.dem.DN[j][k])
                    insoGlobal[k] = insoDi[k] + insoDif[k]
                    parDi[k], parDif[k] = self.InterpolationRAD(self.par.Di, self.par.Dif, self.dem.DN[j][k])
                    parGlobal[k] = parDi[k] + parDif[k]
                u.appendhdf(inso_path, 'DSR', j, insoGlobal)
                u.appendhdf(inso_path, 'Direct', j, insoDi)
                u.appendhdf(inso_path, 'Diffuse', j, insoDif)
                u.appendhdf(par_path, 'PAR', j, parGlobal)
                u.appendhdf(par_path, 'Direct', j, parDil)
                u.appendhdf(par_path, 'Diffuse', j, parDif)

        def InterpolationRAD(self, pDi, pDif, elevation):
            ele_cur = elevation * production.ELE_SCALE
            di, dif = 0.0, 0.0
            for i in range(production.Nele - 1):
                if ele_cur >= self.par.ele[i] and ele_cur <= self.par.ele[i+1]:
                    dw = ele_cur - self.par.ele[i]
                    di = pDi[i] + dw* (pDi[i+1] - pDi[i])
                    dif = pDif[i] + dw * (pDif[i+1] - pDif[i]) 
                    break
            if ele_cur > self.par.ele[production.Nele - 1]:
                di = pDi[production.Nele - 1]
                dif = pDi[production.Nele - 1]
            return di, dif

        def generateOutputFilePath(self, filename, year, jday):
            path = filename[:filename.rfind('\\')]
            insoPath = "%sGLASSDSR.V02.A%d%03d%s.hdf" % (path, year, jday)
            parPath = "%sGLASSDSR.V02.A%d%03d%s.hdf" % (path, year ,jday)
            return insoPath, parPath
            #todo

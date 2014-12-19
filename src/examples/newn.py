#! /usr/bin/env python3
import numpy as np
import wavenet.net as net
import pylab as plb
import wavenet.wavelets as wt
import sys
import csv
import datetime as dt
from scipy import signal

class Test():
    def func1(self, x):
        if x < -2:
            return -2.186*x-12.846
        elif -2 <= x < 0:
            return 4.246*x
        #elif -1 <= x < 0:
        #    return np.sin(10*x)
        elif 0 <= x:
            return 10*np.exp(-.05*x - 0.5)*np.sin((0.3*x + 0.7)*x)

    def func0(x):
        return 10*np.exp(-.05*x - 0.5)*np.sin((0.3*x + 0.7)*x)+np.random.random()

    def func2(self, x):
        return np.cos(x*3+1)*np.exp(-0.5*x**2+1)

    def func3(x):
        return 0.0001*x**5

    def show(self, t, target, wavenet, param):
        plb.rc('font', family='serif')
        plb.rc('font', size=8)
        plb.subplot(211)
        plb.title('Вейвсеть из 30 вейвлетов Морле')
        plb.plot(t, target, label='Модельный сигнал')
        plb.plot(t, wavenet.sim(t), label='Аппроксимация')
        plb.legend()
        plb.subplot(212)
        plb.title('Суммарная квадратичная ошибка')
        plb.plot(param['e'])
        plb.xlabel('Эпохи')

        plb.figure()
        plb.subplot(131)
        plb.title('Масштабы, a')
        plb.plot(param['a'])
        plb.subplot(132)
        plb.title('Сдвиги, b')
        plb.plot(param['b'])
        plb.subplot(133)
        plb.title('Веса, w')
        plb.plot(param['w'])
        plb.figure()
        plb.subplot(131)
        plb.title('Параметры, p')
        plb.plot(param['p'])
        plb.subplot(132)
        plb.title('Константа, c')
        #import pdb; pdb.set_trace()
        plb.plot(param['c'])
        plb.subplot(133)
        plb.title('Скорость, r')
        plb.plot(param['j'])
        
        plb.show()

    def load(self):
        with open(self.fileName, 'rt') as csvdata:
            date = []
            value = []
            self.header = []
            for row in csv.reader(csvdata, delimiter=' '):
                #print (row)
                if ('#' in row[0]):
                    self.header.append(row)
                else:
                    date.append(' '.join([row[0], row[1]]))
                    value.append(row[2])
        #import pdb; pdb.set_trace()
        signal = np.array((date, value), dtype=np.dtype('a25'))
        signal = signal[:, np.logical_not(np.isnan(signal[1, :].astype(np.float)))]
        #self.notifyProgress.emit(40)
        self.value = signal[1, :].astype(np.float)
        # self.value=np.nan_to_num(self.value)
        #self.notifyProgress.emit(60)
        self.time = signal[0, :].astype(np.datetime64).astype(dt.datetime)
        #self.notifyProgress.emit(80)

    def calc(self):
        print('loaded...')
        #target = np.vectorize(self.func2)(t)
        #t = signal.detrend(plb.date2num(self.time)[0:], type='constant')
        t = plb.date2num(self.time)
        #t-=t.min()
        #
        #t/=t.max()
        target = signal.detrend(self.value[0:], type='constant')
        target = self.value
        #target = target / np.max(target)
        #t -= np.min(t)
        t*=12
#        import pdb; pdb.set_trace()
        wn = net.Wavenet(wt.Morlet, 30, t, target)
        #param = wn.train(t, target, 1, epoch=1000, extend=True)
        wn.smart_train(t, target)
        plb.plot(t, wn.sim(t), label='Аппроксимация')
        plb.show()
        #yself.show(t, target, wn, param)
        #err1 = wn1.train(t, target, 2., epoch=100, extend=False)
        #plb.plot(err1['p'], label='Обычный метод')y
        #plb.plot(err['p'])
        #plb.legend()
        #plb.show()
        #import pdb; pdb.set_trace()
        #plb.rc('font', serif='Times New Roman')
        #plb.rc('text', usetex = True)
        sys.exit()
    def __init__(self):
        self.fileName = "./spidr_1417683781512_0.txt"
        self.load()
t = Test()
t.calc()
















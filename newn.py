#! /usr/bin/env python3
import numpy as np
from wavenet import Wavenet
import pylab as plb
from wavelets import Morlet
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

    def test(self):
        t = np.linspace(-10, 10, num=100)
        target = np.vectorize(self.func1)(t)
        wn = Wavenet(Morlet, 30, t, target)
        wn.smart_train(t, target, maxiter=500)
        plb.plot(t, target, label='Сигнал')
        plb.plot(t, wn.sim(t), label='Аппроксимация')
        plb.show()

    def show():
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

          
        #param = wn.train(t, target, 1, epoch=1000, extend=True)
        
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

t = Test()
t.test()
















import numpy as np
import scipy.optimize as opt
from wavelets import Morlet
import pylab as plb


class Wavenet():
    def __init__(self, wavelet, ncount, x, y):
        """Create wavenet.

        Args:
        cn: int: Number  of neurons

        Returns:
        net: Wavenet

        Attributes:
        param: dict: Parameters of net
        ncount: int: Count of neuron
        """
        self.wt = wavelet
        self.param = {}
        self.param['p'] = np.ones(ncount)*5
        self.deltax = np.max(x)-np.min(x)
        countx = (np.max(x)-np.min(x))         
        deltay = np.max(y)-np.min(y)
        self.param['b'] = np.ones(ncount)*(np.max(x)+np.min(x))*0.5
        #self.param['b'] = np.random.random(ncount)
        self.param['c'] = np.zeros(1) + np.mean(y)
        self.param['d'] = np.zeros(1)
        #self.param['c']= np.random.random(1)+y[0]
        self.ncount = ncount
        self.tau = np.vectorize(self._tau, cache=True)
        self.h = np.vectorize(self.wt.wavelet, cache=True)
        self.step = np.vectorize(self._step, cache=True)
        self.xcount = x.shape[-1]
        self.param['a'] = np.ones(ncount)*(np.max(x)-np.min(x))*0.2
        self.param['w'] = np.zeros(ncount)

    def __call__(self, x, *args):
        self.param = self.unpack(x)
        return self.energy(self.input, self.target)

    def derivative(self, x, *args):
        self.param = self.unpack(x)
        gr = self.antigradient(self.input, self.target)
        #import pdb;pdb.set_trace()
        return self.pack(gr)
    
    def sim(self, t):
        """Simulate network

        Args:
        t: array: Time series

        Returns:
        array: Net output
        """
        #t = (t-self.d)*self.k
        return self.step(t)+self.param['c']

    def _step(self, t):
        """
        Compute result for one time moment

        Args:
        double: t: Time moment

        Return:
        double: x Net resulst
        """
        tau =(t-self.param['b'])/self.param['a']
        return np.sum(self.wt.wavelet(self.param['p'], tau)*self.param['w'])+t*self.param['d']

    def train(self, input, target, maxiter):
        x0 = self.pack(self.param)
        self.input=input
        self.target=target
        res1 = opt.fmin_bfgs(self, x0, fprime=self.derivative, maxiter=maxiter)
        return res1

    def pack(self, param):
        x = np.array([])
        for k in self.param.keys():
            x = np.append(x, param[k])
        return np.array(x)

    def unpack(self, aparam):
        inx = 0
        p = {}
        for k in self.param.keys():
            l = self.param[k].shape[-1]
            p[k] = aparam[inx:inx+l]
            inx += l
        return p
        
    def _train(self, input, target, error, extend=False, epoch=None):
        """Train network
        Args:
        input: array: Input signal
        target: array: Target output
        error: double: Sumsqrt error
        """
        errlist = {}
        err = []
        a = []
        b = []
        w = []
        p = []
        c = []
        j = []
        count = 0
        while True:
            e = self.energy(input, target)
            #delta = np.abs(e-e0)
            print (count+1,':', e)
            err.append(e)
            b.append(list(self.param['b']))
            w.append(list(self.param['w']))
            a.append(list(self.param['a']))
            p.append(list(self.param['p']))
            c.append(self.param['c'])
            j.append(self.rinx)
            if e <= error or epoch is not None and count >= epoch-1:
                errlist['e'] = np.array(err)
                errlist['a'] = np.array(a)
                errlist['b'] = np.array(b)
                errlist['w'] = np.array(w)
                errlist['p'] = np.array(p)
                errlist['c'] = np.array(c)
                errlist['j'] = np.array(j)
                #import pdb; pdb.set_trace()
                return errlist
            count += 1
            #e0 = self.energy(input, target)
            #da, db, dw, dp, dc = self.antigradient(input, target, extend=extend)
            delta = self.antigradient(input, target, extend=extend)
            self.try_dzeta(delta, input, target, extend=extend)



    def error(self, input, target):
        """Error function

        Args:
        array: input: Input series
        array: target: Target resposnse

        Return:
        array: Error
        """
        return target - self.sim(input)

    def energy(self, input, target):
        """Energy function

        Args:
        array: input: Input series
        array: target: Target resposnse

        Return:
        array: Energy of error
        """
        return np.sum(self.error(input, target)**2)/2

    def _tau(self, t, k=None):
        """
        Args:
        t: array: time Point
        k: int: index (opitonal, use when need compute result by neurons)

        Return:
        array: Scaled and shifted time point
        """
        if k is None:
            #import pdb; pdb.set_trace()
            return (t-self.param['b'])/self.param['a']
        else:
            return (t-self.param['b'][k])/self.param['a'][k]

    def antigradient(self, input, target):
        """
        Return:
        da:antigradient by scales
        db: double: Antigradient by shiftes
        dw: double: Antigradient by weightes
        """
        e = self.error(input, target)
        da = np.zeros(self.ncount)
        db = np.zeros(self.ncount)
        dw = np.zeros(self.ncount)
        dp = np.zeros(self.ncount)
        a = self.param['a']
        p = self.param['p']
        w = self.param['w']
        for k in range(self.ncount):
            tau = (input-self.param['b'][k])/self.param['a'][k]
            h_tau = self.h(p[k], tau)
            dw[k] = -np.sum(e*h_tau)
            d = e*w[k]*self.wt._dh_db(p[k], tau, h_tau, a[k])
            db[k] = -np.sum(d)
            da[k] = -np.sum(d*tau)
            dp[k] = -np.sum(e*w[k]*self.wt._dh_dp(p[k], tau, h_tau, a[k]))
        dc = -np.sum(e)
        dd = -np.sum(e*input)
        return {'a': da, 'b': db, 'w': dw, 'p': dp, 'c': dc, 'd':dd}


def test_func(x):
        if x < -2:
            return -2.186*x-12.846-2
        elif -2 <= x < 0:
            return 4.246*x-2
        elif 0 <= x:            
            return 10*np.exp(-.05*x - 0.5)*np.sin((0.3*x + 0.7)*x)-2
        
if __name__ == "__main__":
    t = np.linspace(-10, 10, num=100)
    target = np.vectorize(test_func)(t)
    wn = Wavenet(Morlet, 30, t, target)
    wn.train(t, target, maxiter=300)
    plb.plot(t, target, label='Сигнал')
    plb.plot(t, wn.sim(t), label='Аппроксимация')
    plb.show()
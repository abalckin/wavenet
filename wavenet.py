import numpy as np
import copy
#import scipy_mod.optimize as opt
import scipy.optimize as opt
class Func(object):
    """ Decorator that caches the value gradient of function each time it
    is called. """
    def __init__(self, net, input, target):
        self.net = net
        self.input = input
        self.target = target

    def __call__(self, x, *args):
        self.net.param = self.net.unpack(x)
        return self.net.energy(self.input, self.target)

    def derivative(self, x, *args):
        self.net.param = self.net.unpack(x)
        gr = self.net.antigradient(self.input, self.target, extend=True)
        #import pdb;pdb.set_trace()
        return self.net.pack(gr)


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
        self.param['p'] = np.zeros(ncount)
        self.deltax = np.max(x)-np.min(x)
        countx = (np.max(x)-np.min(x))         
        deltay = np.max(y)-np.min(y)
        self.param['b'] = np.linspace(np.min(x)-countx/2, np.max(x)+countx/2, num=ncount)
        #self.param['b'] = np.random.random(ncount)
        self.param['c'] = np.zeros(1) + y[0]
        #self.param['c']= np.random.random(1)+y[0]
        self.ncount = ncount
        self.tau = np.vectorize(self._tau, cache=True)
        self.h = np.vectorize(self.wt.wavelet, cache=True)
        self.step = np.vectorize(self._step, cache=True)
        self.xcount = x.shape[-1]
        self.param['a'] = np.zeros(ncount)+1
        self.param['w'] = np.zeros(ncount)

    
    def sim(self, t):
        """Simulate network

        Args:
        t: array: Time series

        Returns:
        array: Net output
        """
        #t = (t-self.d)*self.k
        return self.step(t)*t+self.param['c']

    def _step(self, t):
        """
        Compute result for one time moment

        Args:
        double: t: Time moment

        Return:
        double: x Net resulst
        """
        tau = self._tau(t)
        return np.sum(self.wt.wavelet(self.param['p'], tau)*self.param['w'])

    def smart_train(self, input, target, maxiter):
        f = Func(self, input, target)
        x0 = self.pack(self.param)
        res1 = opt.fmin_bfgs(f, x0, fprime=f.derivative, maxiter=maxiter)
        return res1

    def pack(self, param):
        x = np.array([])
        for k in self.param.keys():
#            import pdb; pdb.set_trace()
            x = np.append(x, param[k])
        return np.array(x)

    def unpack(self, aparam):
        inx = 0
        p = {}
        #import pdb; pdb.set_trace()
        for k in self.param.keys():
            l = self.param[k].shape[-1]
            p[k] = aparam[inx:inx+l]
            inx += l
        return p
        
    def train(self, input, target, error, extend=False, epoch=None):
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


    def try_dzeta(self, delta, input, target, extend=False):
        low = copy.deepcopy(self.param)
        hight = copy.deepcopy(self.param)
        low_r = self.rinx/self.dzeta**2
        hig_r = self.rinx*self.dzeta
        for p  in low.keys():
            low[p] += -delta[p]*self.wt.rate[p]*low_r
            hight[p] += -delta[p]*self.wt.rate[p]*hig_r
        self.param = low
        err_low = self.energy(input, target)
        self.param = hight
        if extend:
            self.rinx = hig_r
        err_hig = self.energy(input, target)
        if err_hig > err_low:
            self.param = low
            if extend:
                self.rinx = low_r
            ## self.a += -da*self.wt.ra*self.rinx
            ## self.b += -db*self.wt.rb*self.rinx
            ## self.w += -dw*self.wt.rw*self.rinx
            ## #import pdb; pdb.set_trace()
            ## self.p += -dp*self.wt.rp*self.rinx
            ## self.a = np.clip(self.a, 1e-3, np.Inf)
            ## self.c += -dc*self.wt.rc*self.rinx
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
        return np.sum(self.error(input, target)**2/2)

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

    def antigradient(self, input, target, extend=False):
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
            tau = self.tau(input, k)
            h_tau = self.h(p[k], tau)
            dw[k] = -np.sum(e*h_tau*input)
            d = e*input*w[k]*self.wt._dh_db(p[k], tau, h_tau, a[k])
            db[k] = -np.sum(d)
            da[k] = -np.sum(d*tau)
            if extend:
                dp[k] = -np.sum(e*input*w[k]*self.wt._dh_dp(p[k], tau, h_tau, a[k]))
            else:
                dp[k] = 0.
        if extend:
            dc = -np.sum(e)
        else:
            dc = 0.
        #da = np.clip(da, -1, 1)    
        #db = np.clip(db, 0, 0)
        ## dw = np.clip(dw, -1, 1)
        ## dp = np.clip(dp, -1, 1)
        ## dc = np.clip(dc, -10, 10)
        return {'a': da, 'b': db, 'w': dw, 'p': dp, 'c': dc}


















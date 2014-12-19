import numpy as np


class Wavelet:
    def wavelet(self, tau):
        """Wavelet funtion

        Args:
        double, array: t: Argument

        Return:
        array: Result
        """
        pass

    def _dh_db(tau, h_tau, k):
        pass


class Morlet(Wavelet):
    rp = 500
    ra = 50
    rb = 5
    rw = 5000
    rc = 500
    rate = {'a': ra, 'b': rb, 'c': rc, 'w': rw, 'p': rp}
    pinit = np.pi/100
    def wavelet(p, tau):
        return np.cos(p*tau)*np.exp(-0.5*tau**2)

    def _dh_db(p, tau, h_tau, a):
        return (p*np.sin(p*tau) *
                np.exp(-0.5*tau**2) +
                tau*h_tau)/a

    def _dh_dp(p, tau, h_tau, a):
        return -np.sin(p*tau)*tau*np.exp(-0.5*tau**2)

class MorletB(Wavelet):
    rp = 0.0006
    ra = 0.0001
    rb = 0.0001
    rw = 0.001
    pinit = 0.5
    def wavelet(p, tau):
        return np.cos(0.5*tau)*np.exp(-p*tau**2)

    def _dh_db(p, tau, h_tau, a):
        return (p*np.sin(0.5*tau) *
                np.exp(-p*tau**2) +
                tau*h_tau)/a

    def _dh_dp(p, tau, h_tau, a):
        return -np.cos(0.5*tau)*tau**2*np.exp(-p*tau**2)

class RASP1A(Wavelet):
    rp = 0.001
    ra = 0.05
    rb = 0.05
    rw = 0.01
    pinit = 1.
    def wavelet(p, tau):
        return tau/(tau**2+p)**2

    def _dh_db(p, tau, h_tau, a):
        return (3*tau**2-p)/a/(tau**2+p)**3

    def _dh_dp(p, tau, h_tau, a):
        return (-2*tau/(tau**2+p)**3)

class RASP1B(Wavelet):
    rp = 0.001
    ra = 0.0001
    rb = 0.0001
    rw = 0.001
    pinit = 1.
    def wavelet(p, tau):
        return p*tau/(tau**2+1)**2

    def _dh_db(p, tau, h_tau, a):
        return p*(3*p*tau**2-1)/a/(tau**2+1)**3

    def _dh_dp(p, tau, h_tau, a):
        return (-2*tau**3/(p*tau**2+1)**3)

class RASP2A(Wavelet):
    rp = 0.000001
    ra = 0.0000001
    rb = 0.00001
    rw = 0.0001
    pinit = 1.
    def wavelet(p, tau):
        return tau*np.cos(tau)/(tau**2+p)

    def _dh_db(p, tau, h_tau, a):
        return (tau*(tau**2+p)/a*np.sin(tau)+(tau**2-p)/a*np.cos(tau))/(tau**2+p)**2

    def _dh_dp(p, tau, h_tau, a):
        return -tau*np.cos(tau)/(tau**2+p)**2

class RASP(Wavelet):
    rp = 0.001
    ra = 0.001
    rb = 0.001
    rw = 0.01
    pinit = 1.
    #rp = 0.0001
    #ra = 0.00001
    #rb = 0.0001
    #rw = 0.001
    #pinit = 1
    def wavelet(p, tau):
        return tau*np.cos(tau)/(tau**2+p)

    def _dh_db(p, tau, h_tau, a):
        return (tau*(tau**2+p)/a*np.sin(tau)+(tau**2-p)/a*np.cos(tau))/(tau**2+p)**2

    def _dh_dp(p, tau, h_tau, a):
        return -tau*np.sin(tau)/(tau**2+p)**2

class RASP3(Wavelet):
    rp = 0.000000001
    ra = 0.000000001
    rb = 0.000000001
    rw = 0.00000001
    pinit = 3.14
    def wavelet(p, tau):
        return np.sin(tau*p)/(tau**2-1)

    def _dh_db(p, tau, h_tau, a):
        return (2*tau/a*np.sin(p*tau)-p*(tau**2-1)/a*np.cos(p*tau))/(tau**2-1)**2

    def _dh_dp(p, tau, h_tau, a):
        return tau*np.cos(p*tau)/(tau**2-1)

class SLOG1A(Wavelet):
    rp = 0.00001
    ra = 0.00001
    rb = 0.00001
    rw = 0.00001
    pinit = 0.01

    def wavelet(p, tau):
        return p/(1+np.exp(-tau-3)) +\
        p/(1+np.exp(-tau+3)) -\
        p/(1+np.exp(-tau-3)) +\
        p/(1+np.exp(-tau-1))

    def _dh_db(p, tau, h_tau, a):
        return (-p*np.exp(-tau+1)/(1+np.exp(-tau+1))**2 +
                p*np.exp(-tau+3)/(1+np.exp(-tau+3))**2 +
                p*np.exp(-tau-3)/(1+np.exp(-tau-3))**2 +
                p*np.exp(-tau-1)/(1+np.exp(-tau-1))**2)/a

    def _dh_dp(p, tau, h_tau, a):
        return h_tau

class ShannonA(Wavelet):
    rp = 0.0000001
    ra = 0.000000001
    rb = 0.0000000001
    rw = 0.000001
    pinit = 3.14

    def wavelet(p, tau):
        #import pdb; pdb.set_trace()
        return (np.sin(2*p*tau)-np.sin(p*tau))/p/tau

    def _dh_db(p, tau, h_tau, a):
        return p/a * (p*tau*np.cos(p*tau) -
                      2*p*np.cos(2*p*tau) +
                      np.sin(p*tau) +
                      np.sin(2*p*tau))/(p*tau)**2

    def _dh_dp(p, tau, h_tau, a):
        return (-tau*np.cos(p*tau) -
                2*tau*np.cos(2*p*tau)/(p*tau)**2 -
                (2*-np.sin(p*tau) +
                np.sin(2*p*tau)/p**3/tau**2))


class POLYWOG(Wavelet):
    rp = 0.1
    ra = 0.01
    rb = 0.02
    rw = 0.01
    pinit = 1.

    def wavelet(p, tau):
        return (1-p*tau**2)*np.exp(-0.5*tau**2)

    def _dh_db(p, tau, h_tau, a):
        #return p*(tau**2+1)*np.exp(-0.5*tau**2)/a
        return (3*p*tau-p*tau**3)*np.exp(-0.5*tau**2)/a

    def _dh_dp(p, tau, h_tau, a):
        return -tau**2*np.exp(-0.5*tau**2)

class POLYWOG2(Wavelet):
    rp = 0.001
    ra = 0.0001
    rb = 0.0001
    rw = 0.001
    pinit = 1.

    def wavelet(p, tau):
        return p*tau*np.exp(-0.5*tau**2)

    def _dh_db(p, tau, h_tau, a):
        return (p*tau**2+1)*np.exp(-0.5*tau**2)/a

    def _dh_dp(p, tau, h_tau, a):
        return tau*np.exp(-0.5*tau**2)
















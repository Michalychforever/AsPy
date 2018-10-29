import numpy as np
from multiprocessing import Pool
import scipy
from numpy import exp, log, log10, cos, sin, pi, cosh, sinh , sqrt
from scipy.special import gamma
from scipy.special import *
from scipy import interpolate
import sys
from time import time
from scipy.interpolate import interp1d
from scipy.integrate import quad
import scipy.integrate as integrate


#dWtabdown = np.linspace(-0.9,-0.1,15)
##dWtabup = np.linspace(log(0.1),log(10.),10)
##dWtabup = exp(dWtabup)
#dWtabup = np.linspace(0.1,9.0,15)
#dWtab=np.concatenate((dWtabdown,dWtabup))
#dWtab = np.linspace(-0.9,-0.1,15)

dWtab = np.array([2.00714286])

# computing some spherical collapse functions


epsinv = 1.49012e-20
fup = lambda x: 9.*(x-sin(x))**2./(1.-cos(x)+epsinv)**3./2.
fdown = lambda x: 9.*(sinh(x)-x)**2./(cosh(x)-1.+epsinv)**3./2.
gup = lambda x: 3.*pow((6.*(x-sin(x))),2./3.)/20.
gdown = lambda x: -3.*pow((6.*(sinh(x)-x)),2./3.)/20.
func0up = lambda x,y: fup(x) - y
Theta0up = lambda y: scipy.optimize.fsolve(func0up,2.64,args=y,fprime = lambda x, y: 9.*(x-sin(x))/(1.-cos(x))**2. - 27.*sin(x)*(-x+sin(x))**2./2./(1.-cos(x))**4.,xtol=1.49012e-11)[0]
Fup = lambda x: gup(Theta0up(x))
func0down = lambda x,y: fdown(x) - y
Theta0down = lambda y: scipy.optimize.fsolve(func0down,1.,args=y,fprime = lambda x, y: 9.*(x-sinh(x))*(-9.+8.*cosh(x)+cosh(2.*x)-6.*x*sinh(x))/(4.*(-1.+cosh(x))**4.),xtol=1.49012e-11)[0]
Fdown = lambda x: gdown(Theta0down(x))

def F(x):
    if x >= 1:
        return Fup(x)
    else:
        return Fdown(x)

fupprime = lambda x: 9.*(x-sin(x))/(1.-cos(x))**2. - 27.*sin(x)*(-x+sin(x))**2./2./(1.-cos(x))**4.
gupprime = lambda x: 3.**(2./3.)*(1.-cos(x))/5./2.**(1./3.)/(x-sin(x))**(1./3.)
fdownprime = lambda x: 9.*(-x+sinh(x))/(-1.+cosh(x))**2. - 27.*sinh(x)*(-x+sinh(x))**2./2./(1.-cosh(x))**4.
gdownprime = lambda x: 3.**(2./3.)*(1.-cosh(x))/5./2.**(1./3.)/(sinh(x)-x)**(1./3.)

Fprimeup = lambda x: gupprime(Theta0up(x))/fupprime(Theta0up(x))
Fprimedown = lambda x: gdownprime(Theta0down(x))/fdownprime(Theta0down(x))

def Fprime(x):
    if x > 1:
        return Fprimeup(x)
    if x < 1:
        return Fprimedown(x)
    else:
        return 1.


### interpolating the initial power spectrum and computing the variance

d=np.loadtxt('matterpower_horizonRun3.dat')
k0=d[:,0]; P0=d[:,1]
Plin = interpolate.InterpolatedUnivariateSpline(k0,P0,ext=3)
W = lambda x: (3./x**3.)*(sin(x) - x*cos(x))

### some global parameters

R = 10.

### grid and perturbation parameters

# main grid parameters

Nx = 1500
Nt = 500
jRin = 400  #number of Rin in the array of coordinates

rmin = 1.e-2
r0 = 1.

etamin = -7.
etamax = 0.
tau = (etamax - etamin)/Nt
T = np.linspace(etamin,etamax,num=Nt+1)

# inputting Omega/f^2

dOm=np.loadtxt('Omftab.dat')
eta0=dOm[:,0]; Om0=dOm[:,1]
Omffunc = interpolate.InterpolatedUnivariateSpline(eta0,Om0,ext=3)
Omf = Omffunc(T)
alpha = 3.*Omffunc(T) - 2.

# parameters of perturbations

l = 1
Nk = 199
kmin = 0.001
kmax = 5.

log_mode = True
#log_mode = False
multi_processing = True
standard_integration = True
#standard_integration = False

if log_mode > 0:
    ktab = np.linspace(log(kmin),log(kmax),Nk+1)
    ktab = exp(ktab) 
else:
    ktab = np.linspace(kmin,kmax,Nk+1)

rmin2 = 1.e-5

### cycle over dW starts
detQhat = np.zeros(len(dWtab))
detDIR = np.zeros(len(dWtab))
detDIR2 = np.zeros(len(dWtab))
traceQhat = np.zeros(len(dWtab))
A1tab = np.zeros(len(dWtab))
A2tab = np.zeros(len(dWtab))


for dWindex in range(len(dWtab)):

    ### define the background input quantities

    deltaW = np.asscalar(dWtab[dWindex])
    Rin = R * pow((1+deltaW),1/3.)
    print('deltaW = ',deltaW)

    # dervied grid parameters

    rmax = rmin + (Rin - rmin)*Nx/jRin
    h = (rmax - rmin)/Nx
    X = np.linspace(rmin,rmax,num=Nx+1)
    print('Rin =',X[jRin])
    print('rmax = ',rmax)

    ### Compute the background configuration

    print('computing the background...')

    t01 = time()



    if standard_integration>0:

        integrand1 = lambda k: exp(3.*k)*Plin(exp(k))*(W(exp(k)*Rin))**2./(2.*pi**2.)
        SigmaSquareRofdeltaW  = integrate.quad(integrand1, log(k0[0]), log(k0[-1]))[0]
        print('SigmaSquareRofdeltaW = ', SigmaSquareRofdeltaW)

        rinvals = np.linspace(log(rmin2),log(rmax),3000)
        rinvals = exp(rinvals)

        xiRintab = np.zeros(len(rinvals))
        for j in range(len(rinvals)):
            integrand2 = lambda k: exp(2.*k)*Plin(exp(k))*W(exp(k)*Rin)*sin(exp(k)*rinvals[j])/(2.*pi**2.)
            xiRintab[j] = integrate.quad(integrand2, log(k0[0]), log(k0[-1]))[0]/rinvals[j]

        xiRin = interpolate.InterpolatedUnivariateSpline(rinvals,xiRintab,bbox=[rinvals[0], rinvals[-1]],ext=3)

        deltaL = lambda x: F(1.+deltaW)/SigmaSquareRofdeltaW*xiRin(x)

        rinvals2 = np.linspace(rinvals[0],rinvals[-1],3000)
        integrand3 = lambda x: x**2.*deltaL(x)
        deltaLbartab = np.zeros(len(rinvals2))
        for j in range(1,len(rinvals2)):
            deltaLbartab[j] = integrate.quad(integrand3, rinvals2[0], rinvals2[j])[0]*3./rinvals2[j]**3.
        deltaLbartab[0] = deltaLbartab[1]
        deltaLbar = interpolate.InterpolatedUnivariateSpline(rinvals2,deltaLbartab,bbox=[rinvals2[0], rinvals2[-1]],ext=3)

        deltaLbarprime = lambda rin: 3./rin*(deltaL(rin)-deltaLbar(rin))

        deltaLprimetab1 = np.zeros(len(rinvals))
        for j in range(len(rinvals)):
            integrand4 = lambda k: exp(3.*k)*Plin(exp(k))*W(exp(k)*Rin)*cos(exp(k)*rinvals[j])/(2.*pi**2.)/rinvals[j]**0.
            deltaLprimetab1[j] = integrate.quad(integrand4, log(k0[0]), log(k0[-1]))[0]
        deltaLprime1 = interpolate.InterpolatedUnivariateSpline(rinvals,deltaLprimetab1,bbox=[rinvals[0], rinvals[-1]],ext=3)
        deltaLprime = lambda x: (-1.*xiRin(x)/x + deltaLprime1(x)/x)*F(1.+deltaW)/SigmaSquareRofdeltaW

    else:

        SigmaSquareRofdeltaW = 0.
        Nint = 4000;
        integrand1 = np.zeros(Nint)
        integrandk = np.zeros(Nint)
        for i in range(Nint):
            integrandk[i] = k0[0] * exp(i*log(k0[-1]/k0[0])/(Nint-1.))
            integrand1[i] = Plin(integrandk[i])*(W(integrandk[i]*Rin))**2.*integrandk[i]**2./(2.*pi**2.)
        for i in range(Nint-1):
            SigmaSquareRofdeltaW = SigmaSquareRofdeltaW + (log(integrandk[i+1])-log(integrandk[i]))*(integrand1[i+1]*integrandk[i+1]+integrand1[i]*integrandk[i])/2.
        print('SigmaSquareRofdeltaW = ', SigmaSquareRofdeltaW)


        rinvals = np.linspace(rmin2,rmax,3000)

        Nint2 = 1500;
        xiRintab = np.zeros(len(rinvals))
        integrand2 = np.zeros(Nint2)
        integrandk2 = np.zeros(Nint2)
        for j in range(len(rinvals)):
            xiRintab[j] = 0.
            for i in range(Nint2):
                integrandk2[i] = k0[0] * exp(i*log(k0[-1]/k0[0])/(Nint2-1.))
                integrand2[i] = Plin(integrandk2[i])*W(integrandk2[i]*Rin)*sin(integrandk2[i]*rinvals[j])*integrandk2[i]/(2.*pi**2.)/rinvals[j]
            for i in range(Nint2-1):
                xiRintab[j] = xiRintab[j] + (log(integrandk2[i+1])-log(integrandk2[i]))*(integrand2[i+1]*integrandk2[i+1]+integrand2[i]*integrandk2[i])/2.

        xiRin = interpolate.InterpolatedUnivariateSpline(rinvals,xiRintab,bbox=[rinvals[0], rinvals[-1]],ext=3)


        deltaL = lambda x: F(1.+deltaW)/SigmaSquareRofdeltaW*xiRin(x)

#        rinvals2 = np.linspace(rmin2,rmax,1000)
#        Nint3 = 1500;
#        deltaLbartab = np.zeros(len(rinvals2))
#        integrand3 = np.zeros(Nint3)
#        integrandr3 = np.zeros(Nint3)
#        for j in range(1,len(rinvals2)):
#            deltaLbartab[j] = 0.
#            for i in range(Nint3):
#                integrandr3[i] = rinvals2[0] + (rinvals2[j] - rinvals2[0])*i/(Nint3-1.)
#                integrand3[i] = deltaL(integrandr3[i])*integrandr3[i]**2.*3./rinvals2[j]**3.
#            for i in range(Nint3-1):
#                deltaLbartab[j] = deltaLbartab[j] + (integrandr3[i+1]-integrandr3[i])*(integrand3[i+1]+integrand3[i])/2.
#        deltaLbartab[0] = deltaLbartab[1]
#        deltaLbar = interpolate.InterpolatedUnivariateSpline(rinvals2,deltaLbartab,bbox=[rinvals2[0], rinvals2[-1]],ext=3)

#        integrand3 = lambda x: x**2.*deltaL(x)
#        deltaLbartab = np.zeros(len(rinvals))
#        for j in range(1,len(rinvals)):
#            deltaLbartab[j] = integrate.quad(integrand3, rinvals[0], rinvals[j])[0]*3./rinvals[j]**3.
#        deltaLbartab[0] = deltaLbartab[1]
#deltaLbar = interpolate.InterpolatedUnivariateSpline(rinvals,deltaLbartab,bbox=[rinvals[0], rinvals[-1]],ext=3)

        integrand3 = np.zeros(Nint2)
        integrandk3 = np.zeros(Nint2)
        deltaLbartab0 = np.zeros(len(rinvals))
        for j in range(len(rinvals)):
            deltaLbartab0[j] = 0.
            for i in range(Nint2):
                integrandk3[i] = k0[0] * exp(i*log(k0[-1]/k0[0])/(Nint2-1.))
                integrand3[i] = (F(1.+deltaW)/SigmaSquareRofdeltaW)*Plin(integrandk2[i])*W(integrandk2[i]*Rin)*W(integrandk2[i]*rinvals[j])*integrandk2[i]**2./(2.*pi**2.)
            for i in range(Nint2-1):
                deltaLbartab0[j] = deltaLbartab0[j] + (log(integrandk3[i+1])-log(integrandk3[i]))*(integrand3[i+1]*integrandk3[i+1]+integrand3[i]*integrandk3[i])/2.        

        deltaLbar = interpolate.InterpolatedUnivariateSpline(rinvals,deltaLbartab0,bbox=[rinvals[0], rinvals[-1]],ext=3)

        deltaLbarprime = lambda rin: 3./rin*(deltaL(rin)-deltaLbar(rin))

        Nint4 = 1500;
        deltaLprimetab1 = np.zeros(len(rinvals))
        integrand4 = np.zeros(Nint4)
        integrandk4 = np.zeros(Nint4)
        for j in range(len(rinvals)):
            deltaLprimetab1[j] = 0.
            for i in range(Nint4):
                integrandk4[i] = k0[0] * exp(i*log(k0[-1]/k0[0])/(Nint4 - 1.))
                integrand4[i] = Plin(integrandk4[i])*W(integrandk4[i]*Rin)*cos(integrandk4[i]*rinvals[j])*integrandk4[i]**2./(2.*pi**2.)/rinvals[j]**0.
            for i in range(Nint4-1):
                deltaLprimetab1[j] = deltaLprimetab1[j] + (log(integrandk4[i+1])-log(integrandk4[i]))*(integrand4[i+1]*integrandk4[i+1]+integrand4[i]*integrandk4[i])/2.

        deltaLprime1 = interpolate.InterpolatedUnivariateSpline(rinvals,deltaLprimetab1,bbox=[rinvals[0], rinvals[-1]],ext=3)
        deltaLprime = lambda x: (-1.*xiRin(x)/x + deltaLprime1(x)/x)*F(1.+deltaW)/SigmaSquareRofdeltaW


    # computing the background functions: overdensity

    func = lambda x,y: x-sin(x) - y
    @np.vectorize
    def Thetanew(y):
        a = scipy.optimize.fsolve(func,2.,args=y,fprime = lambda x, y: 1. - cos(x),xtol=1.49012e-11)[0]
        return a

    ThetaupBG = lambda eta, rin: Thetanew(exp(3.*eta/2.)/6.*(20./3.*deltaLbar(rin))**1.5)

    rup = lambda eta, rin: rin*(2./9.)**(1./3.)*(1. - cos(ThetaupBG(eta,rin)))/(ThetaupBG(eta,rin) - sin(ThetaupBG(eta,rin)))**(2./3.)

    drupdrin = lambda eta, rin: rup(eta,rin)/rin*(1.+rin*deltaLbarprime(rin)/deltaLbar(rin)*(3.*(ThetaupBG(eta,rin)-sin(ThetaupBG(eta,rin)))*sin(ThetaupBG(eta,rin))/2./(1.-cos(ThetaupBG(eta,rin)))**2.-1.))

    deltaupBG = lambda eta, rin: (rin/rup(eta, rin))**2.*drupdrin(eta, rin)**(-1.) - 1.

    drPsiupBG = lambda eta, rin: -rin*(2./9.)**(1./3.)*(1. - cos(ThetaupBG(eta,rin)))/(ThetaupBG(eta,rin) - sin(ThetaupBG(eta,rin)))**(2./3.)*(3.*(ThetaupBG(eta,rin) - sin(ThetaupBG(eta,rin)))*sin(ThetaupBG(eta,rin))/2./(1. - cos(ThetaupBG(eta,rin)))**2. - 1.)

    dThetaupBGdrin = lambda eta, rin: deltaLbarprime(rin)/deltaLbar(rin)*3./2.*(ThetaupBG(eta,rin) - sin(ThetaupBG(eta,rin)))/(1. - cos(ThetaupBG(eta,rin)))

    TThetaupBG = lambda eta, rin: -3.*(3.*(ThetaupBG(eta,rin) - sin(ThetaupBG(eta,rin)))*sin(ThetaupBG(eta,rin))/2./(1. - cos(ThetaupBG(eta,rin)))**2. - 1.) - rup(eta, rin)*drupdrin(eta,rin)**(-1.)*((-3./8.)*sin(ThetaupBG(eta,rin)/2.)**(-4.)*(2*ThetaupBG(eta,rin) + ThetaupBG(eta,rin)*cos(ThetaupBG(eta,rin)) - 3.*sin(ThetaupBG(eta,rin))))*dThetaupBGdrin(eta, rin)

    ddrupddrin = lambda eta, rin: 1./rup(eta,rin)*drupdrin(eta,rin)**2. - 1./rin*drupdrin(eta,rin) + 3.*rup(eta,rin)/rin*((deltaLprime(rin)/deltaLbar(rin)-deltaL(rin)*deltaLbarprime(rin)/deltaLbar(rin)**2.)*(3.*(ThetaupBG(eta,rin)-sin(ThetaupBG(eta,rin)))*sin(ThetaupBG(eta,rin))/2./(1.-cos(ThetaupBG(eta,rin)))**2.-1.)+(deltaL(rin)/deltaLbar(rin)-1.)*(-3./8.*(sin(ThetaupBG(eta,rin)/2.))**(-4.)*(2.*ThetaupBG(eta,rin)+ThetaupBG(eta,rin)*cos(ThetaupBG(eta,rin))-3.*sin(ThetaupBG(eta,rin))))*dThetaupBGdrin(eta,rin))


    ddThetaupBGddrin = lambda eta, rin: 3.*(ThetaupBG(eta,rin)-sin(ThetaupBG(eta,rin)))/2./(1.-cos(ThetaupBG(eta,rin)))*(-3./rin**2.*(deltaL(rin)/deltaLbar(rin)-1.)+3./rin*(deltaLprime(rin)/deltaLbar(rin) - deltaL(rin)*deltaLbarprime(rin)/deltaLbar(rin)**2.))+3./rin*(deltaL(rin)/deltaLbar(rin)-1.)*(-1.*((3.*(-2.+2.*cos(ThetaupBG(eta,rin))+ThetaupBG(eta,rin)*sin(ThetaupBG(eta,rin))))/(2.*(-1.+cos(ThetaupBG(eta,rin)))**2.)))*dThetaupBGdrin(eta,rin)


    dTThetaupBGdrin = lambda eta,rin: -4.*(-3./8.*sin(ThetaupBG(eta,rin)/2.)**(-4.)*(2.*ThetaupBG(eta,rin) + ThetaupBG(eta,rin)*cos(ThetaupBG(eta,rin)) - 3.*sin(ThetaupBG(eta,rin))))*dThetaupBGdrin(eta, rin) + rup(eta, rin)*drupdrin(eta, rin)**(-2.)*ddrupddrin(eta,rin)*(-3./8.*sin(ThetaupBG(eta,rin)/2.)**(-4.)*(2.*ThetaupBG(eta,rin) + ThetaupBG(eta,rin)*cos(ThetaupBG(eta,rin)) - 3.*sin(ThetaupBG(eta,rin))))*dThetaupBGdrin(eta, rin)-rup(eta, rin)*drupdrin(eta, rin)**(-1.)*(3./16.*sin(ThetaupBG(eta,rin)/2)**(-5.)*(11.*ThetaupBG(eta,rin)*cos(ThetaupBG(eta,rin)/2.) + ThetaupBG(eta,rin)*cos((3.*ThetaupBG(eta,rin))/2.) - 4.*(3.*sin(ThetaupBG(eta,rin)/2.) + sin((3.*ThetaupBG(eta,rin))/2.))))*dThetaupBGdrin(eta, rin)**2. - rup(eta, rin)*drupdrin(eta,rin)**(-1.)*(-3./8.*sin(ThetaupBG(eta,rin)/2)**(-4.)*(2.*ThetaupBG(eta,rin) + ThetaupBG(eta,rin)*cos(ThetaupBG(eta,rin)) - 3.*sin(ThetaupBG(eta,rin))))*ddThetaupBGddrin(eta, rin)

    # computing the background functions: underdensity

    funcdown = lambda x,y: sinh(x) - x - y

    @np.vectorize
    def Thetadown(y):
        a0 = 5.
        a = scipy.optimize.fsolve(funcdown,a0,args=y,fprime = lambda x, y: cosh(x) - 1.,xtol=1.49012e-11)[0]
        return a

    ThetadownBG = lambda eta, rin: Thetadown(exp(3.*eta/2.)/6.*(-20./3.*deltaLbar(rin))**1.5)

    rdown = lambda eta, rin: rin*(2./9.)**(1./3.)*(cosh(ThetadownBG(eta,rin))-1.)/(-1.*ThetadownBG(eta,rin) + sinh(ThetadownBG(eta,rin)))**(2./3.)

    drdowndrin = lambda eta, rin: rdown(eta,rin)/rin*(1.+rin*deltaLbarprime(rin)/deltaLbar(rin)*(3.*(-1.*ThetadownBG(eta,rin)+sinh(ThetadownBG(eta,rin)))*sinh(ThetadownBG(eta,rin))/2./(-1.+cosh(ThetadownBG(eta,rin)))**2.-1.))

    drPsidownBG = lambda eta, rin: -rdown(eta,rin)*(3.*(-1.*ThetadownBG(eta,rin) + sinh(ThetadownBG(eta,rin)))*sinh(ThetadownBG(eta,rin))/2./(cosh(ThetadownBG(eta,rin))-1.)**2. - 1.)

    dThetadownBGdrin = lambda eta, rin: deltaLbarprime(rin)/deltaLbar(rin)*3./2.*(-ThetadownBG(eta,rin) + sinh(ThetadownBG(eta,rin)))/(-1. + cosh(ThetadownBG(eta,rin)))

    TThetadownBG = lambda eta, rin: -3.*(3.*(-1.*ThetadownBG(eta,rin) + sinh(ThetadownBG(eta,rin)))*sinh(ThetadownBG(eta,rin))/2./(-1. + cosh(ThetadownBG(eta,rin)))**2. - 1.) - rdown(eta, rin)*drdowndrin(eta,rin)**(-1.)*((3./8.)*sinh(ThetadownBG(eta,rin)/2.)**(-4.)*(2.*ThetadownBG(eta,rin) + ThetadownBG(eta,rin)*cosh(ThetadownBG(eta,rin)) - 3.*sinh(ThetadownBG(eta,rin))))*dThetadownBGdrin(eta, rin)


    ddrdownddrin = lambda eta, rin: 1./rdown(eta,rin)*drdowndrin(eta,rin)**2. - 1./rin*drdowndrin(eta,rin) + 3.*rdown(eta,rin)/rin*((deltaLprime(rin)/deltaLbar(rin)-deltaL(rin)*deltaLbarprime(rin)/deltaLbar(rin)**2.)*(3.*(-1.*ThetadownBG(eta,rin)+sinh(ThetadownBG(eta,rin)))*sinh(ThetadownBG(eta,rin))/2./(-1.+cosh(ThetadownBG(eta,rin)))**2.-1.)+(deltaL(rin)/deltaLbar(rin)-1.)*(3./8.*(sinh(ThetadownBG(eta,rin)/2.))**(-4.)*(2.*ThetadownBG(eta,rin)+ThetadownBG(eta,rin)*cosh(ThetadownBG(eta,rin))-3.*sinh(ThetadownBG(eta,rin))))*dThetadownBGdrin(eta,rin))

    ddThetadownBGddrin = lambda eta, rin: 3.*(-1.*ThetadownBG(eta,rin)+sinh(ThetadownBG(eta,rin)))/2./(-1.+cosh(ThetadownBG(eta,rin)))*(-3./rin**2.*(deltaL(rin)/deltaLbar(rin)-1.)+3./rin*(deltaLprime(rin)/deltaLbar(rin) - deltaL(rin)*deltaLbarprime(rin)/deltaLbar(rin)**2.))+3./rin*(deltaL(rin)/deltaLbar(rin)-1.)*(6.-6.*cosh(ThetadownBG(eta,rin))+3.*ThetadownBG(eta,rin)*sinh(ThetadownBG(eta,rin)))/(2.*(-1.+cosh(ThetadownBG(eta,rin)))**2.)*dThetadownBGdrin(eta,rin)

    dTThetadownBGdrin = lambda eta,rin: -4.*(3./8.*sinh(ThetadownBG(eta,rin)/2.)**(-4.)*(2.*ThetadownBG(eta,rin) + ThetadownBG(eta,rin)*cosh(ThetadownBG(eta,rin)) - 3.*sinh(ThetadownBG(eta,rin))))*dThetadownBGdrin(eta, rin) + rdown(eta, rin)*drdowndrin(eta, rin)**(-2.)*ddrdownddrin(eta,rin)*(3./8.*sinh(ThetadownBG(eta,rin)/2.)**(-4.)*(2.*ThetadownBG(eta,rin) + ThetadownBG(eta,rin)*cosh(ThetadownBG(eta,rin)) - 3.*sinh(ThetadownBG(eta,rin))))*dThetadownBGdrin(eta, rin)-rdown(eta, rin)*drdowndrin(eta, rin)**(-1.)*(-3./16.*sinh(ThetadownBG(eta,rin)/2.)**(-5.)*(11.*ThetadownBG(eta, rin)*cosh(ThetadownBG(eta,rin)/2.) + ThetadownBG(eta,rin)*cosh((3.*ThetadownBG(eta,rin))/2.) - 4.*(3.*sinh(ThetadownBG(eta,rin)/2.) + sinh((3.*ThetadownBG(eta,rin))/2.))))*dThetadownBGdrin(eta, rin)**2. - rdown(eta, rin)*drdowndrin(eta,rin)**(-1.)*(3./8.*sinh(ThetadownBG(eta,rin)/2.)**(-4.)*(2.*ThetadownBG(eta,rin) + ThetadownBG(eta,rin)*cosh(ThetadownBG(eta,rin)) - 3.*sinh(ThetadownBG(eta,rin))))*ddThetadownBGddrin(eta, rin)
    def r(eta,rin):
        if deltaW > 0.:
            return rup(eta,rin)
        else:
            return rdown(eta,rin)
    def drdrin(eta,rin):
        if deltaW > 0.:
            return drupdrin(eta,rin)
        else:
            return drdowndrin(eta,rin)

    def ddrddrin(eta,rin):
        if deltaW > 0.:
            return ddrupddrin(eta,rin)
        else:
            return ddrdownddrin(eta,rin)

    def drPsiBG(eta,rin):
        if deltaW > 0.:
            return drPsiupBG(eta,rin)
        else:
            return drPsidownBG(eta,rin)

    def TThetaBG(eta,rin):
        if deltaW > 0.:
            return TThetaupBG(eta,rin)
        else:
            return TThetadownBG(eta,rin)

    def dTThetaBGdrin(eta,rin):
        if deltaW > 0.:
            return dTThetaupBGdrin(eta,rin)
        else:
            return dTThetadownBGdrin(eta,rin)


    if multi_processing>0:
        def r2(arg):
            return r(arg[0], arg[1])
        def drdrin2(arg):
            return drdrin(arg[0], arg[1])
        def drPsi2(arg):
            return drPsiBG(arg[0], arg[1])
        def TTheta2(arg):
            return TThetaBG(arg[0], arg[1])
        def ddrddrin2(arg):
            return ddrddrin(arg[0], arg[1])
        def dTThetaBGdrin2(arg):
            return dTThetaBGdrin(arg[0], arg[1])

        temp = []
        for i in range(len(X)):
            temp.append(np.meshgrid(T, X[i], sparse=False, indexing='ij'))
        with Pool(28) as p:
            resr = (p.map(r2, temp))
            rBG = np.array(resr).reshape([Nx+1, Nt+1]).T
            resdrdrin = (p.map(drdrin2, temp))
            drdrinBG = np.array(resdrdrin).reshape([Nx+1, Nt+1]).T
            resdrPsi = (p.map(drPsi2, temp))
            drPsiBGmat = np.array(resdrPsi).reshape([Nx+1, Nt+1]).T
            resTTheta = (p.map(TTheta2, temp))
            TThetaBGmat = np.array(resTTheta).reshape([Nx+1, Nt+1]).T
            resddrddrin = (p.map(ddrddrin2, temp))
            ddrddrinBG = np.array(resddrddrin).reshape([Nx+1, Nt+1]).T
            resdTThetaBGdrin = (p.map(dTThetaBGdrin2, temp))
            dTThetaBGdrinmat = np.array(resdTThetaBGdrin).reshape([Nx+1, Nt+1]).T
    else:

        rBG = r(*np.meshgrid(T, X, sparse=False, indexing='ij'))
        drdrinBG = drdrin(*np.meshgrid(T, X, sparse=False, indexing='ij'))
        drPsiBGmat = drPsiBG(*np.meshgrid(T, X, sparse=False, indexing='ij'))
        TThetaBGmat = TThetaBG(*np.meshgrid(T, X, sparse=False, indexing='ij'))
        ddrddrinBG = ddrddrin(*np.meshgrid(T, X, sparse=False, indexing='ij'))
        dTThetaBGdrinmat = dTThetaBGdrin(*np.meshgrid(T, X, sparse=False, indexing='ij'))


    OmtabA = np.zeros((Nt+1,Nx+1))
    for j in range(len(X)):
        for k in range(len(T)):
            OmtabA[k][j] = alpha[k]

    A = -0.5*OmtabA + 2.*(TThetaBGmat - 2.*drPsiBGmat/rBG)
    C = (dTThetaBGdrinmat/drdrinBG - 4.*(TThetaBGmat - 3.*drPsiBGmat/rBG)/rBG)/drdrinBG
    D = 2.*(TThetaBGmat - 3.*drPsiBGmat/rBG)/rBG/rBG
    E = TThetaBGmat
    G = (1./rBG/rBG)*X*X/drdrinBG
    F1 = (2./rBG/rBG*X*(1./drdrinBG - 1./rBG*X) - 1./rBG/rBG*X*X*ddrddrinBG/drdrinBG/drdrinBG)/drdrinBG/drdrinBG

    t02 = time()

    print('background computation time (in sec.) =', t02-t01)

    ### Solving PDE's for linear aspherical fluctuations


    print('Computing linear aspherical fluctuations...')

    u = np.zeros((Nk+1,Nt+1,Nx+1));
    psi = np.zeros((Nk+1,Nt+1,Nx+1));
    rho = np.zeros((Nk+1,Nt+1,Nx+1));

    rhofinal = np.zeros((Nk+1,Nt+1,Nx+1));
    ufinal = np.zeros((Nk+1,Nt+1,Nx+1));
    psifinal = np.zeros((Nk+1,Nt+1,Nx+1));

    PsiPrimeRin = np.zeros((Nk+1,Nt+1))
    PsiRin = np.zeros((Nk+1,Nt+1))
    DeltaRin = np.zeros((Nk+1,Nt+1))
    ThetaRin = np.zeros((Nk+1,Nt+1))

    t1 = time()

    for kindex in range(Nk+1):

        # Initial and boundary conditions:

        for j in range(Nx+1):
            u[kindex][0][j] = (4.*np.pi) * exp(etamin) * spherical_jn(l,ktab[kindex]*X[j]) / (pow(rBG[0][j],l)/(pow(rBG[0][j],l)+pow(r0,l)))
            rho[kindex][0][j] = u[kindex][0][j]
            psi[kindex][0][j] = -u[kindex][0][j]/ktab[kindex]**2. + 4.*np.pi*X[j]/(3.*ktab[kindex])*exp(etamin)/(pow(rBG[0][j],l)/(pow(rBG[0][j],l)+pow(r0,l)))

        # here comes the implicit scheme

        for k in range(Nt):

            b = np.zeros(3*(Nx + 1))
            for j in range(Nx):
                b[j+Nx+1] = u[kindex][k][j]*(1./tau + A[k][j]/2.  - l*(-1.*drPsiBGmat[k][j])/rBG[k][j]*(1. - pow(rBG[k][j],l)/( pow(r0,l) + pow(rBG[k][j],l) ))/2.) + 1.5/2.*rho[kindex][k][j]*Omf[k]+C[k][j]/h/2.*psi[kindex][k][j+1] - psi[kindex][k][j]/2.*(C[k][j]* (1./h-1.*l*pow(r0,l)*drdrinBG[k][j]/(pow(rBG[k][j],l)+pow(r0,l))/rBG[k][j])-D[k][j]*l*(l+1))
                b[j+2*(Nx+1)] = rho[kindex][k][j]*(1./tau+E[k][j]/2. - l*(-1.*drPsiBGmat[k][j])/rBG[k][j]*(1. - pow(rBG[k][j],l)/( pow(r0,l) + pow(rBG[k][j],l) ))/2.)+F1[k][j]/h/2.*psi[kindex][k][j+1]-psi[kindex][k][j]/2.*F1[k][j]* (1./h-1.*l*pow(r0,l)*drdrinBG[k][j]/(pow(rBG[k][j],l)+pow(r0,l))/rBG[k][j])+G[k][j]/2.*u[kindex][k][j]

            # boundary conditions at infinity

            b[2*(Nx+1)-1] = ((4.*np.pi)* exp(T[k+1]) *  scipy.special.spherical_jn(l,ktab[kindex]*rBG[k+1][Nx]) )/ (pow(rBG[k+1][Nx],l)/(pow(rBG[k+1][Nx],l)+pow(r0,l)))
            b[3*(Nx+1)-1] = b[2*(Nx+1)-1]
            b[Nx] = -1.*b[2*(Nx+1)-1]/ktab[kindex]**2. +  4.*np.pi*rBG[k+1][Nx]/(3.*ktab[kindex])*exp(T[k+1])/(pow(rBG[k+1][Nx],l)/(pow(rBG[k+1][Nx],l)+pow(r0,l)))

            L = np.zeros((3*(Nx+1),3*(Nx+1)))

            # imposing boundary conditions
            L[0][0] = 1.
            L[0][2] = -1.
            L[Nx][Nx] = 1.
            L[2*(Nx+1)-1][2*(Nx+1)-1] = 1.
            L[3*(Nx+1)-1][3*(Nx+1)-1] = 1.


            # left upper section: Poisson - psi

            for j in range(1,Nx):

                L[j][j] = -2./h/h/pow(drdrinBG[k+1][j],2.) + l*pow(r0,l)*((l-1)*pow(r0,l)-(l+1)*pow(rBG[k+1][j],l))/pow(pow(r0,l)+pow(rBG[k+1][j],l),2.)/pow(rBG[k+1][j],2.) + 2.*l*pow(r0,l)/(pow(r0,l)+pow(rBG[k+1][j],l))/pow(rBG[k+1][j],2.) - l*(l+1)*1./rBG[k+1][j]/rBG[k+1][j]
                L[j][j-1] = 1./h/h/pow(drdrinBG[k+1][j],2.) - 0.5/h*(-1.*ddrddrinBG[k+1][j]/pow(drdrinBG[k+1][j],3.)+2./rBG[k+1][j]/drdrinBG[k+1][j]*(1+l*pow(r0,l)/(pow(r0,l)+pow(rBG[k+1][j],l))))
                L[j][j+1] = 1./h/h/pow(drdrinBG[k+1][j],2.) + 0.5/h*(-1.*ddrddrinBG[k+1][j]/pow(drdrinBG[k+1][j],3.)+2./rBG[k+1][j]/drdrinBG[k+1][j]*(1+l*pow(r0,l)/(pow(r0,l)+pow(rBG[k+1][j],l))))

            # middle upper section: Poisson - theta
                L[j][j + Nx +1] = -1.

            #left central section: Euler - psi
            for j in range(Nx +1,2*(Nx+1)-1):
                L[j][j - Nx -1] =  (C[k+1][j-Nx-1]* (1./h-1.*l*pow(r0,l)*drdrinBG[k+1][j-Nx-1]/(pow(rBG[k+1][j-Nx-1],l)+pow(r0,l))/rBG[k+1][j-Nx-1]) - D[k+1][j-Nx-1]*l*(l+1))/2.
                L[j][j - Nx] = -1.*C[k+1][j-Nx-1]/h/2.

            #middle central section: Euler - theta
#                L[j][j] = 1./tau - A[k+1][j-Nx-1]/2.
                L[j][j] = 1./tau - A[k+1][j-Nx-1]/2. + l*(-1.*drPsiBGmat[k+1][j-Nx-1])/rBG[k+1][j-Nx-1]*(1. - pow(rBG[k+1][j-Nx-1],l)/( pow(r0,l) + pow(rBG[k+1][j-Nx-1],l) ))/2.

            #right central section: Euler - delta
            for j in range(Nx):
                L[(Nx+1)+j][2*(Nx+1)+j] = -1.5/2.*Omf[k+1]

            #lower left section: Continuity - psi
                L[2*(Nx+1)+j][j] = F1[k+1][j]* (1./h-1.*l*pow(r0,l)*drdrinBG[k+1][j]/(pow(rBG[k+1][j],l)+pow(r0,l))/rBG[k+1][j])/2.
                L[2*(Nx+1)+j][j+1] = -1.*F1[k+1][j]/h/2.

            #lower middle section: Continuity - theta
                L[2*(Nx+1)+j][(Nx+1)+j] = -1.*G[k+1][j]/2.

            #lower right section: Continuity - delta
#                L[2*(Nx+1)+j][2*(Nx+1)+j] = 1./tau - E[k+1][j]/2.
                L[2*(Nx+1)+j][2*(Nx+1)+j] = 1./tau - E[k+1][j]/2. + l*(-1.*drPsiBGmat[k+1][j])/rBG[k+1][j]* (1. - pow(rBG[k+1][j],l)/( pow(r0,l) + pow(rBG[k+1][j],l) ))/2.

            z = np.linalg.solve(L,b)
            for j in range(Nx+1):
                psi[kindex][k+1][j] = z[j]
                u[kindex][k+1][j] =  z[j+Nx+1]
                rho[kindex][k+1][j] =  z[j+2*(Nx+1)]

        rhofinal = np.zeros((Nk+1,Nt+1,Nx+1));
        ufinal = np.zeros((Nk+1,Nt+1,Nx+1));
        psifinal = np.zeros((Nk+1,Nt+1,Nx+1));

        for k in range(Nt+1):
            for j in range(Nx+1):
                rhofinal[kindex][k][j] = rho[kindex][k][j]*(pow(rBG[k][j],l)/(pow(rBG[k][j],l)+pow(r0,l)))
                ufinal[kindex][k][j] = u[kindex][k][j]*(pow(rBG[k][j],l)/(pow(rBG[k][j],l)+pow(r0,l)))
                psifinal[kindex][k][j] = psi[kindex][k][j]*(pow(rBG[k][j],l)/(pow(rBG[k][j],l)+pow(r0,l)))
            PsiPrimeRin[kindex][k] = (psifinal[kindex][k][jRin+1]-psifinal[kindex][k][jRin-1])/drdrinBG[k][jRin]/h/2.
            PsiRin[kindex][k] = psifinal[kindex][k][jRin]
            DeltaRin[kindex][k] = rhofinal[kindex][k][jRin]
            ThetaRin[kindex][k] = ufinal[kindex][k][jRin]

    t2 = time()
    print('Linear fluctuations solved in (sec.)', t2-t1)

    WFofktab = [ [0 for x in range(6)] for y in range((Nk+1)*(Nt+1))];
    for i in range(Nk+1):
        for k in range(Nt+1):
            WFofktab[i*(Nt+1) + k][0] = ktab[i]
            WFofktab[i*(Nt+1) + k][1] = T[k]
            WFofktab[i*(Nt+1) + k][2] = DeltaRin[i][k]
            WFofktab[i*(Nt+1) + k][3] = ThetaRin[i][k]
            WFofktab[i*(Nt+1) + k][4] = PsiRin[i][k]
            WFofktab[i*(Nt+1) + k][5] = PsiPrimeRin[i][k]
    np.savetxt('Dipole_wavefunctions_at_Rin_for_dW'+str(deltaW)+'_x'+str(Nx)+'_t'+str(Nt)+'_Nk'+str(Nk)+'_.dat', WFofktab)

    ### computing upsilons and mu's

    P1 = np.zeros(Nt+1)
    L1 = np.zeros(Nt+1)
    for k in range(Nt+1):
        P1[k] = (1.5*G[k][jRin]-(Rin/rBG[k][jRin])**3.)*Omf[k]
        L1[k] = rBG[k][jRin]**2.*G[k][jRin]

    print('Computing ODEs for mu2 and r2...')



    mustartab = np.zeros((Nk+1,Nk+1))
    UpsilonDelta0 = np.zeros((Nk+1,Nk+1))
    UpsilonTheta0 = np.zeros((Nk+1,Nk+1))
    UpsilonDeltaRin = np.zeros((Nk+1,Nk+1,Nt+1))
    UpsilonThetaRin = np.zeros((Nk+1,Nk+1,Nt+1))

    for kindex1 in range(Nk+1):
        for kindex2 in range(Nk+1):
            for k in range(Nt+1):
                UpsilonDeltaRin[kindex1][kindex2][k] = 1./(4.*np.pi)*(DeltaRin[kindex1][k]*PsiPrimeRin[kindex2][k] + DeltaRin[kindex2][k]*PsiPrimeRin[kindex1][k])/2.
                UpsilonThetaRin[kindex1][kindex2][k] = 1./(4.*np.pi)*(ThetaRin[kindex1][k]*PsiPrimeRin[kindex2][k]+ThetaRin[kindex2][k]*PsiPrimeRin[kindex1][k] - 2./rBG[k][jRin]*(PsiPrimeRin[kindex1][k]*PsiPrimeRin[kindex2][k]+PsiPrimeRin[kindex2][k]*PsiPrimeRin[kindex1][k])+ 2.*l*(l+1)*(PsiPrimeRin[kindex1][k]*PsiRin[kindex2][k]+PsiPrimeRin[kindex2][k]*PsiRin[kindex1][k])/rBG[k][jRin]**2.-1.*l*(l + 1)*(PsiRin[kindex1][k]*PsiRin[kindex2][k]+PsiRin[kindex2][k]*PsiRin[kindex1][k])/rBG[k][jRin]**3.)/2.

            UpsilonDelta0[kindex1][kindex2] = UpsilonDeltaRin[kindex1][kindex2][Nt]
            UpsilonTheta0[kindex1][kindex2] = UpsilonThetaRin[kindex1][kindex2][Nt]


            ### solving the ODE's for mu2 and r2


            bessel1 = 4.*np.pi*scipy.special.spherical_jn(l,ktab[kindex1]*Rin)
            bessel2 = 4.*np.pi*scipy.special.spherical_jn(l,ktab[kindex2]*Rin)
            psi1 = 4.*np.pi*(scipy.special.spherical_jn(l,ktab[kindex1]*Rin)/(-1.*ktab[kindex1]**2.) + Rin/(3.*ktab[kindex1]))
            psi2 = 4.*np.pi*(scipy.special.spherical_jn(l,ktab[kindex2]*Rin)/(-1.*ktab[kindex2]**2.) + Rin/(3.*ktab[kindex2]))
            derivpsi1 = (-1.*l/Rin*scipy.special.spherical_jn(l,ktab[kindex1]*Rin)/ktab[kindex1]**2. + scipy.special.spherical_jn(l+1,ktab[kindex1]*Rin)/ktab[kindex1] + 1./(3.*ktab[kindex1])) *(4.*np.pi)
            derivpsi2 = (-1.*l/Rin*scipy.special.spherical_jn(l,ktab[kindex2]*Rin)/ktab[kindex2]**2. + scipy.special.spherical_jn(l+1,ktab[kindex2]*Rin)/ktab[kindex2] + 1./(3.*ktab[kindex2])) *(4.*np.pi)      

#            derivbessel1 = (1.*l/Rin*scipy.special.spherical_jn(l,ktab[kindex1]*Rin)-scipy.special.spherical_jn(l+1,ktab[kindex1]*Rin)*ktab[kindex1]) *(4.*np.pi)
#            derivbessel2 = (1.*l/Rin*scipy.special.spherical_jn(l,ktab[kindex2]*Rin)-scipy.special.spherical_jn(l+1,ktab[kindex2]*Rin)*ktab[kindex2]) *(4.*np.pi)

            UpsDeltaini = exp(2.*etamin)/(4.*np.pi)* (bessel1*derivpsi2 + bessel2*derivpsi1)/2.
            UspThetaini = exp(2.*etamin)/(4.*np.pi)*(bessel1*derivpsi2 + bessel2*derivpsi1 - 2.*2./Rin*derivpsi1*derivpsi2 + 2.*l*(l+1)/Rin**2.*(psi1*derivpsi2 + psi2*derivpsi1)-2.*l*(l+1)/Rin**3.*psi1*psi2)/2.

            bt = np.zeros(2*(Nt + 1))
            for k in range(1,Nt):
                bt[k+Nt+1] = (rBG[k][jRin]**2.*UpsilonDeltaRin[kindex1][kindex2][k] + rBG[k+1][jRin]**2.*UpsilonDeltaRin[kindex1][kindex2][k+1])/2.
                bt[k] = -1.*UpsilonThetaRin[kindex1][kindex2][k]

            bt[Nt+1] = (rBG[0][jRin]**2.*UpsilonDeltaRin[kindex1][kindex2][0]+rBG[1][jRin]**2.*UpsilonDeltaRin[kindex1][kindex2][1])/2.

            bt[2*(Nt+1)-1] = Rin**2.*(5.*UpsDeltaini+2.*UspThetaini)/7.
            bt[0] = -2.*Rin**2.*(3.*UpsDeltaini + 4.*UspThetaini)/14.

            Matt = np.zeros((2*(Nt+1),2*(Nt+1)))

            # initial conditions for the dot{r2} constraint

            Matt[0][0] = -L1[1]/2./tau
            Matt[0][1] = (L1[2]-L1[0])/tau/2.
            Matt[0][2] = L1[1]/2./tau

            # final condition for r2

            Matt[Nt][Nt] = 1.

            # initial condition for mu2

            Matt[2*(Nt+1)-1][Nt+1] = 1.

            # the rest of the matrix

            for k in range(1,Nt):
                Matt[k][k-1] = 1./tau/tau - 1./tau/4.*alpha[k]
                Matt[k][k] = -2./tau/tau -0.5*Omf[k] + P1[k]
                Matt[k][k+1] = 1./tau/tau + 1./tau/4.*alpha[k]
                Matt[k][(Nt+1)+k] = 1.5/rBG[k][jRin]**2.*Omf[k]
            for k in range(Nt):
                Matt[(Nt+1)+k][k] = (L1[k+1]-L1[k])/tau - L1[k]/tau
                Matt[(Nt+1)+k][k+1] = L1[k]/tau
                Matt[(Nt+1)+k][(Nt+1)+k] = -1./tau
                Matt[(Nt+1)+k][(Nt+1)+k+1] = 1./tau

            zt = np.linalg.solve(Matt,bt)

            r2 = np.zeros(Nt+1)
            mu2 = np.zeros(Nt+1)

            for k in range(Nt+1):
                r2[k] = zt[k]
                mu2[k] =  zt[k+Nt+1]

            mustartab[kindex1][kindex2] = mu2[Nt]

    t3 = time()
    print('Mu2 and r2 equations are solved in (sec.),', t3-t2)

    ### using symmetry of the matrix 

    for kindex1 in range(Nk+1):
        for kindex2 in range(kindex1):
            UpsilonDelta0[kindex2][kindex1] = UpsilonDelta0[kindex1][kindex2]      
            UpsilonTheta0[kindex2][kindex1] = UpsilonTheta0[kindex1][kindex2]
            mustartab[kindex2][kindex1] = mustartab[kindex1][kindex2]

    Qofkstab = [ [0 for x in range(5)] for y in range((Nk+1)*(Nk+1))];
    for kindex1 in range(Nk+1):
        for kindex2 in range(Nk+1):
            Qofkstab[kindex1*(Nk+1)+kindex2][0] = ktab[kindex1]
            Qofkstab[kindex1*(Nk+1)+kindex2][1] = ktab[kindex2]
            Qofkstab[kindex1*(Nk+1)+kindex2][2] = UpsilonDelta0[kindex1][kindex2]
            Qofkstab[kindex1*(Nk+1)+kindex2][3] = UpsilonTheta0[kindex1][kindex2]
            Qofkstab[kindex1*(Nk+1)+kindex2][4] = mustartab[kindex1][kindex2]*3./R**3.
    np.savetxt('Dipole_Qhat_and_Upshat_dW'+str(deltaW)+'_Nk'+str(Nk)+'_.dat', Qofkstab)

    ### computing the trace and determinant

    Dvalue = Fprime(1.+deltaW) + F(1.+deltaW)/(1.+deltaW)*(1. - xiRin(Rin)/SigmaSquareRofdeltaW)
    lambdaStar = -1.*F(1.+deltaW)*Dvalue/SigmaSquareRofdeltaW


    Qhatmatrix = np.zeros((Nk+1,Nk+1))
        
    if log_mode>0:
        dlnk = (log(ktab[Nk])-log(ktab[0]))/Nk
        for kindex1 in range(Nk+1):
            for kindex2 in range(Nk+1):
                Qhatmatrix[kindex1][kindex2] = 2.*lambdaStar*dlnk*sqrt(Plin(ktab[kindex1]))*sqrt(Plin(ktab[kindex2]))*mustartab[kindex1][kindex2]*3./R**3./8./np.pi**3.*exp(1.5*log(ktab[kindex1]))*exp(1.5*log(ktab[kindex2]))
            
    else:
        dk = (kmax - kmin)/Nk
        for kindex1 in range(Nk+1):
            for kindex2 in range(Nk+1):
                Qhatmatrix[kindex1][kindex2] = 2.*lambdaStar*dk*sqrt(Plin(ktab[kindex1]))*sqrt(Plin(ktab[kindex2]))*mustartab[kindex1][kindex2]*3./R**3./8./np.pi**3.*ktab[kindex1]*ktab[kindex2]


    traceQhat[dWindex] = np.trace(Qhatmatrix)
        
    ### adding the Kronecker delta
        
    for kindex1 in range(Nk+1):
        Qhatmatrix[kindex1][kindex1] = Qhatmatrix[kindex1][kindex1] + 1.


    detQhat[dWindex] = np.linalg.det(Qhatmatrix)

    Preftab = [ [0 for x in range(3)] for y in range(1)];
    Preftab[0][0] = dWtab[dWindex]
    Preftab[0][1] = exp(traceQhat[dWindex])
    Preftab[0][2] = detQhat[dWindex]
    np.savetxt('Dipole_determinant_Nk'+str(Nk)+'_dW'+str(deltaW)+'_.dat', Preftab)


    ### solving for B's

    Btab = np.zeros((Nk+1))
    BDelta0 = np.zeros((Nk+1))
    BTheta0 = np.zeros((Nk+1))
    BDeltaRin = np.zeros((Nk+1,Nt+1))
    BThetaRin = np.zeros((Nk+1,Nt+1))

    for kindex1 in range(Nk+1):
        for k in range(Nt+1):
            BDeltaRin[kindex1][k] = (-1./6.)*((1.+TThetaBGmat[k][jRin] - 2.*drPsiBGmat[k][jRin]/rBG[k][jRin])*DeltaRin[kindex1][k] + (F1[k][jRin]*drdrinBG[k][jRin])*PsiPrimeRin[kindex1][k])*exp(T[k])
            BThetaRin[kindex1][k] = (-1./6.)*((1.+TThetaBGmat[k][jRin] - 2.*drPsiBGmat[k][jRin]/rBG[k][jRin])*ThetaRin[kindex1][k] + (dTThetaBGdrinmat[k][jRin]/drdrinBG[k][jRin]-4.*(TThetaBGmat[k][jRin] - 3.*drPsiBGmat[k][jRin]/rBG[k][jRin])/rBG[k][jRin])*PsiPrimeRin[kindex1][k] + (4.*(TThetaBGmat[k][jRin] - 2.*drPsiBGmat[k][jRin]/rBG[k][jRin])/rBG[k][jRin]**2. - 4.*drPsiBGmat[k][jRin]/rBG[k][jRin]**3.)*PsiRin[kindex1][k])*exp(T[k])

        BDelta0[kindex1] = BDeltaRin[kindex1][Nt]
        BTheta0[kindex1] = BThetaRin[kindex1][Nt]


        ### solving the ODE's for B and r2-B

        BDeltaini = BDeltaRin[kindex1][0]
        BThetaini = BThetaRin[kindex1][0]

        btB = np.zeros(2*(Nt + 1))
        for k in range(1,Nt):
            btB[k+Nt+1] = (rBG[k][jRin]**2.*BDeltaRin[kindex1][k] + rBG[k+1][jRin]**2.*BDeltaRin[kindex1][k+1])/2.
            btB[k] = -1.*BThetaRin[kindex1][k]

        btB[Nt+1] = (rBG[0][jRin]**2.*BDeltaRin[kindex1][0]+rBG[1][jRin]**2.*BDeltaRin[kindex1][1])/2.
        btB[2*(Nt+1)-1] = rBG[0][jRin]**2.*(5.*BDeltaini+2.*BThetaini)/7.
        btB[0] = -2.*rBG[0][jRin]**2.*(3.*BDeltaini + 4.*BThetaini)/14.

        MattB = np.zeros((2*(Nt+1),2*(Nt+1)))

        # initial conditions for the dot{r2} constraint

        MattB[0][0] = -L1[1]/2./tau
        MattB[0][1] = (L1[2]-L1[0])/tau/2.
        MattB[0][2] = L1[1]/2./tau

        # final condition for r2

        MattB[Nt][Nt] = 1.

        # initial condition for mu2

        MattB[2*(Nt+1)-1][Nt+1] = 1.

        # the rest of the matrix

        for k in range(1,Nt):
            MattB[k][k-1] = 1./tau/tau - 1./tau/4.*alpha[k]
            MattB[k][k] =  -2./tau/tau -0.5*Omf[k] + P1[k]
            MattB[k][k+1] = 1./tau/tau + 1./tau/4.*alpha[k]
            MattB[k][(Nt+1)+k] = 1.5/rBG[k][jRin]**2.*Omf[k]
        for k in range(Nt):
            MattB[(Nt+1)+k][k] = (L1[k+1]-L1[k])/tau - L1[k]/tau
            MattB[(Nt+1)+k][k+1] = L1[k]/tau
            MattB[(Nt+1)+k][(Nt+1)+k] = -1./tau
            MattB[(Nt+1)+k][(Nt+1)+k+1] = 1./tau

        ztB = np.linalg.solve(MattB,btB)


        r2B = np.zeros(Nt+1)
        mu2B = np.zeros(Nt+1)

        for k in range(Nt+1):
            r2B[k] = ztB[k]
            mu2B[k] =  ztB[k+Nt+1]

        Btab[kindex1] = mu2B[Nt]

    t3 = time()
    print('Mu2 and r2 equations are solved in (sec.),', t3-t2)


    Bofkstab = [ [0 for x in range(4)] for y in range((Nk+1))];
    for kindex1 in range(Nk+1):
        Bofkstab[kindex1][0] = ktab[kindex1]
        Bofkstab[kindex1][1] = BDelta0[kindex1]
        Bofkstab[kindex1][2] = BTheta0[kindex1]
        Bofkstab[kindex1][3] = Btab[kindex1]*3./R**3.
    np.savetxt('Dipole_B_of_k_dW'+str(deltaW)+'_Nk'+str(Nk)+'_.dat', Bofkstab)

        ### computing D_IR

    Dvalue = Fprime(1.+deltaW) + F(1.+deltaW)/(1.+deltaW)*(1. - xiRin(Rin)/SigmaSquareRofdeltaW)
    lambdaStar = -1.*F(1.+deltaW)*Dvalue/SigmaSquareRofdeltaW

    a1 = np.zeros(Nk+1)
    b1 = np.zeros(Nk+1)
    b2 = np.zeros(Nk+1)

    Minv = np.linalg.inv(Qhatmatrix)


    ### solving for A

    ADeltaRin = np.zeros(Nt+1)
    AThetaRin = np.zeros(Nt+1)
    
    for k in range(Nt+1):
        ADeltaRin[k] = (4.*np.pi/9.)*(1.+TThetaBGmat[k][jRin] - 2.*drPsiBGmat[k][jRin]/rBG[k][jRin])*(F1[k][jRin]*drdrinBG[k][jRin])*exp(2.*T[k])
        AThetaRin[k] = (4.*np.pi/9.)*((1.+TThetaBGmat[k][jRin] - 2.*drPsiBGmat[k][jRin]/rBG[k][jRin])*dTThetaBGdrinmat[k][jRin]/drdrinBG[k][jRin] - 2.*(TThetaBGmat[k][jRin] - 2.*drPsiBGmat[k][jRin]/rBG[k][jRin])**2./rBG[k][jRin] + 4.*(TThetaBGmat[k][jRin] - 2.*drPsiBGmat[k][jRin]/rBG[k][jRin])*drPsiBGmat[k][jRin]/rBG[k][jRin]**2. - 2.*drPsiBGmat[k][jRin]**2./rBG[k][jRin]**3.)*exp(2.*T[k])



    ADeltaini = ADeltaRin[0]
    AThetaini = AThetaRin[0]

    btA = np.zeros(2*(Nt + 1))
    for k in range(1,Nt):
        btA[k+Nt+1] = (rBG[k][jRin]**2.*ADeltaRin[k] + rBG[k+1][jRin]**2.*ADeltaRin[k+1])/2.
        btA[k] = -1.*AThetaRin[k]

    btA[Nt+1] = (rBG[0][jRin]**2.*ADeltaRin[0]+rBG[1][jRin]**2.*ADeltaRin[1])/2.
    btA[2*(Nt+1)-1] = rBG[0][jRin]**2.*(7.*ADeltaini+2.*AThetaini)/18.
    btA[0] = -3.*rBG[0][jRin]**2.*(ADeltaini + 2.*AThetaini)/18.

    MattA = np.zeros((2*(Nt+1),2*(Nt+1)))


    # initial conditions for the dot{r2} constraint

    MattA[0][0] = -L1[1]/2./tau
    MattA[0][1] = (L1[2]-L1[0])/tau/2.
    MattA[0][2] = L1[1]/2./tau

    # final condition for r2

    MattA[Nt][Nt] = 1.

    # initial condition for mu2

    MattA[2*(Nt+1)-1][Nt+1] = 1.

    # the rest of the matrix

    for k in range(1,Nt):
        MattA[k][k-1] = 1./tau/tau - 1./tau/4.*alpha[k]
        MattA[k][k] = -2./tau/tau -0.5*Omf[k] + P1[k]
        MattA[k][k+1] = 1./tau/tau + 1./tau/4.*alpha[k]
        MattA[k][(Nt+1)+k] = 1.5/rBG[k][jRin]**2.*Omf[k]
    for k in range(Nt):
        MattA[(Nt+1)+k][k] = (L1[k+1]-L1[k])/tau - L1[k]/tau
        MattA[(Nt+1)+k][k+1] = L1[k]/tau
        MattA[(Nt+1)+k][(Nt+1)+k] = -1./tau
        MattA[(Nt+1)+k][(Nt+1)+k+1] = 1./tau

    ztA = np.linalg.solve(MattA,btA)


    r2A = np.zeros(Nt+1)
    mu2A = np.zeros(Nt+1)

    for k in range(Nt+1):
        r2A[k] = ztA[k]
        mu2A[k] =  ztA[k+Nt+1]

    A2tab[dWindex] = mu2A[Nt]*3./R**3.

    if log_mode>0:
        dlnk = (log(ktab[Nk])-log(ktab[0]))/Nk
        A1 = 0.
        for kindex1 in range(Nk+1):
            for kindex2 in range(Nk+1):
                A1 = A1 + 2.*lambdaStar*sqrt(Plin(ktab[kindex1]))*sqrt(Plin(ktab[kindex2]))*dlnk*exp(1.5*log(ktab[kindex1]))*exp(1.5*log(ktab[kindex2]))*Minv[kindex1][kindex2]*Btab[kindex1]*Btab[kindex2]*9./R**6./(8.*np.pi**3.)

        for kindex1 in range(Nk+1):
            a1[kindex1] = lambdaStar*dlnk*sqrt(Plin(ktab[kindex1]))/8./np.pi**3.*exp(1.5*log(ktab[kindex1]))/ktab[kindex1]
            b1[kindex1] = sqrt(Plin(ktab[kindex1]))*(2.*Btab[kindex1]*3./R**3.+A1/ktab[kindex1])*exp(1.5*log(ktab[kindex1]))
            b2[kindex1] = sqrt(Plin(ktab[kindex1]))*(2.*Btab[kindex1]*3./R**3.+A2tab[dWindex]/ktab[kindex1])*exp(1.5*log(ktab[kindex1]))


    else:
        dk = (kmax - kmin)/Nk
        A1 = 0.
        for kindex1 in range(Nk+1):
            for kindex2 in range(Nk+1):
                A1 = A1 + 2.*lambdaStar*sqrt(Plin(ktab[kindex1]))*sqrt(Plin(ktab[kindex2]))*dk*ktab[kindex1]*ktab[kindex2]*Minv[kindex1][kindex2]*Btab[kindex1]*Btab[kindex2]*9./R**6./(8.*np.pi**3.)
        for kindex1 in range(Nk+1):
            a1[kindex1] = lambdaStar*dk*sqrt(Plin(ktab[kindex1]))/8./np.pi**3.
            b1[kindex1] = sqrt(Plin(ktab[kindex1]))*(2.*Btab[kindex1]*3./R**3.+A1/ktab[kindex1])*ktab[kindex1]
            b2[kindex1] = sqrt(Plin(ktab[kindex1]))*(2.*Btab[kindex1]*3./R**3.+A2tab[dWindex]/ktab[kindex1])*ktab[kindex1]            

    A1tab[dWindex] = A1
    DIRmatrix = np.zeros((Nk+1,Nk+1))
    DIRmatrixver2 = np.zeros((Nk+1,Nk+1))
    for kindex1 in range(Nk+1):
        for kindex2 in range(Nk+1):
            DIRmatrix[kindex1][kindex2] = a1[kindex1]*b1[kindex2] + b1[kindex1]*a1[kindex2]
            DIRmatrixver2[kindex1][kindex2] = a1[kindex1]*b2[kindex2] + b2[kindex1]*a1[kindex2]
#        DIRmatrix[kindex1][kindex1] = DIRmatrix[kindex1][kindex1] + 1.

    DIRmatrix2 = np.dot(Minv,DIRmatrix)
    DIRmatrix2ver2 = np.dot(Minv,DIRmatrixver2)
    for kindex1 in range(Nk+1):
        DIRmatrix2[kindex1][kindex1] = DIRmatrix2[kindex1][kindex1] + 1.
        DIRmatrix2ver2[kindex1][kindex1] = DIRmatrix2ver2[kindex1][kindex1] + 1.
    detDIR[dWindex] = np.linalg.det(DIRmatrix2)
    detDIR2[dWindex] = np.linalg.det(DIRmatrix2ver2)

    Preftab = [ [0 for x in range(9)] for y in range(1)];
    Preftab[0][0] = dWtab[dWindex]
    Preftab[0][1] = detDIR[dWindex]
    Preftab[0][2] = detDIR[dWindex]*detQhat[dWindex]
    Preftab[0][3] = detDIR2[dWindex]
    Preftab[0][4] = detDIR2[dWindex]*detQhat[dWindex] 
    Preftab[0][5] = detQhat[dWindex]
    Preftab[0][6] = exp(traceQhat[dWindex])
    Preftab[0][7] = A1tab[dWindex]
    Preftab[0][8] = A2tab[dWindex]
    np.savetxt('Dipole_determinant_IR_Nk'+str(Nk)+'_dW'+str(deltaW)+'_.dat', Preftab)


#Preftab = [ [0 for x in range(9)] for y in range(len(dWtab))];
#for dWindex in range(len(dWtab)):
#    Preftab[dWindex][0] = dWtab[dWindex]
#    Preftab[dWindex][1] = detDIR[dWindex]
#    Preftab[dWindex][2] = detDIR[dWindex]*detQhat[dWindex]
#    Preftab[dWindex][3] = detDIR2[dWindex]
#    Preftab[dWindex][4] = detDIR2[dWindex]*detQhat[dWindex]
#    Preftab[dWindex][5] = detQhat[dWindex]
#    Preftab[dWindex][6] = exp(traceQhat[dWindex])
#    Preftab[dWindex][7] = A1tab[dWindex]
#    Preftab[dWindex][8] = A2tab[dWindex]
#np.savetxt('Dipole_determinant_all_Nk'+str(Nk)+'_.dat', Preftab)

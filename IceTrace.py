import os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si
import scipy.optimize as op
from scipy.optimize.minpack import curve_fit


C = 299792458.0
PI = np.pi

def nFromRho(rho):
    #rho is density as fraction of liquid water density
    n = (rho/0.9167)*0.78 + 1
    return n

def GetN(z,c,b):
    #posative z is up, so z should be negative
    a = 1.78 #index of solid ice
    #b = 1.222 #index at top of firn
    #c = 34.7 #exponential factor
    return np.exp(z/c)*(b-a) + a

def GetIndexFromDensity(zMeas,densityMeas,order=1):
    #returns interpolated function of order which is valid over the range of zMeas
    nMeas=nFromRho(densityMeas)
    nFit = si.interp1d(zMeas, nMeas, kind=order)
    return nFit

def nAllDepth(zMeas, nMeas, order=1):
    #takes measured values of n, interpolates between measured values
    #returns a function that is valid for any value of z
    popt,pcov = curve_fit(GetN,zMeas,nMeas)
    c=popt[0]
    b=popt[1]
    if order != 0:
        fitmin = zMeas.min()
        fitmax = zMeas.max()
        nFit = si.interp1d(zMeas, nMeas, kind=order)
        def nOut(z):
            if z>0:
                return 1.0
            elif (z<fitmin) or (z>fitmax):
                return GetN(z,c,b)
            else:
                return nFit(z)
    else:
        def nOut(z):
            if z>0:
                return 1.0
            else :
                return GetN(z,c,b)
    return nOut

def Refract(n1,n2,Sin1):
    Sin2=n1*Sin1/n2
    return Sin2

def RefractVector(n1,n2,v):
    # v must be a unit vectors
    v2=np.zeros(2)
    Sin1=v[0]
    Sin2=n1*Sin1/n2
    if np.abs(Sin2)>=1:
        v2[0]=v[0]
        v2[1]=-v[1]
    else:
        v2[0]=Sin2
        v2[1]=np.sign(v[1])*np.sqrt(1-Sin2*Sin2)
    return v2

def Getdr(n1,n2,Sin1,rStep,vert,Wiggle=False,WigSig=1.0):
    Sin2=Refract(n1,n2,Sin1)
    if np.abs(Sin2)>=1:
        vert*=-1
        if Wiggle:
            Sin2,vert=WiggleRayGaussian(Sin1,WigSig,vert)
        else:
            Sin2=Sin1
        Cos2=np.sqrt(1-Sin2*Sin2)
        dr=[Sin2*rStep,Cos2*rStep*vert]
    else:
        Cos2=np.sqrt(1-Sin2*Sin2)
        dr=[Sin2*rStep,Cos2*rStep*vert]
    return dr, vert

def WiggleVectorGaussian(v,sigma):
    #Sigma should be in degrees
    v2=np.zeros(2)
    sigma=np.radians(sigma)
    theta=np.arctan2(v[1],v[0])
    theta+=np.random.normal(scale=sigma)
    v2[0]=np.cos(theta)
    v2[1]=np.sin(theta)
    return v2

def WiggleRayGaussian(Sin1,Sigma,vert):
    #sigma should be in degrees
    Sigma=np.radians(Sigma)
    theta=np.arcsin(Sin1)
    delta=np.random.normal(scale=Sigma)
    theta+=delta
    Sin2=np.sin(theta)
    if vert>0 and theta<np.pi/2.0:
        vert*=-1
    if vert<0 and theta>np.pi/2.0:
        vert*=-1
    return Sin2, vert

def ShadowRange(NvsZ,depth,rStep,ybot,ytop,Wiggle=False,WigSig=1.0):
    ntop=NvsZ(ytop)
    n0=NvsZ(depth)

    SinC=ntop/n0

    Vx0=SinC
    Vy0=np.sqrt(1-SinC*SinC)
    v0=np.array([Vx0,Vy0])

    v=v0
    r=np.array([0,depth])

    while v[1]>0:
        r,v = ItterateRay(NvsZ,r,v,rStep,ybot,ytop)

    hdist=r[0]
    return hdist

def ItterateRay(NvsZ,r0,v0,rStep,ybot,ytop,Wiggle=False,WigSig=1.0):
    v=v0
    r=np.zeros(2)
    n2 = NvsZ(r0[1]+(v0[1]*rStep/2.0))
    n1 = NvsZ(r0[1]-(v0[1]*rStep/2.0))
    v = RefractVector(n1,n2,v0)
    flipped = np.sign(v[1]) != np.sign(v0[1])
    if Wiggle and flipped: v=WiggleVectorGaussian(v,WigSig)

    if (r0[1]+v[1])<ybot:
        flipped=True
        v[1]*=-1

    r[0]=r0[0]+v[0]*rStep
    r[1]=r0[1]+v[1]*rStep

    return r,v

def PerpendicularDistance(A,B,C): #A and B are 2-D points that define a line segment. C is a point of interest
    #set origin to A
    B = B - A
    C = C - A
    #Normal Vector perpendicular to AB
    n=np.array([B[1],-B[0]])/np.linalg.norm(B)
    #projection of AC into n gives the perpendicular distance
    dist=np.absolute(np.dot(C,n))
    return dist

def FindDistOfClosestApproach(NvsZ,r1,r2,v1,rStep,ybot,ytop): # assumes rightward going ray
    r=r1
    v=v1
    rpast=r1
    while r[0] < r2[0]:
        rpast=r
        r,v = ItterateRay(NvsZ,r,v,rStep,ybot,ytop)
    rmiss=PerpendicularDistance(rpast,r,r2)
    return rmiss

def FindYmiss(NvsZ,r1,r2,v1,rStep,ybot,ytop): # assumes rightward going ray
    r=r1
    v=v1
    rpast=r1
    while r[0] < r2[0]:
        rpast=r
        r,v = ItterateRay(NvsZ,r,v,rStep,ybot,ytop)
    ymiss=Interpolate2Point(rpast,r,r2[0]) - r2[1]
    return ymiss

def Interpolate2Point(r1,r2,x):
    y=r1[1]+(r2[1]-r1[1])/(r2[0]-r1[0])*(x-r1[0])
    return y

def FollowRaySolution(NvsZ,r1,r2,v1,rStep,ybot,ytop):
    r=r1
    v=v1
    xspace=[]
    yspace=[]
    time=0
    xspace.append(r[0])
    yspace.append(r[1])
    while r[0] < r2[0]:
        r,v = ItterateRay(NvsZ,r,v,rStep,ybot,ytop)
        xspace.append(r[0])
        yspace.append(r[1])
        time+=rStep*NvsZ(r[1])/c
    xspace=np.array(xspace)
    yspace=np.array(yspace)
    return xspace, yspace, time

#def FindRefractedPathsOpt(NvsZ,r1,r2,rStep,ybot,ytop,eps=0.01,Wiggle=False,WigSig=1.0):
#    PathsX=[]
#    PathsY=[]
#    Times=[]
#    direction=np.sign(r2[0]-r1[0])
#    if direction==-1:
#        r1[0]*=-1
#        r2[0]*=-1
#    maxTheta=np.arctan(np.absolute(2*(ytop-ybot)/(r2[0]-r1[0])))
#    minTheta=-maxTheta
#    print("minTheta = {}, maxTheta = {}".format(minTheta,maxTheta))
#    def PositionError(Theta):
#        v=np.array([np.cos(Theta),np.sin(Theta)])
#        error = FindDistOfClosestApproach(NvsZ,r1,r2,v,rStep,ybot,ytop)
#        return error
#    MinAngles=[]
#    MinDists=[]
#    thetaWalk=minTheta
#    while thetaWalk<maxTheta:
#        toptmin, foptmin, d  = op.fmin_l_bfgs_b(PositionError,thetaWalk,bounds=(minTheta,maxTheta),approx_grad=True)
#        print("Min at theta = {}, dist = {}".format(toptmin,foptmin))
#        MinAngles.append(toptmin)
#        MinDists.append(foptmin)
#        thetaWalk = toptmin + eps
#        toptmax, foptmax, d  = op.fmin_l_bfgs_b(-PositionError,thetaWalk,bounds=(minTheta,maxTheta),approx_grad=True)
#        print("Max at theta = {}, dist = {}".format(toptmax,foptmax))
#        thetaWalk = toptmax + eps
#    for i in range(0,len(MinDists)):
#        if MinDists[i]<rStep:
#            thetatmp = MinAngles[i]
#            v=np.array([np.cos(thetatmp),np.sin(thetatmp)])
#            xspacetmp, yspacetmp, time = FollowRaySolution(NvsZ,r1,r2,v,rStep,ybot,ytop)
#            if direction == -1:
#                xspacetmp *= -1
#            PathsX.append(xspacetmp)
#            PathsY.append(yspacetmp)
#            Times.append(TravelTime)
#
#    return PathsX, PathsY, Times

def FindRefractedPaths(NvsZ,r1,r2,rStep,ybot,ytop,Wiggle=False,WigSig=1.0):
    PathsX=[]
    PathsY=[]
    Times=[]
#    MaxRange=ShadowRange(NvsZ,r1[1],rStep,ybot,ytop)+ShadowRange(NvsZ,r2[1],rStep,ybot,ytop)
#    if MaxRange < np.absolute(r2[0]-r1[0]):
#        print("Horizontal Distance {} is greater than Max Range of {}".format(np.absolute(r2[0]-r1[0]),MaxRange))
#        return PathsX, PathsY, Times
    direction=np.sign(r2[0]-r1[0])
    if direction==-1:
        r1[0]*=-1
        r2[0]*=-1
    nTestAngles=200
    maxTheta=np.arctan(np.absolute(2*(ytop-ybot)/(r2[0]-r1[0])))
    minTheta=-maxTheta
    testAngles=np.linspace(minTheta,maxTheta,nTestAngles,endpoint=True)
    overAngles=[]
    overYmiss=[]
    underAngles=[]
    underYmiss=[]
    ymiss=0
    ymisslast=0
    anglast=0
    for ang in testAngles:
        v=np.array([np.cos(ang),np.sin(ang)])
        r=r1
        while r[0] < r2[0]:
            r,v = ItterateRay(NvsZ,r,v,rStep,ybot,ytop)
        ymiss=r[1]-r2[1]
        if ymiss*ymisslast < 0:
            overAngles.append(ang)
            overYmiss.append(ymiss)
            underAngles.append(anglast)
            underYmiss.append(ymisslast)
        ymisslast=ymiss
        anglast=ang
    for i in range(0,len(overAngles)):
        Theta1=underAngles[i]
        ymiss1=underYmiss[i]
        Theta2=overAngles[i]
        ymiss2=overYmiss[i]
        ymiss=rStep*100
        xspacetmp=[]
        yspacetmp=[]
        TravelTime=0
        while np.absolute(ymiss) > rStep:
            if np.absolute(Theta1-Theta2) < 1e-3:
                break
            xspacetmp=[]
            yspacetmp=[]
            Theta = (Theta1+Theta2)/2
            r=r1
            v=np.array([np.cos(Theta),np.sin(Theta)])
            xspacetmp.append(r[0])
            yspacetmp.append(r[1])
            TravelTime=0
            while r[0] < r2[0]:
                r,v = ItterateRay(NvsZ,r,v,rStep,ybot,ytop)
                TravelTime+=rStep*NvsZ(r[1])/C
                xspacetmp.append(r[0])
                yspacetmp.append(r[1])
            ymiss=r[1]-r2[1]
            if ymiss*ymiss1 < 0:
                Theta2 = Theta
            else:
                Theta1 = Theta
        if np.absolute(ymiss)<rStep:
            xspacetmp=np.array(xspacetmp)
            yspacetmp=np.array(yspacetmp)
            if direction == -1:
                xspacetmp *= -1
            PathsX.append(xspacetmp)
            PathsY.append(yspacetmp)
            Times.append(TravelTime)

    return PathsX, PathsY, Times

def hasFlipped(NvsZ,r0,v0,rStep,):
    v=v0
    r=np.zeros(2)
    n2 = NvsZ(r0[1]+(v0[1]*rStep/2.0))
    n1 = NvsZ(r0[1]-(v0[1]*rStep/2.0))
    v = RefractVector(n1,n2,v0)
    return np.sign(v[1]) != np.sign(v0[1])

def FindRayToSurface(NvsZ,r0,dist,rStep,ybot,ytop,Wiggle=False,WigSig=1.0,ReturnFullPath=False):

    thetahi = 0.0
    thetanew= PI/4.
    thetalo = PI/2.
    deltaTheta = PI/4.
    dThetCut = PI/180.0/60.0
    i=0
    while deltaTheta > dThetCut:
        i+=1
        flipped=False
        v0=[np.sin(thetanew),np.cos(thetanew)]
        xspace = [r0[0]]
        yspace = [r0[1]]
        TravelTime = 0
        v=v0
        r=r0
        while r[0]<dist and flipped==False:
            flipped = hasFlipped(NvsZ,r,v,rStep)
            r,v = ItterateRay(NvsZ,r,v,rStep,ybot,ytop,Wiggle,WigSig)
            xspace.append(r[0])
            yspace.append(r[1])
            TravelTime+=rStep*NvsZ(r[1])/C

        if flipped or r[1]>ytop:
            thetahi = thetanew
        else:
            thetalo = thetanew
        thetanew = (thetalo + thetahi)/2
        deltaTheta = thetalo - thetanew
        print 'try {}, flipped {}, deltaTheta {}, v {}'.format(i,flipped,deltaTheta,v)
    if flipped:
        v[1]*=-1
    if ReturnFullPath:
        return xspace, yspace, TravelTime
    else:
        return v0,v

def TraceRay(NvsZ,r0,v0,rDist,rStep,ybot,ytop,Wiggle=False,WigSig=1.0):
    #NvsZ should be a function that returns index of refraction for any value between ybot and ytop
    #r0 should be a 2D, normalized  array (vectors), with v0 being normalized
    #Theta should be in degrees, cw off y axis (-180 t0 180)

    nSteps = int(rDist/rStep)
    xspace = np.zeros(nSteps)
    yspace = np.zeros(nSteps)

    xspace[0]=r0[0]
    yspace[0]=r0[1]
    r=np.zeros(2)
    v=v0

    for i in range(0,nSteps-1):
        r[0]=xspace[i]
        r[1]=yspace[i]
        r,v = ItterateRay(NvsZ,r,v,rStep,ybot,ytop,Wiggle,WigSig)
        xspace[i+1]=r[0]
        yspace[i+1]=r[1]

    return xspace,yspace

def TraceRayToX(NvsZ,r0,v0,xf,rStep,ybot,ytop,Wiggle=False,WigSig=1.0):
    #Ray Travel must be in +x direction
    #NvsZ should be a function that returns index of refraction for any value between ybot and ytop
    #r0 should be a 2D, normalized  array (vectors), with v0 being normalized
    #Theta should be in degrees, cw off y axis (-180 t0 180)

    xspace = np.zeros(nSteps)
    yspace = np.zeros(nSteps)
    TravelTime = 0

    xspace.append(r0[0])
    yspace.append(r0[1])
    r=r0
    v=v0

    while r[0] < xf :
        r,v = ItterateRay(NvsZ,r,v,rStep,ybot,ytop,Wiggle,WigSig)
        TravelTime+=rStep*NvsZ(r[1])/C
        xspace.append(r[0])
        yspace.append(r[1])


    return xspace, yspace, TravelTime

if __name__ == "__main__":
    from timeit import default_timer as timer

    print('Initializing...')
    startTime=timer()

    rStep = 0.05
    rDist = 1200.0

    ybot=-575.0
    ytop=0

    #initial position
    r0=[0,-20.0]
    v0 = [0,0]
    #launch angles to trace
    ThetaSpace=np.linspace(4,175,10)
    ThetaSpace=np.radians(ThetaSpace)



    infn = "DensityData/SnowDensity_MooresBay.csv"
    DensityArray=np.genfromtxt(infn,delimiter=',',skip_header=1,names=True)
    densityMeas=DensityArray['density']
    zMeas=-DensityArray['depth']

    #Calculate index of refraction from density
    nMeas=nFromRho(densityMeas)

    #Create best Fit exponential function
    N1=nAllDepth(zMeas,nMeas,0)


    fig, ax = plt.subplots()

    for i, Theta in enumerate(ThetaSpace):
        print('Tracing ray {} of {}...'.format(i+1,len(ThetaSpace)))
        t1=timer()
        v0[0]=np.sin(Theta)
        v0[1]=np.cos(Theta)
        xspace, yspace = TraceRay(N1,r0,v0,rDist,rStep,ybot,ytop)
        t2=timer()
        print('in {}s'.format(t2-t1))
        ax.plot(xspace,yspace)

    ax.axhline(y=0,c='r',ls=':')
    ax.axhline(y=ybot,c='k')
    ax.scatter(*r0,marker='x',s=150,c='k')
    ax.set_xlabel("distance (m)")
    ax.set_ylabel("height (m)")
    ax.set_ylim(top=50)
    endTime = timer()
    print('All Done in {}s'.format(endTime-startTime))
    plt.show()

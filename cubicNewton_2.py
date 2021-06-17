# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 15:09:06 2020

@author: Servet
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fftpack import ifft
from scipy.interpolate import CubicSpline


# Parallel function for evaluating complex polynomial
# def poly(z,aVec):
#     result = 1j*np.zeros(len(z))
#     nVec = np.arange(len(aVec))
#     for ii in range(len(z)):
#         zVec = z[ii]**(nVec)
#         result[ii] = np.sum(aVec*zVec)
#     return result

# # Parallel function for evaluating complex derivative
# def dpoly(z,aVec):
#     result = 1j*np.zeros(len(z))
#     nVec = np.arange(len(aVec))
#     d_aVec = nVec*aVec
#     for ii in range(len(z)):
#         zVec = z[ii]**(nVec)
#         result[ii] = np.sum(d_aVec*zVec)
#     return result

# # define the function on the right-hand side of first eq. in (18)
# # Use IVP on this function
# def dphi(r,phi,aVec):
#     z = r*np.exp(1j*phi)
#     dTmp = dpoly(z,aVec)*z
#     return -np.imag(dTmp) /(r* np.real(dTmp))

# Parallel function for evaluating complex derivative
def poly4(z,aVec):
    N = len(aVec)
    result = 1j*np.zeros((4,len(z)))
    nVec = np.arange(N)
    aMx = 1j*np.zeros((4,N))
    aMx[0,:] = aVec
    aMx[1,:] = nVec*aVec
    aMx[2,:-1] = nVec[1:]*aMx[1,1:]
    aMx[3,:-1] = nVec[1:]*aMx[2,1:]
    for ii in range(len(z)):
        zMx = np.ones((4,1))* z[ii]**nVec
        result[:,ii] = np.sum(aMx*zMx, axis=1)
    return result

def polyN(z,aVec,m):
    N = len(aVec)
    result = 1j*np.zeros((m+1,len(z)))
    nVec = np.arange(N)
    aMx = 1j*np.zeros((m+1,N))
    aMx[0,:] = aVec
    aMx[1,:] = nVec*aVec
    for ii in range(2,m+1):
        aMx[ii,:-1] = nVec[1:]*aMx[ii-1,1:]
    for ii in range(len(z)):
        zMx = np.ones((m+1,1))* z[ii]**nVec
        result[:,ii] = np.sum(aMx*zMx, axis=1)
    return result


def   cubicRootsFn(coefMx):
    aa = coefMx[0,:]
    bb = coefMx[1,:]
    cc = coefMx[2,:]
    dd = coefMx[3,:]
    
    # Solve for inverse roots to avoid division by zero. 
    # Compute p,q,r in the cubic formula
    pp = -bb/(3*aa)
    qq = pp**3 + (bb*cc-3*aa*dd)/(6*aa**2)
    rr = cc/(3*aa)
    
    # Compute discriminant
    disc = np.sqrt(0j+ qq**2 + (rr - pp**2)**3)
    
    # Cubic has only one real root if the discriminant is real.
    # Logical vector to identify single roots.
    oneRoot = np.isreal(disc)
    # Logical vector to identify multiple roots.
    multiRoot = np.logical_not(oneRoot)
    
    
    # Compute all single roots.
    # First factor in cubic solution.
    fac1 = (qq + disc)*oneRoot
    fac2 = (qq - disc)*oneRoot
    fac1 = np.sign(fac1)*np.abs(fac1)**(1/3)
    fac2 = np.sign(fac2)*np.abs(fac2)**(1/3)
    recipRoots =  oneRoot*(fac1 + fac2 + pp)
    # For imag discriminant, compute 3 roots, and take the largest
    fac1multi = (multiRoot*(qq+disc))**(1/3)
    # Compute three real roots for each point.
    tmpRoots = pp*multiRoot + 2*np.array([np.real(fac1multi), \
                                np.real(fac1multi*np.exp(2*np.pi*1j/3)), \
                                np.real(fac1multi*np.exp(4*np.pi*1j/3))])
    
    maxVec = np.amax(tmpRoots, axis=0)
    minVec = np.amin(tmpRoots, axis=0)
    # Identify the reciprocal root with the largest absolute value.
    multiRecipRoots = maxVec + (minVec-maxVec)*(maxVec+minVec<0)
    # Take reciprocals to obtain roots.
    recipRoots = recipRoots + multiRecipRoots

    return 1/recipRoots


def cubicNewtonFn(aVec,nOrder):
    aVec = aVec/aVec[-1]
    # Initial conditions 
    N = len(aVec)-1 #N is the degree of the polynomial
    # Insert the code here to find r from eqn, 1 in overleaf
    # Define a,b,c in the quadratic formula: a=1
    nb = np.abs(aVec[-2]/aVec[-1])
    nc = 1/np.abs(aVec[-1])*np.sum(np.abs(aVec[0:-3]))
    r = (nb+np.sqrt(nb**2+4*nc))/ 1.9 # r should be slightly bigger than the solution
    r = max(1,r)
    
    arVec = r**(np.arange(N+1))*aVec
    
    # Create matrix whose rows are, function and first k derivatives with respect to theta
    polyMx = np.zeros((nOrder+1,2*N))+0j
    # Fills initial matrix
    polyMx[0,:N+1] =arVec
    dThetaVec = np.arange(N+1)*1j
    for ii in range(nOrder):
        polyMx[ii+1,:N+1] = dThetaVec*polyMx[ii,:N+1]
    
    
    # Evaluate first nOrder derivatives of the function at midpts using IFFT
    pMx = np.imag(ifft(polyMx)*2*N)
    thetaVec = (np.arange(2*N)*np.pi)/N
    
    # Find cube roots
    # create matrix of cubic coefficients.
    # If const coeff  = 0, then 0 is a root --no need to continue
    # Identify nonzero roots
    nonZeroRoots = pMx[0,:]!=0;
    
    # Remove columns for zero roots
    pMx = pMx[:,nonZeroRoots]
    
    # Define cubic coefficients
    tmpMx = np.diag(np.array([1,1,1./2.,1./6.]))
    coefMx = tmpMx @ pMx[:4,:]
    
    # Find cubic roots
    h = cubicRootsFn(coefMx)
       
    # @@ Add fourth and possibly fifth order correction terms.
    # Define new cubic coefficient matrix
    coefMx =np.zeros_like(coefMx)
    # Define matrix with rows = powers of h
    hMx = np.array([h,h**2/2.,h**3/6.,h**4/24.,h**5/120.])
    # Construct new cubic coefficent matrix row by row
    coefMx[0,:] = pMx[4,:]*hMx[3,:]+ pMx[5,:]*hMx[4,:]
    coefMx[1,:] = pMx[1,:]+pMx[2,:]*hMx[0,:]+pMx[3,:]*hMx[1,:]+pMx[4,:]*hMx[2,:]+pMx[5,:]*hMx[3,:]
    coefMx[2,:] = pMx[2,:]+pMx[3,:]*hMx[0,:]+pMx[4,:]*hMx[1,:]+pMx[5,:]*hMx[2,:]
    coefMx[3,:] = pMx[3,:]+pMx[4,:]*hMx[0,:]+pMx[5,:]*hMx[1,:]
    
    # Find cubic roots
    k = cubicRootsFn(coefMx)

    # Add corrections to existing theta values.
    thetaVec[nonZeroRoots] = thetaVec[nonZeroRoots] + h +k
   
      
    return r, thetaVec

#aVec = np.array([-2j,-1,2j,1])/2j
aVec = np.array([ -9.16212853 -8.69018418j,   4.8773068 +13.1439682j ,
       -12.46657218 +6.22541424j,  -3.30874102 -0.98232887j,
         9.7621899  -0.10630278j,   4.86313499 +7.66433247j,
         3.182697  -13.51425322j,  12.76965285 -4.77139123j,
        10.81404766 +9.18917836j,   1.         +0.j        ])
r, thetaVec = cubicNewtonFn(aVec,5)

zVec = r*np.exp(1j*thetaVec)
    
nderiv=5
hMx = polyN(zVec, aVec,nderiv)
#print(output)
   
# This is the beginning of the differential equation section.
h_i = np.imag(hMx[1,:])
h_r = np.real(hMx[1,:])
d_theta = (-1/r)*(h_i/h_r)
d_z = zVec/r+(1j*zVec*d_theta)
d_h = hMx[2,:]*d_z
d_hi = np.imag(d_h)
d_hr = np.real(d_h)
d2_theta =-(d_theta  + d_hi/h_r - d_hr*h_i/h_r**2)/r
d2_z = -zVec/r**2+d_z/r+1j*d_z*d_theta+1j*zVec*d2_theta
d2_h = hMx[3,:]*d_z**2+hMx[2,:]*d2_z
d2_hi = np.imag(d2_h)
d2_hr = np.real(d2_h) 
d3_theta = -(2*d2_theta + d2_hi/h_r - 2*d_hi*d_hr/h_r**2 \
             + 2*h_i*d_hr**2/h_r**3 - h_i*d2_hr/h_r**2)/r
d_x = h_r/r-h_i*d_theta
d2_x = d_hr/r-h_r/r**2-d_hi*d_theta - h_i*d2_theta
d3_x = d2_hr/r - 2*d_hr/r**2 + 2*h_r/r**3 - d2_hi*d_theta \
       - 2*d_hi*d2_theta - h_i*d3_theta
    
    







  
# Find cube roots
# create matrix of cubic coefficients.
# If const coeff  = 0, then 0 is a root --no need to continue
# Identify nonzero roots
nonZeroRoots = hMx[0,:]!=0;

# Reduce matrix to only compute nonzero roots. 
hMx = hMx[:,nonZeroRoots]
          
aa = np.real(hMx[0,:])
bb = d_x
cc = d2_x/2
dd = d3_x/6

# Compute p,q,r in the cubic formula
pp = -bb/(3*aa)
qq = pp**3 + (bb*cc-3*aa*dd)/(6*aa**2)
rr = cc/(3*aa)

# Compute discriminant
disc = np.sqrt(0j+ qq**2 + (rr - pp**2)**3)
# Cubic has only one real root if the discriminant is real.
# Logical vector to identify single roots.
oneRoot = np.isreal(disc)
# Logical vector to identify multiple roots.
multiRoot = np.logical_not(oneRoot)

# Compute all single roots.
# First factor in cubic solution.
fac1 = (qq + disc)*oneRoot
fac2 = (qq - disc)*oneRoot
fac1 = np.sign(fac1)*np.abs(fac1)**(1/3)
fac2 = np.sign(fac2)*np.abs(fac2)**(1/3)
recipRoots =  oneRoot*(fac1 + fac2 + pp)
# For imag discriminant, compute 3 roots, and take the largest
fac1multi = (multiRoot*(qq+disc))**(1/3)
# Compute three real roots for each point.
tmpRoots = pp*multiRoot + 2*np.array([np.real(fac1multi), \
                            np.real(fac1multi*np.exp(2*np.pi*1j/3)), \
                            np.real(fac1multi*np.exp(4*np.pi*1j/3))])
# Find the smallest reciprocal root:
recipRoots += np.amin(tmpRoots, axis=0)

# Take reciprocals to obtain roots.
rVec = 0.*zVec+r
rVec[nonZeroRoots] = rVec[nonZeroRoots]+1/recipRoots
 
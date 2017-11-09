"""
A reference implementation of local density-fitted MP2 from a RHF reference.

References: 
Algorithm modified from Rob Parrish's most excellent Psi4 plugin example
Bottom of the page: http://www.psicode.org/developers.php
"""

__authors__   = "Brandon W. Bakr"
__credits__   = ["Brandon W. Bakr"]

__copyright__ = "(c) 2014-2017, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2017-05-23"

import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4
import sys

# Set memory & output
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)

mol = psi4.geometry(""" 
C    1.39410    0.00000   0.00000
C    0.69705   -1.20732   0.00000
C   -0.69705   -1.20732   0.00000
C   -1.39410    0.00000   0.00000
C   -0.69705    1.20732   0.00000
C    0.69705    1.20732   0.00000
H    2.47618    0.00000   0.00000
H    1.23809   -2.14444   0.00000
H   -1.23809   -2.14444   0.00000
H   -2.47618    0.00000   0.00000
H   -1.23809    2.14444   0.00000
H    1.23809    2.14444   0.00000
symmetry c1
""")

# Basis used in mp2 density fitting
psi4.set_options({'basis': 'aug-cc-pVDZ',
                  'df_basis_scf': 'aug-cc-pvdz-ri'})

check_energy = False

print('\nStarting RHF...')
t = time.time()
RHF_E, wfn = psi4.energy('SCF', return_wfn=True)
print('...RHF finished in %.3f seconds:   %16.10f' % (time.time() - t, RHF_E))

### BEGIN LOCAL MP2 ###



# Grab data from Wavfunction clas
ndocc = wfn.nalpha()
nbf = wfn.nso()
nvirt = nbf - ndocc

# Split eigenvectors and eigenvalues into o and v
eps_occ = np.asarray(wfn.epsilon_a_subset("AO", "ACTIVE_OCC"))
eps_vir = np.asarray(wfn.epsilon_a_subset("AO", "ACTIVE_VIR"))

# Build DF tensors
print('\nBuilding DF ERI tensor Qov...')
t = time.time()
C = wfn.Ca()
aux = psi4.core.BasisSet.build(mol, "DF_BASIS_MP2", "", "RIFIT", "aug-cc-pvdz")
df = psi4.core.DFTensor(wfn.basisset(), aux, C, ndocc, nvirt) 
# Transformed MO DF tensor
Qov = np.asarray(df.Qov())
print('...Qov build in %.3f seconds with a shape of %s, %.3f GB.' \
% (time.time() - t, str(Qov.shape), np.prod(Qov.shape) * 8.e-9))


#Localize Orbitals
Ca_occ = np.asarray(wfn.Ca_subset("AO", "ACTIVE_OCC"))

nso = Ca_occ.shape[0]
nmo = Ca_occ.shape[1]
nA  = mol.natom() 

S = np.asarray(wfn.S())

L = Ca_occ
U = np.identity(nmo)

if nmo < 1: 
    print("Cannot localize THESE orbitals.")
    sys.exit()

LS = np.dot(S,L)

AtomStarts = []
atom = 0
primary = wfn.basisset()
for i in range(primary.nbf()):
    if (primary.function_to_center(i) == atom):
        AtomStarts.append(i)
        atom += 1
AtomStarts.append(primary.nbf())

metric = 0.0
for i in range(nmo):
    for A in range(nA):
        nm = AtomStarts[A+1] - AtomStarts[A]
        off = AtomStarts[A]
        PA = np.dot(LS[off:off+nm+1,i],L[off:off+nm+1,i])
        metric += PA * PA
old_metric = metric

Aii = 0 
Ajj = 0
Aij = 0
Ad = 0
Ao = 0 
Hd = 0
Ho = 0
theta = 0
cc = 0
ss = 0      
max_iter = 100
import random
random.seed(0)

for it in range(1,max_iter+1):
    order = [i for i in range(nmo)]
    order2 = []
    for i in range(nmo):    
        pivot = random.randint(0,nmo-1)
        i2 = order[pivot]
        order[pivot] = order[nmo - i - 1]
        order2.append(i2)
           
    for i2 in range(nmo-1):
        for j2 in range(i2+1, nmo):
            i = order2[i2]
            j = order2[j2]

            a = 0.0
            b = 0.0
            c = 0.0
            for A in range(nA):       
                nm = AtomStarts[A+1] - AtomStarts[A]
                off = AtomStarts[A]
                Aii = np.dot(LS[off:off+nm+1,i],L[off:off+nm+1,i])
                Ajj = np.dot(LS[off:off+nm+1,j],L[off:off+nm+1,j])
                Aij = 0.5 * np.dot(LS[off:off+nm+1,i],L[off:off+nm+1,j])
                Aij += 0.5 * np.dot(LS[off:off+nm+1,j],L[off:off+nm+1,i])
                Ad = (Aii - Ajj)
                Ao = 2.0 * Aij
                a += Ad * Ad
                b += Ao * Ao
                c += Ad * Ao

            Hd = a - b
            Ho = 2.0 * c
            theta = 0.5 * np.arctan2(Ho, Hd + np.sqrt(Hd * Hd + Ho * Ho))

            if abs(theta) < 1.0E-8:
                O0 = 0.0
                O1 = 0.0
                for A in range(nA):
                    nm = AtomStarts[A+1] - AtomStarts[A]
                    off = AtomStarts[A]
                    Aii = np.dot(LS[off:off+nm+1,i],L[off:off+nm+1,i])
                    Ajj = np.dot(LS[off:off+nm+1,j],L[off:off+nm+1,j])
                    Aij = 0.5 * np.dot(LS[off:off+nm+1,i],L[off:off+nm+1,j])
                    Aij += 0.5 * np.dot(LS[off:off+nm+1,j],L[off:off+nm+1,i])
                    O0 += Aij * Aij
                    O1 += 0.25 * (Ajj - Aii) * (Ajj - Aii)
                if O1 < O0:
                    theta = np.pi / 4.0
            cc = np.cos(theta)
            ss = np.sin(theta)

            #Rotate LS and L
            for r in range(LS.shape[0]):
                temp = cc * LS[r][i] + ss * LS[r][j]
                LS[r][j] = cc * LS[r][j] - ss * LS[r][i]
                LS[r][i] = temp
                temp = cc * L[r][i] + ss * L[r][j]
                L[r][j] = cc * L[r][j] - ss * L[r][i]
                L[r][i] = temp
            for r,row in enumerate(U):
                temp = cc * U[r][i] + ss * U[r][j] 
                U[r][j] = cc * U[r][j] - ss * U[r][i] 
                U[r][i] = temp 
            
    metric = 0.0
    for i in range(nmo):
        for A in range(nA):
            nm = AtomStarts[A+1] - AtomStarts[A]
            off = AtomStarts[A]
            PA = np.dot(LS[off:off+nm+1,i],L[off:off+nm+1,i])
            metric += PA * PA

    conv = abs(metric - old_metric) / abs(old_metric)
    old_metric = metric

    print( "    @PM %4d %24.16E %14.6E\n" % (it, metric, conv))

    if conv < 1.E-3:
        converged = True
        break

if converged: print( "    PM Localizer converged.\n\n")
else: print( "    PM Localizer failed.\n\n")

U = U.T


##local_orbitals = localize(basis, 
##
##
##
##
##
##
##
##
##print('\nComputing MP2 energy...')
##t = time.time()
##
### At this point we can trivially build the ovov MO tensor and compute MP2
### identically to that as MP2.dat. However, this means we have to build the
### 4-index ERI tensor in memory and results in little gains over conventional
### MP2
### MO = np.einsum('Qia,Qjb->iajb', Qov, Qov)
##
### A smarter algorithm, loop over occupied indices and exploit ERI symmetry
##
### This part of the denominator is identical for all i,j pairs
##vv_denom = - eps_vir.reshape(-1, 1) - eps_vir
##
##MP2corr_OS = 0.0
##MP2corr_SS = 0.0
##for i in range(ndocc):
##    eps_i = eps_occ[i]
##    i_Qv = Qov[:, i, :].copy()
##    for j in range(i, ndocc):
##
##        eps_j = eps_occ[j]
##        j_Qv = Qov[:, j, :]
##
##        # We can either use einsum here
###        tmp = np.einsum('Qa,Qb->ab', i_Qv, j_Qv)
##
##        # Or a dot product (DGEMM) for speed)
##        tmp = np.dot(i_Qv.T, j_Qv)
##
##        # Diagonal elements
##        if i == j:
##            div = 1.0 / (eps_i + eps_j + vv_denom)
##        # Off-diagonal elements
##        else:
##            div = 2.0 / (eps_i + eps_j + vv_denom)
##
##        # Opposite spin computation
##        MP2corr_OS += np.einsum('ab,ab,ab->', tmp, tmp, div)
##
##        # Notice the same-spin compnent has an "exchange" like term associated with it
##        MP2corr_SS += np.einsum('ab,ab,ab->', tmp - tmp.T, tmp, div)
##
##print('...finished computing MP2 energy in %.3f seconds.' % (time.time() - t))
##
##MP2corr_E = MP2corr_SS + MP2corr_OS
##MP2_E = RHF_E + MP2corr_E
##
### These are the canonical SCS MP2 coefficients, many others are available however
##SCS_MP2corr_E = MP2corr_SS / 3 + MP2corr_OS * (6. / 5)
##SCS_MP2_E = RHF_E + SCS_MP2corr_E
##
##print('\nMP2 SS correlation energy:         %16.10f' % MP2corr_SS)
##print('MP2 OS correlation energy:         %16.10f' % MP2corr_OS)
##
##print('\nMP2 correlation energy:            %16.10f' % MP2corr_E)
##print('MP2 total energy:                  %16.10f' % MP2_E)
##
##print('\nSCS-MP2 correlation energy:        %16.10f' % MP2corr_SS)
##print('SCS-MP2 total energy:              %16.10f' % SCS_MP2_E)
##
##if check_energy:
##    psi4.energy('MP2')
##    psi4.compare_values(psi4.core.get_variable('MP2 TOTAL ENERGY'), MP2_E, 6, 'MP2 Energy')
##    psi4.compare_values(psi4.core.get_variable('SCS-MP2 TOTAL ENERGY'), SCS_MP2_E, 6, 'SCS-MP2 Energy')
##

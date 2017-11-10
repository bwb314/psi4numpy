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

#Localize Orbitals
def localize(wfn, mol):
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
    return L, U

# Set memory & output
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)

mol = psi4.geometry(""" 
H
H 1 0.9
symmetry c1
""")
#C    1.39410    0.00000   0.00000
#C    0.69705   -1.20732   0.00000
#C   -0.69705   -1.20732   0.00000
#C   -1.39410    0.00000   0.00000
#C   -0.69705    1.20732   0.00000
#C    0.69705    1.20732   0.00000
#H    2.47618    0.00000   0.00000
#H    1.23809   -2.14444   0.00000
#H   -1.23809   -2.14444   0.00000
#H   -2.47618    0.00000   0.00000
#H   -1.23809    2.14444   0.00000
#H    1.23809    2.14444   0.00000
#symmetry c1

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

# Localize the orbitals
L, U = localize(wfn,mol)

# Determine orbital domains
def orbital_domains(L, wfn, mol):
    local_completeness = 0.02
    primary = wfn.basisset()
    nn = primary.nbf()
    nA = mol.natom()
    na = np.asarray(wfn.epsilon_a_subset("AO", "ACTIVE_OCC")).shape[0]
    

domains = orbital_domains(L, wfn, mol)



# Determine orbital domain pairs

# Make Projector

# Orthogonalize 

# Integrals

# Energy

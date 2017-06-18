# A SAPT helper object
#
# Created by: Daniel G. A. Smith
# Date: 12/1/14
# License: GPL v3.0
#

import numpy as np
import time
import psi4


def integral_transformer(I, C1, C2, C3, C4):

    nbf = np.asarray(C1).shape[0]
    if np.asarray(C1).shape[1] < np.asarray(C4.shape[1]):
        v = np.dot(np.asarray(C1).T, np.asarray(I).reshape(nbf, -1))
        v = np.dot(v.reshape(-1, nbf), C4)
    else:
        v = np.dot(np.asarray(I).reshape(-1, nbf), C4)
        v = np.dot(np.asarray(C1).T, v.reshape(nbf, -1))

    v = v.reshape(C1.shape[1], nbf, nbf, C4.shape[1])
    v = np.einsum('qA,pqrs->pArs', C2, v)
    v = np.einsum('rA,pqrs->pqAs', C3, v)
    return v
 


class helper_SAPT(object):

    def __init__(self, dimer, memory=8, algorithm='MO'):
        print("\nInitalizing SAPT object...\n")
        tinit_start = time.time()

        # Set a few crucial attributes
        self.alg = algorithm.upper()
        dimer.reset_point_group('c1')
        dimer.fix_orientation(True)
        dimer.fix_com(True)
        dimer.update_geometry()
        nfrags = dimer.nfragments()
        if nfrags != 2:
            psi4.core.clean()
            raise Exception("Found %d fragments, must be 2." % nfrags)

        # Grab monomers in DCBS
        monomerA = dimer.extract_subsets(1, 2)
        monomerA.set_name('monomerA')
        monomerB = dimer.extract_subsets(2, 1)
        monomerB.set_name('monomerB')

        # Compute monomer properties
        tstart = time.time()
        self.rhfA, self.wfnA = psi4.energy('SCF', molecule=monomerA, return_wfn=True)
        self.V_A = np.asarray(psi4.core.MintsHelper(self.wfnA.basisset()).ao_potential())
        print("RHF for monomer A finished in %.2f seconds." % (time.time() - tstart))

        tstart = time.time()
        self.rhfB, self.wfnB = psi4.energy('SCF', molecule=monomerB, return_wfn=True)
        self.V_B = np.asarray(psi4.core.MintsHelper(self.wfnB.basisset()).ao_potential())
        print("RHF for monomer B finished in %.2f seconds." % (time.time() - tstart))

        # Setup a few variables
        self.memory = memory
        self.nmo = self.wfnA.nmo()

        # Monomer A
        self.nuc_rep_A = monomerA.nuclear_repulsion_energy()
        self.ndocc_A = self.wfnA.doccpi()[0]
        self.nvirt_A = self.nmo - self.ndocc_A
        self.idx_A = ['a', 'r']

        self.C_A = np.asarray(self.wfnA.Ca())
        self.Co_A = self.C_A[:, :self.ndocc_A]
        self.Cv_A = self.C_A[:, self.ndocc_A:]
        self.eps_A = np.asarray(self.wfnA.epsilon_a())

        # Monomer B
        self.nuc_rep_B = monomerB.nuclear_repulsion_energy()
        self.ndocc_B = self.wfnB.doccpi()[0]
        self.nvirt_B = self.nmo - self.ndocc_B
        self.idx_B = ['b', 's']

        self.C_B = np.asarray(self.wfnB.Ca())
        self.Co_B = self.C_B[:, :self.ndocc_B]
        self.Cv_B = self.C_B[:, self.ndocc_B:]
        self.eps_B = np.asarray(self.wfnB.epsilon_a())

        # Dimer
        self.nuc_rep = dimer.nuclear_repulsion_energy() - self.nuc_rep_A - self.nuc_rep_B
        self.vt_nuc_rep = self.nuc_rep / (4 * self.ndocc_A * self.ndocc_B)

        # Make slice, orbital, and size dictionaries
        self.slices = {'a': slice(0, self.ndocc_A),
                       'r': slice(self.ndocc_A, None),
                       'b': slice(0, self.ndocc_B),
                       's': slice(self.ndocc_B, None)}

        self.orbitals = {'a': self.Co_A,
                         'r': self.Cv_A,
                         'b': self.Co_B,
                         's': self.Cv_B}

        self.sizes = {'a': self.ndocc_A,
                      'r': self.nvirt_A,
                      'b': self.ndocc_B,
                      's': self.nvirt_B}

        # Compute size of ERI tensor in GB
        self.dimer_wfn = psi4.core.Wavefunction.build(dimer, psi4.core.get_global_option('BASIS'))
        mints = psi4.core.MintsHelper(self.dimer_wfn.basisset())
        self.mints = mints
        ERI_Size = (self.nmo ** 4) * 8.e-9
        memory_footprint = ERI_Size * 4
        if memory_footprint > self.memory:
            psi4.core.clean()
            raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                            limit of %4.2f GB." % (memory_footprint, self.memory))

        # Integral generation from Psi4's MintsHelper
        print('Building ERI tensor...')
        tstart = time.time()
        # Leave ERI as a Psi4 Matrix
        self.I = mints.ao_eri()
        print('...built ERI tensor in %.3f seconds.' % (time.time() - tstart))
        print("Size of the ERI tensor is %4.2f GB, %d basis functions." % (ERI_Size, self.nmo))
        self.S = np.asarray(mints.ao_overlap())

        # Save additional rank 2 tensors
        self.V_A_BB = np.einsum('ui,vj,uv->ij', self.C_B, self.C_B, self.V_A)
        self.V_A_AB = np.einsum('ui,vj,uv->ij', self.C_A, self.C_B, self.V_A)
        self.V_B_AA = np.einsum('ui,vj,uv->ij', self.C_A, self.C_A, self.V_B)
        self.V_B_AB = np.einsum('ui,vj,uv->ij', self.C_A, self.C_B, self.V_B)

        self.S_AB = np.einsum('ui,vj,uv->ij', self.C_A, self.C_B, self.S)

        if self.alg == "AO":
            tstart = time.time()
            aux_basis = psi4.core.BasisSet.build(self.dimer_wfn.molecule(), "DF_BASIS_SCF",
                                            psi4.core.get_option("SCF", "DF_BASIS_SCF"),
                                            "JKFIT", psi4.core.get_global_option('BASIS'),
                                            puream=self.dimer_wfn.basisset().has_puream())

            self.jk = psi4.core.JK.build(self.dimer_wfn.basisset(), aux_basis)
            self.jk.set_memory(int(memory * 1e9))
            self.jk.initialize()
            print("\n...initialized JK objects in %5.2f seconds." % (time.time() - tstart))

        print("\n...finished initializing SAPT object in %5.2f seconds." % (time.time() - tinit_start))

    # Compute MO ERI tensor (v) on the fly
    def v(self, string, phys=True):
        if len(string) != 4:
            psi4.core.clean()
            raise Exception('v: string %s does not have 4 elements' % string)

        # ERI's from mints are of type (11|22) - need <12|12>
        if phys:
            orbitals = [self.orbitals[string[0]], self.orbitals[string[2]],
                        self.orbitals[string[1]], self.orbitals[string[3]]]

            v = integral_transformer(self.I, *orbitals)
            return np.asarray(v).swapaxes(1, 2)
        else:
            orbitals = [self.orbitals[string[0]], self.orbitals[string[1]],
                        self.orbitals[string[2]], self.orbitals[string[3]]]
            v = integral_transformer(self.I, *orbitals)
            return np.asarray(v)

    # Grab MO overlap matrices
    def s(self, string):
        if len(string) != 2:
            psi4.core.clean()
            raise Exception('S: string %s does not have 2 elements.' % string)

        s1 = string[0]
        s2 = string[1]

        # Compute on the fly
        # return np.einsum('ui,vj,uv->ij', self.orbitals[string[0]], self.orbitals[string[1]], self.S)

        # Same monomer and index- return diaganol
        if (s1 == s2):
            return np.diag(np.ones(self.sizes[s1]))

        # Same monomer, but O-V or V-O means zeros array
        elif (s1 in self.idx_A) and (s2 in self.idx_A):
            return np.zeros((self.sizes[s1], self.sizes[s2]))
        elif (s1 in self.idx_B) and (s2 in self.idx_B):
            return np.zeros((self.sizes[s1], self.sizes[s2]))

        # Return S_AB
        elif (s1 in self.idx_B):
            return self.S_AB[self.slices[s2], self.slices[s1]].T
        else:
            return self.S_AB[self.slices[s1], self.slices[s2]]

    # Grab epsilons, reshape if requested
    def eps(self, string, dim=1):
        if len(string) != 1:
            psi4.core.clean()
            raise Exception('Epsilon: string %s does not have 1 element.' % string)

        shape = (-1,) + tuple([1] * (dim - 1))

        if (string == 'b') or (string == 's'):
            return self.eps_B[self.slices[string]].reshape(shape)
        else:
            return self.eps_A[self.slices[string]].reshape(shape)

    # Grab MO potential matrices
    def potential(self, string, side):
        if len(string) != 2:
            psi4.core.clean()
            raise Exception('Potential: string %s does not have 2 elements.' % string)

        s1 = string[0]
        s2 = string[1]

        # Two separate cases
        if side == 'A':
            # Compute on the fly
            # return np.einsum('ui,vj,uv->ij', self.orbitals[s1], self.orbitals[s2], self.V_A) / (2 * self.ndocc_A)
            if (s1 in self.idx_B) and (s2 in self.idx_B):
                return self.V_A_BB[self.slices[s1], self.slices[s2]]
            elif (s1 in self.idx_A) and (s2 in self.idx_B):
                return self.V_A_AB[self.slices[s1], self.slices[s2]]
            elif (s1 in self.idx_B) and (s2 in self.idx_A):
                return self.V_A_AB[self.slices[s2], self.slices[s1]].T
            else:
                psi4.core.clean()
                raise Exception('No match for %s indices in helper_SAPT.potential.' % string)

        elif side == 'B':
            # Compute on the fly
            # return np.einsum('ui,vj,uv->ij', self.orbitals[s1], self.orbitals[s2], self.V_B) / (2 * self.ndocc_B)
            if (s1 in self.idx_A) and (s2 in self.idx_A):
                return self.V_B_AA[self.slices[s1], self.slices[s2]]
            elif (s1 in self.idx_A) and (s2 in self.idx_B):
                return self.V_B_AB[self.slices[s1], self.slices[s2]]
            elif (s1 in self.idx_B) and (s2 in self.idx_A):
                return self.V_B_AB[self.slices[s2], self.slices[s1]].T
            else:
                psi4.core.clean()
                raise Exception('No match for %s indices in helper_SAPT.potential.' % string)
        else:
            psi4.core.clean()
            raise Exception('helper_SAPT.potential side must be either A or B, not %s.' % side)

    # Compute V tilde, Index as V_{1,2}^{3,4}
    def vt(self, string):
        if len(string) != 4:
            psi4.core.clean()
            raise Exception('Compute tilde{V}: string %s does not have 4 elements' % string)

        # Grab left and right strings
        s_left = string[0] + string[2]
        s_right = string[1] + string[3]

        # ERI term
        V = self.v(string)

        # Potential A
        S_A = self.s(s_left)
        V_A = self.potential(s_right, 'A') / (2 * self.ndocc_A)
        V += np.einsum('ik,jl->ijkl', S_A, V_A)

        # Potential B
        S_B = self.s(s_right)
        V_B = self.potential(s_left, 'B') / (2 * self.ndocc_B)
        V += np.einsum('ik,jl->ijkl', V_B, S_B)

        # Nuclear
        V += np.einsum('ik,jl->ijkl', S_A, S_B) * self.vt_nuc_rep

        return V

    # Compute CPHF orbitals
    def chf(self, monomer, ind=False):

        # This is effectively the conventional CPHF equations written in a way to conform
        # to the SAPT papers.
        if monomer not in ['A', 'B']:
            psi4.core.clean()
            raise Exception('%s is not a valid monomer for CHF.' % monomer)

        if monomer == 'A':
            # Form electrostatic potential
            w_n = 2 * np.einsum('basa->bs', self.v('basa'))
            w_n += self.V_A_BB[self.slices['b'], self.slices['s']]
            eps_ov = (self.eps('b', dim=2) - self.eps('s'))

            # Set terms
            v_term1 = 'sbbs'
            v_term2 = 'sbsb'
            no, nv = self.ndocc_B, self.nvirt_B

        if monomer == 'B':
            w_n = 2 * np.einsum('abrb->ar', self.v('abrb'))
            w_n += self.V_B_AA[self.slices['a'], self.slices['r']]
            eps_ov = (self.eps('a', dim=2) - self.eps('r'))
            v_term1 = 'raar'
            v_term2 = 'rara'
            no, nv = self.ndocc_A, self.nvirt_A

        # Form A matrix (LHS)
        voov = self.v(v_term1)
        v_vOoV = 2 * voov - self.v(v_term2).swapaxes(2, 3)
        v_ooaa = voov.swapaxes(1, 3)
        v_vVoO = 2 * v_ooaa - v_ooaa.swapaxes(2, 3)
        A_ovOV = np.einsum('vOoV->ovOV', v_vOoV + v_vVoO.swapaxes(1, 3))

        # Mangled the indices so badly with strides we need to copy back to C contigous
        nov = nv * no
        A_ovOV = A_ovOV.reshape(nov, nov).copy(order='C')
        A_ovOV[np.diag_indices_from(A_ovOV)] -= eps_ov.ravel()

        # Call DGESV, need flat ov array
        B_ov = -1 * w_n.ravel()
        t = np.linalg.solve(A_ovOV, B_ov)
        # Our notation wants vo array
        t = t.reshape(no, nv).T

        if ind:
            # E200 Induction energy is free at the point
            e20_ind = 2 * np.einsum('vo,ov->', t, w_n)
            return (t, e20_ind)
        else:
            return t

    def compute_sapt_JK(self, Cleft, Cright, tensor=None):

        if self.alg != "AO":
            raise Exception("Attempted a call to JK builder in an MO algorithm")

        return_single = False
        if not isinstance(Cleft, (list, tuple)):
            Cleft = [Cleft]
            return_single = True
        if not isinstance(Cright, (list, tuple)):
            Cright = [Cright]
            return_single = True
        if (not isinstance(tensor, (list, tuple))) and (tensor is not None):
            tensor = [tensor]
            return_single = True

        if len(Cleft) != len(Cright):
            raise Exception("Cleft list is not the same length as Cright list")

        zero_append = []
        num_compute = 0

        for num in range(len(Cleft)):
            Cl = Cleft[num]
            Cr = Cright[num]

            if (Cr.shape[1] == 0) or (Cl.shape[1] == 0):
                zero_append.append(num)
                continue

            if tensor is not None:
                mol = Cl.shape[1]
                mor = Cr.shape[1]
                
                if (tensor[num].shape[0] != mol) or (tensor[num].shape[1] != mor):
                    raise Exception("compute_sapt_JK: Tensor size does not match Cl (%d) /Cr (%d) : %s" %
                                                            (mol, mor, str(tensor[num].shape)))
                if mol < mor:
                    Cl = np.dot(Cl, tensor[num])
                else:
                    Cr = np.dot(Cr, tensor[num].T)

            Cl = psi4.core.Matrix.from_array(Cl)
            Cr = psi4.core.Matrix.from_array(Cr)

            self.jk.C_left_add(Cl)
            self.jk.C_right_add(Cr)
            num_compute += 1
        
        self.jk.compute() 

        J_list = []
        K_list = []
        for num in range(num_compute):
            J_list.append(np.array(self.jk.J()[num])) 
            K_list.append(np.array(self.jk.K()[num])) 

        self.jk.C_clear()

        z = np.zeros((self.nmo, self.nmo))
        for num in zero_append:
            J_list.insert(num, z)
            K_list.insert(num, z)

        if return_single:
            return J_list[0], K_list[0]
        else:
            return J_list, K_list

    def chain_dot(self, *dot_list):
        result = dot_list[0]
        for x in range(len(dot_list) - 1):
            result = np.dot(result, dot_list[x + 1])
        return result

# End SAPT helper

class sapt_timer(object):
    def __init__(self, name):
        self.name = name
        self.start = time.time()
        print('\nStarting %s...' % name)

    def stop(self):
        t = time.time() - self.start
        print('...%s took a total of % .2f seconds.' % (self.name, t))


def sapt_printer(line, value):
    spacer = ' ' * (20 - len(line))
    print(line + spacer + '% 16.8f mH  % 16.8f kcal/mol' % (value * 1000, value * 627.509))
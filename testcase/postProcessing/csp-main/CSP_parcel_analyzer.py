import numpy as np
import os, sys, subprocess
import cantera as ct
from scipy.linalg import eig
from tqdm import tqdm
import vtk
import pandas as pd
import argparse


class CSPAnalyzer_parcel:
    def __init__(self, yaml_file, x=None):
        self.gas = ct.Solution(yaml_file, kinetics="gas")
        self.yaml_file = yaml_file
        self.n_species = len(self.gas.species_names)
        self.n_reactions = self.gas.n_reactions
        self.reaction_names = [self.gas.reaction_equation(i) for i in range(self.gas.n_reactions)]
        #self.n_points = len(x) if x is not None else 0
        self.n_points = 1


    def eigen_decomposition_parcel(self, p, T, Y):
        import pyjacob    
    
        eigenvalues = np.zeros((self.n_points, self.n_species), dtype=complex) # for parcel n_points =1c 
        vl = np.zeros((self.n_points, self.n_species, self.n_species), dtype=complex)
        vr = np.zeros((self.n_points, self.n_species, self.n_species), dtype=complex)
        wherePositive = 0.0

        #for i in tqdm(range(self.n_points)):
        z = np.zeros(self.n_species)
        z[0] = T
        z[1:] = Y[:-1]
        jac = np.zeros(self.n_species**2)
        pyjacob.py_eval_jacobian(0, p, z, jac)
        jac = jac.reshape(self.n_species, self.n_species)
        w, vlI, vrI = eig(jac, left=True, right=True)
        idx = w.argsort()[::-1]
        eigenvalues[0,:] = w[idx]
        vl[0, :, :] = vlI[:, idx]
        vr[0, :, :] = vrI[:, idx]

        if (max(eigenvalues[0,:].real) >= 1e-7):
            wherePositive = 1.0

        return eigenvalues, vl, vr, wherePositive
        
        
    def radical_pointer_parcel(self, vl, vr):
        rad_point = np.zeros((self.n_points, self.n_species, self.n_species), dtype=complex)

        for i in range(self.n_species):  
            # Compute outer product between right and left eigenvectors 
            cd = np.outer(vr[0, :, i], vl[0, :, i])
            rad_point[0, :, i] = np.diag(cd)

        # Normalize across species (axis=1)
        rad_point /= np.sum(np.abs(rad_point), axis=1, keepdims=True)

        return rad_point

        
        
    def generalized_stoi_parcel(self):
        stoi_r = np.zeros((self.gas.n_species, self.gas.n_reactions))
        stoi_p = np.zeros((self.gas.n_species, self.gas.n_reactions))

        wt = self.gas.molecular_weights
        rho = self.gas.density
        #ums = self.gas.partial_molar_int_energies / wt #this is because of const. vol
        #if one wants const press - hence cp, he must use 
        hi = self.gas.partial_molar_enthalpies
        # and gas.cp_mass
        cpb = self.gas.cp_mass
        nu = self.gas.product_stoich_coeffs - self.gas.reactant_stoich_coeffs

        # Species generalized stoichiometric matrix
        stoi_r[1:, :] = nu[:-1, :] * wt[:-1, np.newaxis] / rho
        stoi_p[1:, :] = -nu[:-1, :] * wt[:-1, np.newaxis] / rho

        # Temperature generalized stoichiometric matrix
        stoi_r[0, :] = -np.sum((hi * wt * nu.T).T, axis=0) / (rho * cpb)
        stoi_p[0, :] = -stoi_r[0, :]

        return stoi_r, stoi_p
        
        
    def participation_index_parcel(self, p, T, Y, vr):
        api = np.zeros((self.n_points, 2 * self.n_reactions, self.n_species), dtype=complex)

        #for k in tqdm(range(self.n_points)):
        self.gas.TPY = T, p, Y[:]
        stoi_r, stoi_p = self.generalized_stoi_parcel()
        fwdk = self.gas.forward_rates_of_progress
        revk = self.gas.reverse_rates_of_progress
        reaction_names = self.reaction_names

        for r in range(self.n_reactions):
            BS = np.matmul(vr[0,:,:].T, stoi_r[:, r].reshape(-1, 1))
            api[0, r, :] = fwdk[r] * BS[:, 0]
            BS = np.matmul(vr[0,:,:].T, stoi_p[:, r].reshape(-1, 1))
            api[0, r + self.n_reactions, :] = revk[r] * BS[:, 0]
            sig_h = np.sum(api[0, :, :], axis=0)
            api[0, :, :] = api[0, :, :]*np.sign(sig_h[None, :])

        for i in range(self.n_species):
            sum_mode = np.sum(np.abs(api[0, :, i]))
            api[0, :, i] = api[0, :, i] / sum_mode if sum_mode != 0 and not np.isnan(sum_mode) else 0

        return api,reaction_names


    def explosive_dataframe_parcel(self, rad_point, api, eigenvalues, x, T, n_diag=5):
        radicals = ['T'] + self.gas.species_names

        records = []

        for k in tqdm(range(self.n_points)):
            idx = np.abs(rad_point[k, :, 0]).argsort()[::-1]
            jdx = np.abs(api[k, :, 0]).argsort()[::-1]

            for i in range(n_diag):
                rec = {
                    'x': x[k],
                    'T': T[k],
                    'lambda_r': eigenvalues[k, 0].real,
                    'lambda_i': eigenvalues[k, 0].imag,
                    'radical_name': radicals[idx[i]],
                    'radical_pointer': rad_point[k, idx[i], 0].real,
                    'amp_par_idx': api[k, jdx[i], 0].real
                }
                r_idx = jdx[i]
                reac_eq = self.gas.reaction_equations()[r_idx % self.n_reactions]
                rec['amp_reac'] = ('(f) ' if r_idx < self.n_reactions else '(b) ') + reac_eq
                rec['amp_reac_num'] = str(r_idx % self.n_reactions +1) + ('(f)' if r_idx < self.n_reactions else '(b)')
                records.append(rec)

        return pd.DataFrame(records)

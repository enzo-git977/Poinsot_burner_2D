import numpy as np
import os, sys, subprocess
import cantera as ct
from scipy.linalg import eig
from tqdm import tqdm
import vtk
import pandas as pd
import argparse


class CSPAnalyzer:
    def __init__(self, yaml_file, x=None):
        self.gas = ct.Solution(yaml_file, kinetics="gas")
        self.yaml_file = yaml_file
        self.n_species = len(self.gas.species_names)
        self.n_reactions = self.gas.n_reactions
        self.n_points = len(x) if x is not None else 0


    def eigen_decomposition(self, p, T, Y):
        import pyjacob    
    
        eigenvalues = np.zeros((self.n_points, self.n_species), dtype=complex) # for parcel n_points =1c 
        vl = np.zeros((self.n_points, self.n_species, self.n_species), dtype=complex)
        vr = np.zeros((self.n_points, self.n_species, self.n_species), dtype=complex)
        wherePositive = np.zeros(self.n_points)

        for i in tqdm(range(self.n_points)):
            z = np.zeros(self.n_species)
            z[0] = T[i]
            z[1:] = Y[i, :-1]
            jac = np.zeros(self.n_species**2)
            pyjacob.py_eval_jacobian(0, p[i], z, jac)
            jac = jac.reshape(self.n_species, self.n_species)
            w, vlI, vrI = eig(jac, left=True, right=True)
            idx = w.argsort()[::-1]
            eigenvalues[i, :] = w[idx]
            vl[i, :, :] = vlI[:, idx]
            vr[i, :, :] = vrI[:, idx]

            if (max(eigenvalues[i, :].real) >= 1e-7):
                wherePositive[i] = 1#max(eigenvalues[i, :].real)

        return eigenvalues, vl, vr, wherePositive
        
        
    def radical_pointer(self, vl, vr):
        rad_point = np.zeros((self.n_points, self.n_species, self.n_species), dtype=complex)

        for i in tqdm(range(self.n_species)):
            cd = np.einsum('kj,km->kjm', vr[:, :, i], vl[:, :, i])
            diag = np.einsum('kjj->kj', cd)
            rad_point[:, :, i] = diag

        # normalization of radical pointer for better representation
        rad_point /= np.sum(np.abs(rad_point), axis=1, keepdims=True)
        return rad_point
        
        
    def generalized_stoi(self):
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
        
        
    def participation_index(self, p, T, Y, vr):
        api = np.zeros((self.n_points, 2 * self.n_reactions, self.n_species), dtype=complex)

        for k in tqdm(range(self.n_points)):
            self.gas.TPY = T[k], p[k], Y[k, :]

            stoi_r, stoi_p = self.generalized_stoi()

            fwdk = self.gas.forward_rates_of_progress
            revk = self.gas.reverse_rates_of_progress

            for r in range(self.n_reactions):
                BS = np.matmul(vr[k,:,:].T, stoi_r[:, r].reshape(-1, 1))
                api[k, r, :] = fwdk[r] * BS[:, 0]

                BS = np.matmul(vr[k,:,:].T, stoi_p[:, r].reshape(-1, 1))
                api[k, r + self.n_reactions, :] = revk[r] * BS[:, 0]

            sig_h = np.sum(api[k, :, :], axis=0)
            api[k, :, :] = api[k, :, :]*np.sign(sig_h[None, :])

            for i in range(self.n_species):
                sum_mode = np.sum(np.abs(api[k, :, i]))
                api[k, :, i] = api[k, :, i] / sum_mode if sum_mode != 0 and not np.isnan(sum_mode) else 0

        return api


    def explosive_dataframe(self, rad_point, api, eigenvalues, x, T, n_diag=5):
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

import os
#os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
#os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
#os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
#os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
#os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

from openbabel import pybel

from rdkit import Chem

from multiprocessing import Pool

from moltaut.molsolv.utils import filter_mol
from moltaut.molsolv.descriptor import mol2vec
from moltaut.molsolv.models import load_model
from moltaut.molsolv.optimize_mol import optimize
from moltaut.molsolv.gen_confs import gen_confs_set

from moltaut.get_vmrs import enumerate_vmrs

from collections import defaultdict

import numpy as np
import pandas as pd
import torch

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nmodel, imodel = load_model()


def linker_to_hydrogen(smi):
    mol = Chem.MolFromSmiles(smi)

    linker_aids = []
    for at in mol.GetAtoms():
        if at.GetSymbol() == "*":
            idx = at.GetIdx()
            linker_aids.append(idx)

    emol = Chem.RWMol(mol)
    for idx in linker_aids:
        emol.ReplaceAtom(idx, Chem.Atom(6))
    nmol = emol.GetMol()
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(nmol)))
    
    return smi


def predict_single(pmol, model, fmax):
    obmol = pmol.OBMol
    if fmax:
        obmol, dE = optimize(obmol, fmax)
    else:
        dE = 0.0
    data = mol2vec(obmol)
    # npmol = pybel.Molecule(obmol)
    with torch.no_grad():
        data = data.to(device)
        solv = model(data).cpu().numpy()[0][0].item() # item() converts np.float32 to float
    #npmol.write("mol", str(-1.0 * solv * 100000) + ".mol")
    return solv, dE / 27.2114 * 627.5094


def predict_multicore_wrapper(param):
    idx, smi, block, fmax = param
    pmol = pybel.readstring("mol", block)
    solv, dE = predict_single(pmol, nmodel, fmax)
    return [idx, smi, solv, dE, solv+dE]


# def predict_by_mol(pmol, fmax, model):
#     if not filter_mol( pmol ):
#         print("#### Warning filter molecule")
#         return
#     solv, dE = predict_single(pmol, model, fmax)

#     return solv, dE


# def predict_by_smi(smi, fmax,  num_confs):
#     # pmol = pybel.readstring("smi", smi)
#     blocks = gen_confs_set(smi, num_confs)
#     params = zip(blocks, [fmax for i in range(len(blocks))])

#     # pool = Pool()
#     # score = pool.map(predict_multicore_wrapper, params)
#     # pool.close()
#     score = predict_multicore_wrapper(params)
    
#     score_sort = sorted(score, key=lambda x: x[2])
#     solv, dE, dG = score_sort[0]
#     return solv, dE 


def predict_by_smis(smis, fmax, num_confs):
    params = []
    scores = defaultdict(list)
    for idx, smi in enumerate(smis):
        blocks = gen_confs_set(smi, num_confs)
        for block in blocks:
            params_ = [idx, smi, block, fmax]
            score_ = predict_multicore_wrapper(params_)
            # [idx, smi, solv, dE, solv+dE]
            params.append(params_)
            scores[idx].append(score_)
            
    # pool = Pool()
    # score = pool.map(predict_multicore_wrapper, params)
    # pool.close()

    output = []
    for k, v in scores.items():
        for (idx, smi, solv, dE, _) in sorted(v, key=lambda x: x[-1]):
            output.append((idx, smi, solv, dE))
    # output = []
    # df_scores = pd.DataFrame(scores)
    # for smi_idx, dfsg in df_scores.groupby(0):
    #     #print(smi_idx)
    #     dfsg_sorted = dfsg.sort_values(4)
    #     smi = dfsg_sorted.iloc[0, 1]
    #     solv = dfsg_sorted.iloc[0, 2]
    #     dE = dfsg_sorted.iloc[0, 3]
    #     output.append([smi_idx, smi, solv, dE])
    
    #print("output:", len(output))
    return output


def calc_solv(tauts, fmax, num_confs, is_fragment) -> list[tuple]:
    if is_fragment:
        tauts_smis_include_linker = [Chem.MolToSmiles(taut.mol) for taut in tauts]
        tauts_smis_exclude_linker = [linker_to_hydrogen(smi) for smi in tauts_smis_include_linker]    
        output = predict_by_smis(tauts_smis_exclude_linker, fmax, num_confs) 
    else:
        tauts_smis = [taut.smi for taut in tauts]
        output = predict_by_smis(tauts_smis, fmax, num_confs)

    res = [(idx, tsmi, solv, dE, solv + 0.72*dE) for idx, tsmi, solv, dE in output]
    # relative energy
    min_dG = min([x[-1] for x in res])
    res = [(idx, tsmi, solv, dE, dG-min_dG) for (idx, tsmi, solv, dE, dG) in res]
    res = sorted(res, key=lambda x: x[-1])
    # df = pd.DataFrame(res)
    # if len(df) == 0:
    #     return df 
    # df[3] = df[1] + df[2]*0.72
    # df[3] = df[3] - df[3].min()
    # df.columns = ["smi", "solv", "internal", "dG"] 
    
    return res

def rank_tauts(tauts, num_confs, fmax=0.01, is_fragment=True):
    res = calc_solv(tauts, fmax, num_confs, is_fragment)
    # smirks_rules = [taut.smirks for taut in tauts]
    # df["smirks"] = smirks_rules
    # df = df.sort_values("dG")
    return res

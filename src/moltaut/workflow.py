from importlib.resources import files, as_file
from collections import defaultdict
from collections.abc import Callable
from typing import Self
from itertools import combinations

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from openbabel import pybel

from moltaut.molsolv.models import load_model
from moltaut.molsolv.descriptor import mol2vec
from moltaut.molgpka.predict_pka import predict

from batchopt import BatchOptimize

import rdworks
import torch
import time


class Dummy_BatchSingplePoint():
    def __init__(self, rdmols):
        self.rdmols = rdmols

    def __str__(self):
        return "Dummy_BatchSinglePoint"

    def run(self, **kwargs):
        from collections import namedtuple
        Calculated = namedtuple('Calculated', ['mols',])
        for rdmol in self.rdmols:
            rdmol.SetProp('E_tot(kcal/mol)', '1.0')
        return Calculated(mols=self.rdmols)


class Dummy_BatchOptimizer():
    def __init__(self, rdmols):
        self.rdmols = rdmols

    def __str__(self):
        return "Dummy_BatchOptimizer"

    def run(self, **kwargs):
        from collections import namedtuple
        Optimized = namedtuple('Optimized', ['mols',])
        for rdmol in self.rdmols:
            rdmol.SetProp('E_tot_init(kcal/mol)', '10.0')
            rdmol.SetProp('E_tot(kcal/mol)', '1.0')
            rdmol.SetProp('Converged', 'True')
        return Optimized(mols=self.rdmols)
    

class Tautomer:
    """Tautomer enumeration, ordering, and protonation
    
    References:

        Workflow:
        X. Pan, et al., MolTaut: A tool for the rapid generation of favorable tautomer in aqueous solution. 
        J. Chem. Inf. Model. 63, 1833-1840 (2023).

        SMIRKS (SMILES ReaKtion Specification):
        Dhaked, D. K.; Ihlenfeldt, W.-D.; Patel, H.; Delannée, V.; Nicklaus, M. C. 
        Toward a Comprehensive Treatment of Tautomerism in  Chemoinformatics Including in  InChI V2. 
        J. Chem. Inf. Model. 2020, 60 (3), 1253-1275.
    """
    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    uncharger = rdMolStandardize.Uncharger()

    forbidden = [ "O=[N+]([O-])", ]
    forbidden_patterns = [Chem.MolFromSmarts(_) for _ in forbidden]

    def __init__(self, smiles: str):
        self.smiles : str = smiles
        self.rdmol : Chem.Mol = Chem.MolFromSmiles(smiles)
        # self.rdmol = self.uncharger.uncharge(self.rdmol)
        self.dict : dict[str,list[str]] = {'self' : [self.smiles]} # transform dictionary
        self.smirks : list[tuple[str,str]] = []
        self.revdict = defaultdict(list) # reverse dictionary
        with as_file(files('moltaut').joinpath("smirks_transform_all.txt")) as path:
            # SMIRKS (SMILES ReaKtion Specification)
            # Dhaked, D. K.; Ihlenfeldt, W.-D.; Patel, H.; Delannée, V.; Nicklaus, M. C. 
            # Toward a Comprehensive Treatment of Tautomerism in  Chemoinformatics Including in  InChI V2. 
            # J. Chem. Inf. Model. 2020, 60 (3), 1253-1275.
            with open(path, 'r') as f:
                # ex. [#1:1][C0:2]#[N0:3]>>[C-:2]#[N+:3][#1:1]	PT_20_00
                contents = f.readlines()
                self.smirks = [line.strip().split("\t") for line in contents]
                # initialize
                for idx, (smrk, name) in enumerate(self.smirks):
                    self.dict[str(idx) + "_" + name] = []
        
        self.enumerated : list[str] = [] # enumerated tautomers (SMILES)
        self.num_confs = None
        self.optimizer = None
        self.device = None
        self.nmodel = None
        self.imodel = None
        self.ordered : list = []
        self.popular : list = []
        self.states : list[tuple[str, str]] = []
        self.microstates : list = []


    def _contains_phosphorus(self) -> bool:
        return any([at.GetAtomicNum() == 15 for at in self.rdmol.GetAtoms()])


    def _acceptable(self, rom: Chem.Mol) -> bool:
        if not rom:
            return False
        for pattern in self.forbidden_patterns:
            matches = sum(rom.GetSubstructMatches(pattern), ())
            for at in rom.GetAtoms():
                if (at.GetFormalCharge() != 0) and (at.GetIdx() not in matches):
                    return False
        return True 


    def _kekulize_resonance_forms(self, rwm: Chem.Mol) -> list[Chem.Mol]:
        rwm = Chem.AddHs(rwm)
        mols = Chem.ResonanceMolSupplier(rwm, Chem.KEKULE_ALL)
        kekulized_mols = []
        for _ in mols:
            if not self._acceptable(_):
                continue
            smi = Chem.MolToSmiles(_, kekuleSmiles=True)
            new_mol = Chem.MolFromSmiles(smi, Tautomer.ps)
            kekulized_mols.append(new_mol)
        return kekulized_mols


    def _transform(self, rwm: Chem.Mol):
        # mark dummy atoms for protection
        # RDKit recognizes the special property key _protected. 
        # If this property is set to a "truthy" value (like '1'), 
        # RDKit will ignore this atom when matching the reactant patterns of the chemical reaction.
        for atom in rwm.GetAtoms():
            if atom.GetAtomicNum() == 0: # if atom == "*": # dummy atom
                atom.SetProp('_protected', '1')

        # Kekule form of the SMILES
        Chem.Kekulize(rwm, clearAromaticFlags=True)

        for idx, (smrk, name) in enumerate(self.smirks):
            rxn = AllChem.ReactionFromSmarts(smrk)
            new_molecules = rxn.RunReactants((rwm,))
            if len(new_molecules) > 0:
                for unit in new_molecules:
                    _ = Chem.MolToSmiles(unit[0])
                    _mol = Chem.MolFromSmiles(_, sanitize=True)
                    _smi = None
                    if _mol:
                        _smi = Chem.MolToSmiles(_mol)
                    if _smi:
                        self.dict[str(idx)+"_"+name].append(_smi)

    def _collect(self):
        for rule, smis in self.dict.items():
            for _ in set(smis):
                self.revdict[_].append(rule)
        return [smi for smi, rules in self.revdict.items()]


    def enumerate(self) -> Self:
        m = Chem.Mol(self.rdmol) # copy
        if self._contains_phosphorus():
            self.enumarated = [self.smiles]
        else:
            kms = self._kekulize_resonance_forms(m)  
            for km in kms: # can be parallel
                self._transform(km)

            for i in range(5):
                transformed_smis = []
                for rule, smis in self.dict.items():
                    transformed_smis += smis # list
                transformed_smis = set(transformed_smis)
                transformed_mols = [Chem.MolFromSmiles(_) for _ in transformed_smis]
                for tm in transformed_mols: # can be parallel
                    kms = self._kekulize_resonance_forms(tm)
                    for km in kms:
                        self._transform(km)
      
            self.enumerated = self._collect()
        
        return self
    

    def _solvation_energy(self, molblock: str) -> float:
        obmol = pybel.readstring("mol", molblock).OBMol
        data = mol2vec(obmol)
        with torch.no_grad():
            data = data.to(self.device)
            solv = self.nmodel(data).cpu().numpy()[0][0].item() 
            # item() converts np.float32 to float
        # return solv, dE / 27.2114 * 627.5094
        return solv


    def order(self, 
              num_confs: int | None = None, 
              optimizer: Callable | None = None,
              device: str | None = None,
              dG_cutoff: float = 2.76) -> Self:
        
        if num_confs is None:
            num_rot = AllChem.CalcNumRotatableBonds(self.rdmol)
            if num_rot < 8:
                self.num_confs = 50
            elif num_rot >= 8 and num_rot <= 12:
                self.num_confs = 200
            elif num_rot > 12:
                self.num_confs = 300
        else:
            self.num_confs = num_confs
        
        if optimizer is None:
            self.optimizer : Callable = Dummy_BatchOptimizer
        else:
            self.optimizer : Callable = optimizer
        
        if device is None:
            self.device : str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if self.nmodel is None or self.imodel is None:
            self.nmodel, self.imodel = load_model(self.device) # (neutral, ionized)
        
        scores = defaultdict(list)
        for tidx, smi in enumerate(self.enumerated):
            mol = rdworks.Mol(smi, f'tautomer_{tidx}')
            # generate and optimize each conformer with default MMFF94
            mol = mol.make_confs(n=self.num_confs, method='ETKDG')
            mol = mol.optimize_confs(calculator=self.optimizer, batchsize_atoms=16384)
            # remove similar conformers (RMSD <0.3) [and stereo-flipped conformers by default]
            mol = mol.drop_confs(similar=True, similar_rmsd=0.3).rename()
            molblocks = [(c.to_molblock(), c.props['E_tot(kcal/mol)']) for c in mol.confs]
            for (molblock, dE) in molblocks:
                solv = self._solvation_energy(molblock)
                dG = solv + 0.72 * dE # ANI2x
                # each conformer has scores
                scores[tidx].append((dG, dE, solv, smi, molblock))

        agg = [] # aggregated output
        for k, v in scores.items():
            for (dG, dE, solv, smi, molblock) in sorted(v): # default ascending (lowest to highest)
                agg.append((dG, dE, solv, smi, molblock))

        min_dG = min([x[0] for x in agg]) # for relative energy
        self.ordered = sorted([(dG-min_dG, dE, solv, smi, molblock) for (dG, dE, solv, smi, molblock) in agg])
        self.popular = [(smi, molblock) for (rel_dG, _, _, smi, molblock) in self.ordered if rel_dG < dG_cutoff]
        temp = {}
        for (smi, molblock) in self.popular:
            if smi not in temp:
                temp[smi] = molblock # first appearance (=lowest energy)
        self.states = [(k,v) for k, v in temp.items()]
        
        return self


    @staticmethod
    def _get_pKa_data(mol, ph, tph):
        stable_data, unstable_data = [], []
        for at in mol.GetAtoms():
            props = at.GetPropsAsDict()
            acid_or_basic = props.get('ionization', False)
            pKa = float(props.get('pKa', False))
            idx = at.GetIdx()
            if acid_or_basic == "A":
                if pKa < ph - tph:
                    stable_data.append( [idx, pKa, "A"] )
                elif ph - tph <= pKa <= ph + tph:
                    unstable_data.append( [idx, pKa, "A"] )
            elif acid_or_basic == "B":
                if pKa > ph + tph:
                    stable_data.append( [idx, pKa, "B"] )
                elif ph - tph <= pKa <= ph + tph:
                    unstable_data.append( [idx, pKa, "B"] )
            else:
                continue
        return stable_data, unstable_data


    @staticmethod
    def _mark_ionized_atoms(mol, acid_dict, base_dict):
        for at in mol.GetAtoms():
            idx = at.GetIdx()
            if idx in set(acid_dict.keys()):
                value = acid_dict[idx] # pKa value
                nat = at.GetNeighbors()[0] # bonded atom
                nat.SetProp("ionization", "A")
                nat.SetProp("pKa", str(value))
            elif idx in set(base_dict.keys()):
                value = base_dict[idx]
                at.SetProp("ionization", "B")
                at.SetProp("pKa", str(value))
            else:
                at.SetProp("ionization", "O")
        nmol = AllChem.RemoveHs(mol)
        return nmol

    @staticmethod
    def _remove_H(at):
        Hs = at.GetNumExplicitHs()
        if Hs > 0:
            at.SetFormalCharge(-1)
            at.SetNumExplicitHs(Hs-1)
        return

    @staticmethod
    def _add_H(at):
        Hs = at.GetNumExplicitHs()
        at.SetFormalCharge(1)
        at.SetNumExplicitHs(Hs+1)
        return

    @staticmethod
    def _modify_stable_pka(new_mol, stable_data):
        for pka_data in stable_data:
            idx, pka, acid_or_basic = pka_data
            at = new_mol.GetAtomWithIdx(idx)
            if acid_or_basic == "A":
                Tautomer._remove_H(at)
            elif acid_or_basic == "B":
                Tautomer._add_H(at)
        return

    @staticmethod
    def _modify_unstable_pka(mol, unstable_data, i):
        combine_pka_datas = list(combinations(unstable_data, i))
        new_unsmis = []
        for pka_datas in combine_pka_datas:
            new_mol = Chem.Mol(mol)
            if len(pka_datas) == 0:
                continue
            for pka_data in pka_datas:
                idx, pka, acid_or_basic = pka_data
                at = new_mol.GetAtomWithIdx(idx)
                if acid_or_basic == "A":
                    Tautomer._remove_H(at)
                elif acid_or_basic == "B":
                    Tautomer._add_H(at)
            smi = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(new_mol)))
            new_unsmis.append(smi)
        return new_unsmis


    def protonate(self, ph: float = 7.4, tol: float = 1.0) -> Self:
        for (smi, molblock) in self.states:
            # omol = Chem.MolFromSmiles(smi)
            omol = Chem.MolFromMolBlock(molblock)
            obase_dict, oacid_dict, omol = predict(omol)
            # xx_dict = {atom_index: pKa_value, ...}
            mc = Tautomer._mark_ionized_atoms(omol, oacid_dict, obase_dict)
            stable_data, unstable_data = Tautomer._get_pKa_data(mc, ph, tol)
            protonated_smiles = []
            n = len(unstable_data)
            if n == 0:
                new_mol = Chem.Mol(mc)
                Tautomer._modify_stable_pka(new_mol, stable_data)
                smi = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(new_mol)))
                protonated_smiles.append(smi)
            else:
                for i in range(n + 1):
                    new_mol = Chem.Mol(mc)
                    Tautomer._modify_stable_pka(new_mol, stable_data)
                    if i == 0:
                        protonated_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))))
                    new_unsmis = Tautomer._modify_unstable_pka(new_mol, unstable_data, i)
                    protonated_smiles.extend(new_unsmis)
            self.microstates.extend(protonated_smiles)
        
        # make SMILES unique
        self.microstates = list(set(self.microstates))
        
        return self
    

if __name__ == '__main__':
    from moltaut.molgpka.protonate import protonate_mol
    for smi in [
        # "Nc1nc(O)c2[nH]nnc2n1",
        "Oc1ccccc1",
        "COC(=O)c1ccc(O)cc1",
        "N#CC1=C(N)Oc2[nH]ncc2C1",
        "OSc1ncc[nH]1",
        # "O=C(Cc1ccccc1)c1cccs1",
        # "Oc1nc2ccccc2nc1",
        # "O=C1NN=CN1",
        # "OC(=O)COC(=O)N[C@]12CC[C@H](CC1)[C@@H]1[C@H]2C(=O)N(C1=O)c1ccc(cc1)NC(=O)C",
        # "OCc(n1)cnc(c12)nc(N)[nH]c2=O",
        # "O=[N+]([O-])c1ccc2cn[nH]c2c1",
        # "c1ccc2cn[nH]c2c1",
        ]:

        # t = Tautomer(smi).enumerate().order(num_confs=200, optimizer=Dummy_BatchOptimizer, device='cuda')
        # print(t.enumerated)
        # print(t.ordered)
        # print()
        pass

    for smi in [
        "Oc1ccccc1",
        "COC(=O)c1ccc(O)cc1",
        "N#CC1=C(N)Oc2[nH]ncc2C1",
        "OSc1ncc[nH]1",
        "O=C(Cc1ccccc1)c1cccs1",
        "Oc1nc2ccccc2nc1",
        "O=C1NN=CN1",
        "OC(=O)COC(=O)N[C@]12CC[C@H](CC1)[C@@H]1[C@H]2C(=O)N(C1=O)c1ccc(cc1)NC(=O)C",
        "OCc(n1)cnc(c12)nc(N)[nH]c2=O",
        "O=[N+]([O-])c1ccc2cn[nH]c2c1",
        "c1ccc2cn[nH]c2c1",
        "CN(C)CCCN1C2=CC=CC=C2SC2=C1C=C(C=C2)C(C)=O",
        "Nc1cc(C(F)(F)F)c(-c2cc(N3CCCC3)nc(N3CCOCC3)n2)cn1"
        ]:
        m = Chem.MolFromSmiles(smi)
        smi = Chem.MolToSmiles(m)

        start_time = time.time()
        t = Tautomer(smi).enumerate().order(optimizer=BatchOptimize).protonate()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")

        print("SMILES=", t.smiles)
        print("enumerated=")
        for i, smiles in enumerate(t.enumerated, start=1):
            print(i, smiles)
        print("states=")
        for i, (smiles, molblock) in enumerate(t.states, start=1):
            print(i, smiles)
        print("microstates=")
        for i, item in enumerate(t.microstates, start=1):
            print(i, item)
        print()

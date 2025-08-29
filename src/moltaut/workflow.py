from importlib.resources import files, as_file
from collections import defaultdict
from collections.abc import Callable

from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import pybel

from moltaut.molsolv.models import load_model
from moltaut.molsolv.descriptor import mol2vec

import rdworks
import torch


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
    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    forbidden = [ "O=[N+]([O-])", ]
    forbidden_patterns = [Chem.MolFromSmarts(_) for _ in forbidden]

    def __init__(self, smiles: str):
        self.smiles : str = smiles
        self.rdmol : Chem.Mol = Chem.MolFromSmiles(smiles)
        self.dict : dict[str,list[str]] = {'self' : [self.smiles]} # transform dictionary
        self.smirks : list[tuple[str,str]] = []
        with as_file(files('moltaut').joinpath("smirks_transform_all.txt")) as path:
            # SMIRKS (SMILES ReaKtion Specification)
            with open(path, 'r') as f:
                # ex. [#1:1][C0:2]#[N0:3]>>[C-:2]#[N+:3][#1:1]	PT_20_00
                contents = f.readlines()
                self.smirks = [line.strip().split("\t") for line in contents]
                # initialize
                for idx, (smrk, name) in enumerate(self.smirks):
                    self.dict[str(idx) + "_" + name] = []
        self.revdict = defaultdict(list) # reverse dictionary
        self.enumerated : list[str] = [] # enumerated tautomers (SMILES)


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


    def enumerate(self) -> list[str]:
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
        
        return self.enumerated
    

class TautomerOrdered:
    def __init__(self, enumerated: list[str], num_confs: int | None = None, device: str | None = None):
        self.enumerated : list[str] = enumerated
        self.optimizer : Callable = Dummy_BatchOptimizer
        self.num_confs : int | None = num_confs

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.nmodel, self.imodel = load_model(self.device) # (neutral, ionized)


    def _solvation_energy(self, molblock: str):
        obmol = pybel.readstring("mol", molblock).OBMol
        # obmol = pmol.OBMol
        # if fmax:
        #     obmol, dE = optimize(obmol, fmax)
        # else:
        #     dE = 0.0
        data = mol2vec(obmol)
        # npmol = pybel.Molecule(obmol)
        with torch.no_grad():
            data = data.to(self.device)
            solv = self.nmodel(data).cpu().numpy()[0][0].item() 
            # item() converts np.float32 to float
        #npmol.write("mol", str(-1.0 * solv * 100000) + ".mol")
        # return solv, dE / 27.2114 * 627.5094
        return solv


    def order(self):
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
                dG = solv + 0.72 * dE
                # each conformer has scores
                scores[tidx].append((dG, dE, solv, smi, tidx))

        agg = [] # aggregated output
        for k, v in scores.items():
            for (dG, dE, solv, smi, tidx) in sorted(v): # default ascending (lowest to highest)
                agg.append((dG, dE, solv, smi, tidx))

        min_dG = min([x[0] for x in agg])
        # relative energy
        res = sorted([(dG-min_dG, solv, smi, tidx) for (dG, dE, solv, smi, tidx) in agg])

        return res
    

if __name__ == '__main__':
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

        enumerated = Tautomer(smi).enumerate()
        print(enumerated)
        results = TautomerOrdered(enumerated, num_confs=50).order()
        print(results)
        print()
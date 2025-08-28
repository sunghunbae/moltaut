from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from moltaut.cut_mol import get_frags
from moltaut.tautomer import enumerate_tauts
from collections import namedtuple


def is_ionic(m):
    charges = []
    for at in m.GetAtoms():
        if at.GetFormalCharge() == 0:
            charges.append(False)
        else:
            charges.append(True)
    return any(charges)


def uncharge_mol(mol):
    un = rdMolStandardize.Uncharger()
    mol = un.uncharge(mol)
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    return mol


def get_tauts(m):
    tauts = enumerate_tauts(m)
    ms = []
    for taut in tauts:
        m = taut.mol
        if not m:
            print("tautomer error: ", smi)
            continue
        ms.append(taut)
    return ms


def unique_and_split(nms):
    smis = [Chem.MolToSmiles(m) for m in nms]
    smis = list(set(smis))
    nmols, imols = [], []
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        if is_ionic(mol):
            imols.append(mol)
        else:
            nmols.append(mol)
    return nmols, imols


def get_tauts_for_vmr(smi):
    vm = Chem.MolFromSmiles(smi)
    vtauts = get_tauts(vm)
    return vtauts


def get_tauts_for_mol(m):
    mtauts = get_tauts(m)
    return mtauts


def is_vmr(vmr_tauts, mol_tauts):
    data = []
    for vm in vmr_tauts:
        labels = []
        vm_pattern = Chem.MolFromSmarts(Chem.MolToSmiles(vm))
        for mm in mol_tauts:
            res = mm.GetSubstructMatches(vm_pattern)
            if len(res) != 0:
                labels.append(1)
            else:
                labels.append(0)
        data.append(any(labels))
        if not any(labels):
            print(Chem.MolToSmiles(vm))
    
    return all(data)


def filter_vmrs(smallest_vmrs, mol_tauts):
    """
    mol_tauts: tautomers of a molecule
    """
    final_smallest_vmrs = []
    for vmr in smallest_vmrs:
        vsmi = vmr.smi
        vmr_tauts = get_tauts_for_vmr(vsmi)
        vmr = vmr._replace(tauts=vmr_tauts)
        if is_vmr(vmr_tauts, mol_tauts):
            final_smallest_vmrs.append(vmr)
    return final_smallest_vmrs


def filter_tauts_of_vmr(vmr_tauts, mol_tauts):
    vmr_tauts_filter = []
    for vm in vmr_tauts:
        labels = []
        vm_pattern = Chem.MolFromSmarts(Chem.MolToSmiles(vm))
        for mm in mol_tauts:
            res = mm.GetSubstructMatches(vm_pattern)
            if len(res) != 0:
                labels.append(1)
            else:
                labels.append(0)
        if any(labels):
            vmr_tauts_filter.append(vm)
    return vmr_tauts_filter


def enumerate_vmrs(smi):
    data = namedtuple('vmrs', 'smi tauts')

    m = Chem.MolFromSmiles(smi)
    m = uncharge_mol(m)

    frag_smis = get_frags(m)

    vmrs = []
    for fsmi in frag_smis:
        ftauts = get_tauts_for_vmr(fsmi)
        vmr = data(smi=fsmi, tauts=ftauts)
        vmrs.append(vmr)
    
    return vmrs

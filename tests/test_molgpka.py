from rdkit import Chem

from moltaut.molgpka.utils.ionization_group import get_ionization_aid
from moltaut.molgpka.predict_pka import predict
from moltaut.molgpka.protonate import protonate_mol


def test_ionization_atom_idx():
    mol = Chem.MolFromSmiles("CN(C)CCCN1C2=CC=CC=C2SC2=C1C=C(C=C2)C(C)=O")
    a, b = get_ionization_aid(mol) # ([], [1, 6])
    assert a == []
    assert b == [1,6]


def test_predict_pka():
    mol = Chem.MolFromSmiles("CN(C)CCCN1C2=CC=CC=C2SC2=C1C=C(C=C2)C(C)=O")
    base_dict, acid_dict, _ = predict(mol)
    print("base:",base_dict)
    print("acid:",acid_dict)
    # base: {9: np.float32(5.5652046), 13: np.float32(9.476429)}
    # acid: {}


def test_protonate():
    mol = Chem.MolFromSmiles("CN(C)CCCN1C2=CC=CC=C2SC2=C1C=C(C=C2)C(C)=O")
    smi = "CN(C)CCCN1C2=CC=CC=C2SC2=C1C=C(C=C2)C(C)=O"
    smi = "Nc1cc(C(F)(F)F)c(-c2cc(N3CCCC3)nc(N3CCOCC3)n2)cn1"
    pt_smis = protonate_mol(smi, ph=7.0, tph=2.5)
    print(pt_smis)
    # [
    # 'Nc1cc(C(F)(F)F)c(-c2cc(N3CCCC3)nc(N3CCOCC3)n2)cn1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc([NH+]3CCCC3)nc(N3CCOCC3)n2)cn1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc(N3CCCC3)[nH+]c(N3CCOCC3)n2)cn1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc(N3CCCC3)nc([NH+]3CCOCC3)n2)cn1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc(N3CCCC3)nc(N3CCOCC3)[nH+]2)cn1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc(N3CCCC3)nc(N3CCOCC3)n2)c[nH+]1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc([NH+]3CCCC3)[nH+]c(N3CCOCC3)n2)cn1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc([NH+]3CCCC3)nc([NH+]3CCOCC3)n2)cn1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc([NH+]3CCCC3)nc(N3CCOCC3)[nH+]2)cn1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc([NH+]3CCCC3)nc(N3CCOCC3)n2)c[nH+]1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc(N3CCCC3)[nH+]c([NH+]3CCOCC3)n2)cn1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc(N3CCCC3)[nH+]c(N3CCOCC3)[nH+]2)cn1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc(N3CCCC3)[nH+]c(N3CCOCC3)n2)c[nH+]1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc(N3CCCC3)nc([NH+]3CCOCC3)[nH+]2)cn1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc(N3CCCC3)nc([NH+]3CCOCC3)n2)c[nH+]1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc(N3CCCC3)nc(N3CCOCC3)[nH+]2)c[nH+]1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc([NH+]3CCCC3)[nH+]c([NH+]3CCOCC3)n2)cn1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc([NH+]3CCCC3)[nH+]c(N3CCOCC3)[nH+]2)cn1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc([NH+]3CCCC3)[nH+]c(N3CCOCC3)n2)c[nH+]1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc([NH+]3CCCC3)nc([NH+]3CCOCC3)[nH+]2)cn1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc([NH+]3CCCC3)nc([NH+]3CCOCC3)n2)c[nH+]1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc([NH+]3CCCC3)nc(N3CCOCC3)[nH+]2)c[nH+]1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc(N3CCCC3)[nH+]c([NH+]3CCOCC3)[nH+]2)cn1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc(N3CCCC3)[nH+]c([NH+]3CCOCC3)n2)c[nH+]1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc(N3CCCC3)[nH+]c(N3CCOCC3)[nH+]2)c[nH+]1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc(N3CCCC3)nc([NH+]3CCOCC3)[nH+]2)c[nH+]1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc([NH+]3CCCC3)[nH+]c([NH+]3CCOCC3)[nH+]2)cn1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc([NH+]3CCCC3)[nH+]c([NH+]3CCOCC3)n2)c[nH+]1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc([NH+]3CCCC3)[nH+]c(N3CCOCC3)[nH+]2)c[nH+]1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc([NH+]3CCCC3)nc([NH+]3CCOCC3)[nH+]2)c[nH+]1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc(N3CCCC3)[nH+]c([NH+]3CCOCC3)[nH+]2)c[nH+]1', 
    # 'Nc1cc(C(F)(F)F)c(-c2cc([NH+]3CCCC3)[nH+]c([NH+]3CCOCC3)[nH+]2)c[nH+]1',
    # ]

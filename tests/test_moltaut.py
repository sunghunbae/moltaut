from moltaut.cut_mol import get_frags
from moltaut.combine_frag import link_fragment
from moltaut.tautomer import enumerate_tauts
from moltaut.get_vmrs import enumerate_vmrs
from moltaut.rank_tautomer import rank_tauts
from moltaut.cli import get_lower_energy_tauts, low_energy_tautomers

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def test_enumerate_tauts():
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
        "Nc1nc(O)c2[nH]nnc2n1",
    ]:
        m = Chem.MolFromSmiles(smi)
        
        enumerator = rdMolStandardize.TautomerEnumerator()
        results_rdmols = enumerator.Enumerate(m)
        results_smiles = [Chem.MolToSmiles(_) for _ in results_rdmols]

        # MolTaut is more thorough in enumeration
        # in other words, it generates unrealistic ones as well.
        ms = enumerate_tauts(m)
        results = [Chem.MolToSmiles(Chem.MolFromSmiles(t.smi)) for t in ms]
        
        print("Input:", smi)
        for _ in results:
            if _ in results_smiles:
                print(_,'Ok')
            else:
                print(_)
        print()
        print("RDkit:")
        for _ in results_smiles:
            print(_)
        
        print()


def test_cut_mol():
    smi = "CN(C)CCCN1C2=CC=CC=C2SC2=C1C=C(C=C2)C(C)=O"
    smi = "Brc1cnn2c1nc(cc2NCc1cccnc1)c1ccccc1"
    smi = "Cc1n[nH]c(c12)OC(N)=C(C#N)C2(C(C)C)c(cc3C(F)(F)F)cc(c3)N4CCCC4"
    smi = "Nc1nc2c([nH]1)cccn2"
    smi = "c1ncccc1-c(n2)[nH]c(c23)CCCc4c3cc(F)cc4"
    smi = "Cc1c2c([nH]n1)OC(=C([C@@]2(c3cc(cc(c3)N4CCCC4)C(F)(F)F)C(C)C)C#N)N"
    mol = Chem.MolFromSmiles(smi)
    frags = get_frags(mol)
    print(frags)
    # [
    # '[*:1]C', 
    # '[*:1]c1n[nH]c2c1[C@]([*:2])([*:3])C(C#N)=C(N)O2', 
    # '[*:2]c1cc([*:4])cc(C(F)(F)F)c1', 
    # '[*:4]N1CCCC1', 
    # '[*:3]C([*:5])[*:6]', 
    # '[*:5]C', 
    # '[*:6]C',
    # ]

def test_combine_frag():
    smi1 = '[*:3]c1cc(C([*:1])([*:2])F)cc(C2(C(C)C)C(C#N)=C(N)Oc3[nH]nc(C)c32)c1'
    smi2 = '[*:1]F'
    smi3 = '[*:2]F'
    smi4 = '[*:3]N1CCCC1'
    smis = [smi3, smi2, smi1, smi4]
    m = link_fragment(smis)
    print("\nCombine Fragment...\n")
    print(Chem.MolToSmiles(m))
    print("\n")


def test_tautomer():
    #smi = "Oc1ccccc1"
    #smi = "COC(=O)c1ccc(O)cc1"
    # smi = "N#CC1=C(N)Oc2[nH]ncc2C1"
    #smi = "OSc1ncc[nH]1"
    #smi = "O=C(Cc1ccccc1)c1cccs1"
    #smi = "Oc1nc2ccccc2nc1"
    #smi = "O=C1NN=CN1"
    #smi = "OC(=O)COC(=O)N[C@]12CC[C@H](CC1)[C@@H]1[C@H]2C(=O)N(C1=O)c1ccc(cc1)NC(=O)C"
    #smi = "OCc(n1)cnc(c12)nc(N)[nH]c2=O"
    #smi = "O=[N+]([O-])c1ccc2cn[nH]c2c1"
    #smi = "c1ccc2cn[nH]c2c1"
    smi = "Nc1nc(O)c2[nH]nnc2n1"
    m = Chem.MolFromSmiles(smi)
    ms = enumerate_tauts(m)
    print(ms)
    print([t.smi for t in ms])


def test_get_vmrs(): # what is vmrs?
    smi = "COC(=O)c1ccc(O)cc1"
    #smi = "CN(C)CCCN1C2=CC=CC=C2OC2=C1C=C(C=C2)C(C)=O"
    #smi = "c1ccccc1C(C(=O)O)NC(=O)C(NC(=O)CCC(N)C(=O)O)CSCc2ccccc2"
    #smi = "Brc1cnn2c1nc(cc2NCc1cccnc1)c1ccccc1"
    #smi = "Cc1n[nH]c(c12)OC(N)=C(C#N)C2(C(C)C)c(cc3C(F)(F)F)cc(c3)N4CCCC4"
    vmrs = enumerate_vmrs(smi)
    print(vmrs)


def test_rank_tauts():
    smi = "Clc1cnn2c1nc(cc2NCc1cccnc1)c1ccccc1"
    #smi = "Cc1n[nH]c(c12)OC(N)=C(C#N)C2(C(C)C)c(cc3C(F)(F)F)cc(c3)N4CCCC4"
    #smi = "CS(=O)(=O)c1ccc(cc1)c1cccn2c1nc(n2)Nc1ccc(cc1)N1CCOCC1"
    vmrs = enumerate_vmrs(smi)
    print(vmrs)
    index = 0
    tauts = vmrs[index].tauts
    df = rank_tauts(tauts, num_confs=50)
    print(df)


def test_get_lower_energy_tauts():
    print("get_lower_energy_tauts()...")
    for smi in [
        #"Oc1ccccc1",
        #"COC(=O)c1ccc(O)cc1",
        "N#CC1=C(N)Oc2[nH]ncc2C1",
        "OSc1ncc[nH]1",
        #"O=C(Cc1ccccc1)c1cccs1",
        #"Oc1nc2ccccc2nc1",
        #"O=C1NN=CN1",
        # "OC(=O)COC(=O)N[C@]12CC[C@H](CC1)[C@@H]1[C@H]2C(=O)N(C1=O)c1ccc(cc1)NC(=O)C",
        #"OCc(n1)cnc(c12)nc(N)[nH]c2=O",
        #"O=[N+]([O-])c1ccc2cn[nH]c2c1",
        #"c1ccc2cn[nH]c2c1",
        #"Nc1nc(O)c2[nH]nnc2n1",
        ]:
        print(smi)
        results = get_lower_energy_tauts(smi, 3.0, 5)
        for res in results:
            print(res)
        print()


def test_low_energy_tautomers():
    print("Low energy tautomers ...")
    for smi in [
        #"Oc1ccccc1",
        #"COC(=O)c1ccc(O)cc1",
        "N#CC1=C(N)Oc2[nH]ncc2C1",
        "OSc1ncc[nH]1",
        #"O=C(Cc1ccccc1)c1cccs1",
        #"Oc1nc2ccccc2nc1",
        #"O=C1NN=CN1",
        # "OC(=O)COC(=O)N[C@]12CC[C@H](CC1)[C@@H]1[C@H]2C(=O)N(C1=O)c1ccc(cc1)NC(=O)C",
        #"OCc(n1)cnc(c12)nc(N)[nH]c2=O",
        #"O=[N+]([O-])c1ccc2cn[nH]c2c1",
        #"c1ccc2cn[nH]c2c1",
        #"Nc1nc(O)c2[nH]nnc2n1",
        ]:
        
        print(smi)
        results = low_energy_tautomers(smi, 
                                       cutmol=True, 
                                       num_confs=5, 
                                       energy_cutoff=2.8, 
                                       ph=7.4)
        print(results)
        print()

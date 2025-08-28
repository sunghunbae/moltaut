from moltaut.molsolv.sasa import calc_atoms_sasa, get_sasa
from moltaut.molsolv.gen_confs import gen_confs_set
from pathlib import Path
from openbabel import pybel


def test_gen_confs():
    smi = "CNCCCO"
    blocks = gen_confs_set(smi, num_confs=50)
    print(len(blocks))
    print(blocks[0])


def test_sasa(request):
    test_file = Path(request.fspath).parent.joinpath("test.sdf")
    filename = test_file.as_posix()
    pmol = next(pybel.readfile("sdf", filename))
    mol = pmol.OBMol
    datas = calc_atoms_sasa(mol)
    sasa_info = get_sasa(mol)
    print(sasa_info)
from torch.nn import functional as F
import torch
from torchdrug import data

import rdkit.Chem as Chem
from rdkit.Chem.rdchem import BondType as BT

bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
edge_feature_dims = len(bonds.keys())

# Adapted from https://github.com/gcorso/DiffDock/blob/main/datasets/process_mols.py#L126
allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)) + ["misc"],
    "possible_chirality_list": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
    ],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_numring_list": [0, 1, 2, 3, 4, 5, 6, "misc"],
    "possible_implicit_valence_list": [0, 1, 2, 3, 4, 5, 6, "misc"],
    "possible_formal_charge_list": [
        -5,
        -4,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        "misc",
    ],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": [
        "SP",
        "SP2",
        "SP3",
        "SP3D",
        "SP3D2",
        "misc",
    ],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring3_list": [False, True],
    "possible_is_in_ring4_list": [False, True],
    "possible_is_in_ring5_list": [False, True],
    "possible_is_in_ring6_list": [False, True],
    "possible_is_in_ring7_list": [False, True],
    "possible_is_in_ring8_list": [False, True],
}

# Adapted from https://github.com/gcorso/DiffDock/blob/main/datasets/process_mols.py#L126
lig_feature_dims = (
    list(
        map(
            len,
            [
                allowable_features["possible_atomic_num_list"],
                allowable_features["possible_chirality_list"],
                allowable_features["possible_degree_list"],
                allowable_features["possible_formal_charge_list"],
                allowable_features["possible_implicit_valence_list"],
                allowable_features["possible_numH_list"],
                allowable_features["possible_number_radical_e_list"],
                allowable_features["possible_hybridization_list"],
                allowable_features["possible_is_aromatic_list"],
                allowable_features["possible_numring_list"],
                allowable_features["possible_is_in_ring3_list"],
                allowable_features["possible_is_in_ring4_list"],
                allowable_features["possible_is_in_ring5_list"],
                allowable_features["possible_is_in_ring6_list"],
                allowable_features["possible_is_in_ring7_list"],
                allowable_features["possible_is_in_ring8_list"],
            ],
        )
    ),
    0,
)  # number of scalar features


# Adapted from https://github.com/gcorso/DiffDock/blob/main/datasets/process_mols.py#L126
def safe_index(l, e):
    """Return index of element e in list l. If e is not present, return the last index"""
    try:
        return l.index(e)
    except:
        return len(l) - 1


# Adapted from https://github.com/gcorso/DiffDock/blob/main/datasets/process_mols.py#L126
def featurizer_advanced(data):
    if isinstance(data, str):
        mol = Chem.MolFromSmiles(data)
    else:
        mol = data

    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        atom_features_list.append(
            [
                safe_index(
                    allowable_features["possible_atomic_num_list"],
                    atom.GetAtomicNum(),
                ),
                allowable_features["possible_chirality_list"].index(
                    str(atom.GetChiralTag())
                ),
                safe_index(
                    allowable_features["possible_degree_list"],
                    atom.GetTotalDegree(),
                ),
                safe_index(
                    allowable_features["possible_formal_charge_list"],
                    atom.GetFormalCharge(),
                ),
                safe_index(
                    allowable_features["possible_implicit_valence_list"],
                    atom.GetImplicitValence(),
                ),
                safe_index(
                    allowable_features["possible_numH_list"],
                    atom.GetTotalNumHs(),
                ),
                safe_index(
                    allowable_features["possible_number_radical_e_list"],
                    atom.GetNumRadicalElectrons(),
                ),
                safe_index(
                    allowable_features["possible_hybridization_list"],
                    str(atom.GetHybridization()),
                ),
                allowable_features["possible_is_aromatic_list"].index(
                    atom.GetIsAromatic()
                ),
                safe_index(
                    allowable_features["possible_numring_list"],
                    ringinfo.NumAtomRings(idx),
                ),
                allowable_features["possible_is_in_ring3_list"].index(
                    ringinfo.IsAtomInRingOfSize(idx, 3)
                ),
                allowable_features["possible_is_in_ring4_list"].index(
                    ringinfo.IsAtomInRingOfSize(idx, 4)
                ),
                allowable_features["possible_is_in_ring5_list"].index(
                    ringinfo.IsAtomInRingOfSize(idx, 5)
                ),
                allowable_features["possible_is_in_ring6_list"].index(
                    ringinfo.IsAtomInRingOfSize(idx, 6)
                ),
                allowable_features["possible_is_in_ring7_list"].index(
                    ringinfo.IsAtomInRingOfSize(idx, 7)
                ),
                allowable_features["possible_is_in_ring8_list"].index(
                    ringinfo.IsAtomInRingOfSize(idx, 8)
                ),
            ]
        )

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += (
            2 * [bonds[bond.GetBondType()]]
            if bond.GetBondType() != BT.UNSPECIFIED
            else [0, 0]
        )

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds))
    return edge_index, edge_attr, torch.tensor(atom_features_list)


def featurizer_simple(smile):
    mol = data.Molecule.from_smiles(smile)
    edge_index, edge_type = (
        torch.tensor(mol.edge_list[:, :2].T),
        mol.edge_list[:, 2:].squeeze(),
    )
    x = torch.tensor(
        F.one_hot(
            mol.atom_type,
            num_classes=lig_feature_dims[0],
            dtype=torch.float32,
        )
    )
    edge_attr = torch.tensor(
        F.one_hot(edge_type, num_classes=len(bonds)),
        dtype=torch.float32,
    )

    return edge_index, edge_attr, x

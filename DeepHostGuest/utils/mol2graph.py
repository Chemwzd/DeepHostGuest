import numpy as np
from rdkit.Chem import rdMolTransforms
import rdkit.Chem as Chem

att_dtype = np.float32

import networkx as nx


def oneHotVector(val, lst):
    '''Converts a value to a one-hot vector based on options in lst'''
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)


def mol_to_nx(mol, removeHs=False):
    G = nx.Graph()

    # Globals.
    G.graph["features"] = np.array([None], dtype=np.float32)
    atomCoords = mol.GetConformer().GetPositions()
    if removeHs:
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum == 1:
                continue
            G.add_node(atom.GetIdx(),
                       pos=atomCoords[i],
                       x=np.array(
                           list(oneHotVector(atom.GetAtomicNum(), [5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53])) + [
                               atom.GetFormalCharge()],
                           dtype=np.float32))

        for bond in mol.GetBonds():
            if mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum() == 1:
                continue
            if mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum() == 1:
                continue
            isConjugated = 1 if bond.GetIsConjugated() and not bond.GetIsAromatic() else 0
            isAromatic = 1 if bond.GetIsAromatic() is True else 0
            G.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       edge_attr=np.array(
                           list(oneHotVector(bond.GetBondTypeAsDouble(), [1.0, 1.5, 2.0, 3.0, 99.0])) + [isConjugated,
                                                                                                         isAromatic],
                           dtype=np.float32))
    else:
        for i, atom in enumerate(mol.GetAtoms()):
            G.add_node(atom.GetIdx(),
                       pos=atomCoords[i],
                       x=np.array(
                           list(oneHotVector(atom.GetAtomicNum(), [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53])) + [
                               atom.GetFormalCharge()],
                           dtype=np.float32))

        for bond in mol.GetBonds():
            isConjugated = 1 if bond.GetIsConjugated() and not bond.GetIsAromatic() else 0
            isAromatic = 1 if bond.GetIsAromatic() is True else 0
            G.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       edge_attr=np.array(
                           list(oneHotVector(bond.GetBondTypeAsDouble(), [1.0, 1.5, 2.0, 3.0, 99.0])) + [isConjugated,
                                                                                                         isAromatic],
                           dtype=np.float32))
    return G


def get_bonds(mol_list, bidirectional=True):
    atom_counter = 0
    bonds = []
    dist = []
    for m in mol_list:
        for x in m.GetBonds():
            conf = m.GetConformer()
            bonds.extend([(x.GetBeginAtomIdx() + atom_counter, x.GetEndAtomIdx() + atom_counter)])
            dist.extend([rdMolTransforms.GetBondLength(conf, x.GetBeginAtomIdx(), x.GetEndAtomIdx())])
            if bidirectional:
                bonds.extend([(x.GetEndAtomIdx() + atom_counter, x.GetBeginAtomIdx() + atom_counter)])
                dist.extend([rdMolTransforms.GetBondLength(conf, x.GetEndAtomIdx(), x.GetBeginAtomIdx())])
        atom_counter += m.GetNumAtoms()
    return bonds, dist


def get_angles(mol_list, bidirectional=True):
    atom_counter = 0
    bendList = []
    angleList = []
    for m in mol_list:
        bendSmarts = '*~*~*'
        bendQuery = Chem.MolFromSmarts(bendSmarts)
        matches = m.GetSubstructMatches(bendQuery)
        conf = m.GetConformer()
        for match in matches:
            idx0 = match[0]
            idx1 = match[1]
            idx2 = match[2]
            bendList.append((idx0 + atom_counter, idx1 + atom_counter, idx2 + atom_counter))
            angleList.append(rdMolTransforms.GetAngleRad(conf, idx0, idx1, idx2))
            if bidirectional:
                bendList.append((idx2 + atom_counter, idx1 + atom_counter, idx0 + atom_counter))
                angleList.append(rdMolTransforms.GetAngleRad(conf, idx2, idx1, idx0))
        atom_counter += m.GetNumAtoms()
    return bendList, angleList


def get_torsions(mol_list, bidirectional=True):
    atom_counter = 0
    torsionList = []
    dihedralList = []
    for m in mol_list:
        torsionSmarts = '[!$(*#*)&!D1]~[!$(*#*)&!D1]'
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = m.GetSubstructMatches(torsionQuery)
        conf = m.GetConformer()
        for match in matches:
            idx2 = match[0]
            idx3 = match[1]
            bond = m.GetBondBetweenAtoms(idx2, idx3)
            jAtom = m.GetAtomWithIdx(idx2)
            kAtom = m.GetAtomWithIdx(idx3)
            if (((jAtom.GetHybridization() != Chem.HybridizationType.SP2)
                 and (jAtom.GetHybridization() != Chem.HybridizationType.SP3))
                    or ((kAtom.GetHybridization() != Chem.HybridizationType.SP2)
                        and (kAtom.GetHybridization() != Chem.HybridizationType.SP3))):
                continue
            for b1 in jAtom.GetBonds():
                if (b1.GetIdx() == bond.GetIdx()):
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in kAtom.GetBonds():
                    if ((b2.GetIdx() == bond.GetIdx())
                            or (b2.GetIdx() == b1.GetIdx())):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    # skip 3.use_deepdock-membered rings
                    if (idx4 == idx1):
                        continue
                    torsionList.append(
                        (idx1 + atom_counter, idx2 + atom_counter, idx3 + atom_counter, idx4 + atom_counter))
                    dihedralList.append(rdMolTransforms.GetDihedralRad(conf, idx1, idx2, idx3, idx4))
                    if bidirectional:
                        torsionList.append(
                            (idx4 + atom_counter, idx3 + atom_counter, idx2 + atom_counter, idx1 + atom_counter))
                        dihedralList.append(rdMolTransforms.GetDihedralRad(conf, idx4, idx3, idx2, idx1))
        atom_counter += m.GetNumAtoms()
    return torsionList, dihedralList


def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
    return mol


def atomenvironments(mol, radius=3):
    envs = []
    for a in mol.GetAtoms():
        idx = a.GetIdx()
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, idx)
        amap = {}
        submol = Chem.PathToSubmol(mol, env, atomMap=amap)
        if amap.get(idx) is not None:
            envs.append(Chem.MolToSmarts(submol))
    return envs

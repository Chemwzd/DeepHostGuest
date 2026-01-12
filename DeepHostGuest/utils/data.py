import torch
from plyfile import PlyData
import torch_geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_networkx
from torch_geometric.transforms import FaceToEdge, Cartesian

from sklearn.cluster import AgglomerativeClustering
from rdkit import Chem
import os

from DeepHostGuest.utils import mol2graph


def read_ply(path):
    with open(path, 'rb') as f:
        data = PlyData.read(f)

    features = ([torch.tensor(data['vertex'][axis.name]) for axis in data['vertex'].properties if
                 axis.name not in ['nx', 'ny', 'nz']])
    pos = torch.stack(features[:3], dim=-1)
    features = torch.stack(features[3:], dim=-1)

    face = None
    if 'face' in data:
        faces = data['face']['vertex_indices']
        faces = [torch.tensor(fa, dtype=torch.long) for fa in faces]
        face = torch.stack(faces, dim=-1)

    data = Data(x=features, pos=pos, face=face)

    return data


class HostGuest_dataset(Dataset):
    def __init__(self, root, removeHs=True, transform=None, pre_transform=None):
        self.removeHs = removeHs
        super(HostGuest_dataset, self).__init__(root, transform, pre_transform)
        self.processed_folder = './data/processed'
        self.root = root
        data_path = os.path.join(self.processed_dir, f'data.pt')
        self.data_list = torch.load(data_path)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        host_path = os.path.join(self.root, "host_ply")
        host_data = self.read_ply_files(folder_path=host_path)
        prefixes = [i.rstrip('.ply') for i in host_data]

        # Pair host and guest data and obtain the mol data
        paired_data = [(f"{prefix}.ply", f"{prefix.replace('_1_', '_2_')}.mol") for prefix in prefixes]
        graph_data = self.mol_to_graph(paired_data)
        data_list = graph_data

        if self.pre_filter is not None:
            data_list = [data for data in self.data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in self.data_list]

        torch.save(data_list, self.processed_paths[0])

    @staticmethod
    def read_ply_files(folder_path=''):
        ply_files = [file for file in os.listdir(folder_path) if file.endswith('.ply')]
        return ply_files

    def mol_to_graph(self, mol_lists):

        guest_list = []
        host_list = []

        guest_path = os.path.join(self.root, "guest_mol")
        host_path = os.path.join(self.root, "host_ply")

        for mols in mol_lists:
            host_file, guest_file = mols
            # print(f'Reading {host_file}')
            hostmol_path = os.path.join(host_path, host_file)
            guestmol_path = os.path.join(guest_path, guest_file)
            guest = Chem.MolFromMolFile(guestmol_path, removeHs=self.removeHs, sanitize=True)

            guest_graph = mol2graph.mol_to_nx(guest, self.removeHs)
            guest_data = from_networkx(guest_graph)
            guest_list.append(guest_data)

            mesh = read_ply(hostmol_path)
            host_data = FaceToEdge()(mesh)
            host_data = Cartesian()(host_data)
            host_list.append(host_data)
            # print(f'Done for {host_file}')

        paired_graph = [(host, guest) for host, guest in zip(host_list, guest_list)]
        return paired_graph

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def compute_clusters(data, n_clusters):
    X = data[1].pos.numpy()
    connectivity = torch_geometric.utils.to_scipy_sparse_matrix(data[1].edge_index)

    if len(X) > n_clusters:
        ward = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity, linkage='ward').fit(X)
        data[1].clus = torch.tensor(ward.labels_)
    else:
        data[1].clus = torch.tensor(range(len(X)))

    return data


def compute_cluster_batch_index(cluster, batch):
    max_prev_batch = 0
    for i in range(batch.max().item() + 1):
        cluster[batch == i] += max_prev_batch
        max_prev_batch = cluster[batch == i].max().item() + 1
    return cluster


def Mol2MolSupplier(file=None, sanitize=True, cleanupSubstructures=True):
    # Taken from https://chem-workflows.com/articles/2020/03/23/building-a-multi-molecule-mol2-reader-for-rdkit-v2/
    mols = []
    with open(file, 'r') as f:
        doc = [line for line in f.readlines()]

    start = [index for (index, p) in enumerate(doc) if '@<TRIPOS>MOLECULE' in p]
    finish = [index - 1 for (index, p) in enumerate(doc) if '@<TRIPOS>MOLECULE' in p]
    finish.append(len(doc))

    try:
        name = [doc[index].rstrip().split('\t')[-1] for (index, p) in enumerate(doc) if 'Name' in p]
    except:
        pass

    interval = list(zip(start, finish[1:]))
    for n, i in enumerate(interval):
        block = ",".join(doc[i[0]:i[1]]).replace(',', '')
        m = Chem.MolFromMol2Block(block, sanitize=sanitize, cleanupSubstructures=cleanupSubstructures)
        if m is not None:
            if name:
                m.SetProp('Name', name[n])
            mols.append(m)

    return (mols)

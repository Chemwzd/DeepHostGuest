"""
Perform data augmentation on the host-guest structure through molecular dynamics.
The process can be executed step by step, following the provided example indices.

Example:
-------------------------------------------------
    import os
    from DeepDockHostGuest.preprocessing.run_multisim import *
    import multiprocessing
    import functools
    from tqdm import tqdm

    run_multisim = RunMultisim(schrodinger='/path/to/your/schrodinger2024-1',
                               obabel='/path/to/your/obabel')

    1. Perform MD through 'multisim' in Schrodinger:

        msjdir = "/path/to/msj"
        structdir = "/path/to/example_strucs"
        outdir = "/path/to/out_dir"
        job_names = [i.split("_")[0] for i in os.listdir(structdir) if i.endswith("_3.mol")]

        partial_run_all = functools.partial(run_multisim.run_multisim, msjdir=msjdir, structdir=structdir, outdir=outdir, verbose=True)

        with multiprocessing.Pool(processes=10) as pool:
            pool.map(partial_run_all, job_names)


    2.train_deepdock. Perform trajectory clustering using $SCHRODINGER/run trj_cluster.py

        path = '/path/to/all_your_trajectories' # generate_work_dir() can copy a new path to perform clustering.
                                                # It is recommended to run generate_work_dir() first.
        names = os.listdir(path)
        results_dict = {}
        with multiprocessing.Pool(processes=12) as pool:
            results = list(tqdm(pool.imap(run_multisim.trj_cluster,
                                          [(os.path.join(path, name, f'{name}_md_out.cms'),
                                            os.path.join(path, name, f'{name}_md_trj'),
                                            os.path.join(path, name, 'clusters', f'{name}_md'),
                                            1000) for
                                           name in
                                           names]),
                                total=len(names)))
            for result in results:
                outname, output, error = result
                results_dict[outname] = [output, error]
                if error:
                    print(f"++++++++++ERROR {outname} ERROR++++++++++")
                    print(error)


    3.use_deepdock. Extract structures from each cluster into .pbd files

        # basedir contains all of the jobname directories(basedir/jobname/clusters/jobname_members-out.cms)
        basedir = "/path/to/your/basedir"
        # directory contains each frame in .pdb format (out_basedir/jobname/cluster_n/jobname_xxx.pdb)
        out_basedir = "/path/to/your/outdir"
        job_names = [i for i in os.listdir(basedir)]

        for job_name in tqdm(job_names):
            print(f"Processing {job_name}")
            if os.path.exists(os.path.join(out_basedir, job_name)):
                print(f"{job_name} has exists!\n")
                continue
            def process_job(job):
                print(f"Processing {job}")
                cms_file = os.path.join(basedir, job_name, 'clusters', f"{job}-out.cms")
                trj_file = os.path.join(basedir, job_name, 'clusters', f"{job}_trj/")
                outfilename = os.path.join(out_basedir, job_name, job, f"{job}")
                os.makedirs(os.path.join(out_basedir, job_name), exist_ok=True)
                os.makedirs(os.path.join(out_basedir, job_name, job), exist_ok=True)
                out, errors = run_multisim.convert_cms_to_pdb(cms_file, trj_file, outfilename)
                if errors:
                    print(f"error for {job} is: {errors}\n")
                print(f"Job {job} finished successfully.")

            cluster_job_names = [i for i in os.listdir(os.path.join(basedir, job_name, 'clusters')) if i.endswith('.cms')]
            cluster_job_names = [i.split('-out')[0] for i in cluster_job_names]

            with multiprocessing.Pool(processes=32) as pool:
                pool.map(process_job, cluster_job_names)

    4. Extract one structure from each cluster and convert .pdb to .mol file.

        # basedir contains all of the jobname directories(basedir/jobname/clusters_pdb)
        basedir = "/path/to/your/basedir"
        # directory contains each frame in .mol format (out_basedir/jobname/jobname_xxx.pdb)
        out_basedir = '/path/to/your/outdir'
        os.makedirs(out_basedir, exist_ok=True)
        job_names = [i for i in os.listdir(basedir)]

        for job_name in tqdm(job_names):
            if os.path.exists(os.path.join(out_basedir, job_name)):
                print(f"{job_name} has exists!\n")
                continue


            def process_job(job):
                num_pdb = int(job.split('_')[-1].rstrip('members'))
                random_index = random.randint(0, num_pdb - 1)
                pdb_file = os.path.join(basedir, job_name, job, f"{job}_{random_index}.pdb")
                mol_file = os.path.join(out_basedir, job_name, f"{job}_{random_index}.mol")
                os.makedirs(os.path.join(out_basedir, job_name), exist_ok=True)
                out, errors = run_multisim.convert_pdb_to_mol(pdb_file, mol_file)


            cluster_job_names = os.listdir(os.path.join(basedir, job_name))
            with multiprocessing.Pool(processes=32) as pool:
                pool.map(process_job, cluster_job_names)


    5. Generate a name-atom_number dictionary from origin host-guest structure files.

        generate_number_dict('/path/to/your/RefStructures',
                     '/path/to/your/NumberDict.json')


    6. Recenter the .mol files to [0, 0, 0] and split it into separated host and guest file.

        basedir = '/path/to/your/step4_outdir'
        names = os.listdir(basedir)
        out_path = '/path/to/your/step6_outdir'
        number_dict_path = '/path/to/your/NumberDict.json'
        with open(number_dict_path, 'r') as f:
            number_dict = json.load(f)
        os.makedirs(os.path.join(out_path, 'Host'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'Guest'), exist_ok=True)
        for name in tqdm(names):
            names_list = os.listdir(os.path.join(basedir, name))
            with multiprocessing.Pool(processes=32) as pool:
                for f in names_list:
                    pool.apply_async(split_host_guest,
                                     args=(os.path.join(basedir, name, f), name, number_dict, out_path))
                pool.close()
                pool.join()

"""
import os
import time
import shutil
import json
from sugar.molecule import HostMolecule
import numpy as np
import subprocess
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import Counter


def run_command(command):
    """
    Return the state of command line.
    """
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    return output, error


def generate_work_dir(infile_path, out_path):
    """
    Generate a copied workdir for step 2.train_deepdock.
    Only the files in need will be copied to {out_path}.
    """
    if os.path.exists(out_path):
        print(f"Working Directory has been Generated, skip function::generate_work_dir()\n")
        return None
    os.makedirs(out_path)
    for direc in os.listdir(infile_path):
        if f'{direc}_md_trj' in os.listdir(os.path.join(infile_path, direc)):
            os.makedirs(os.path.join(out_path, direc), exist_ok=True)
            # os.makedirs(os.path.join(out_path, direc, 'clusters'), exist_ok=True)
            for file in os.listdir(os.path.join(infile_path, direc)):
                if file.endswith('md_out.cms'):
                    shutil.copy(os.path.join(infile_path, direc, file), os.path.join(out_path, direc))
                elif file.endswith('_md_trj'):
                    shutil.copytree(os.path.join(infile_path, direc, file), os.path.join(out_path, direc, file))
        else:
            continue
    return None


def contains_string_or_substring(lst, target):
    for i in lst:
        if isinstance(i, str) and target in i:
            return True
    return False


def check_run_multisim(outdir):
    not_complete = []
    for directory in os.listdir(outdir):
        if os.path.isdir(directory):
            out_file = os.path.join(outdir, directory, f"{directory}_md_out.cms")
            if not os.path.exists(out_file):
                not_complete.append(directory)
    return not_complete


def get_charge_list(mol):
    charge_list = []
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            charge_list.append(f"{atom.GetSymbol()}{atom.GetFormalCharge()}")
    charge_counts = Counter(charge_list)
    return charge_counts


def generate_number_dict(mol_folder, json_path):
    """
    In the given folder, execute get_host_guest_atom_number for each molecule file.
    In the form of: {'RefCode': [NumberOfHostAtoms, NumberOfGuestAtoms]}
    """
    names = os.listdir(mol_folder)
    names = set([name.split('_')[0] for name in names])
    os.chdir(mol_folder)
    number_dict = {}
    for name in names:
        guest = AllChem.MolFromMolFile(f"{name}_2.mol", removeHs=False, sanitize=False)
        host = AllChem.MolFromMolFile(f"{name}_1.mol", removeHs=False, sanitize=False)
        guest_number = guest.GetNumAtoms()
        host_number = host.GetNumAtoms()
        guest_charge = get_charge_list(guest)
        host_charge = get_charge_list(host)
        number_dict[name] = {'HostNumber': host_number, 'GuestNumber': guest_number,
                             'HostCharge': host_charge, 'GuestCharge': guest_charge}
    with open(json_path, 'w') as f:
        json.dump(number_dict, f)


def generate_mol_block(atoms, bonds, positions, title='    ', reset_index=True):
    blank = '  0  0  0  0  0  0  0  0  0  0  0  0'
    counts_line = "{:3}{:3}  0  0  0  0  0  0  0  0999 V2000".format(len(atoms), len(bonds))
    mol_block = [title, '     HostGuestDataset          3D', '', counts_line]
    index_reset_dict = {}
    charge_block = ['M  CHG', 0]
    for i in range(len(atoms)):
        atom_block = positions[i]
        atom_block = [p.round(4) for p in atom_block]
        atom_block.append(f' {atoms[i].GetSymbol()}')
        atom_block.append(blank)
        atom_block = [
            '{:>10}'.format(str(atom_block[0])),
            '{:>10}'.format(str(atom_block[1])),
            '{:>10}'.format(str(atom_block[2])),
            '{:<3}'.format(str(atom_block[3])),
            atom_block[4]
        ]
        mol_block.append(''.join(atom_block))
        if reset_index:
            index_reset_dict[atoms[i].GetIdx()] = i + 1
        else:
            index_reset_dict[atoms[i].GetIdx()] = atoms[i].GetIdx()
        if atoms[i].GetFormalCharge() != 0:
            charge_block[1] += 1
            charge_block.append(index_reset_dict[atoms[i].GetIdx()])
            charge_block.append(atoms[i].GetFormalCharge())
    for j in range(len(bonds)):
        bond_type = int(bonds[j].GetBondTypeAsDouble())
        if reset_index:
            idx1 = bonds[j].GetBeginAtomIdx()
            idx2 = bonds[j].GetEndAtomIdx()
            idx1 = index_reset_dict[idx1]
            idx2 = index_reset_dict[idx2]
        else:
            idx1 = bonds[j].GetBeginAtomIdx() + 1
            idx2 = bonds[j].GetEndAtomIdx() + 1
        bond_block = ['{:>3}'.format(str(idx1)),
                      '{:>3}'.format(str(idx2)),
                      '{:>3}'.format(str(bond_type)),
                      '{:>3}'.format(str(0))]
        mol_block.append(''.join(bond_block))
    if charge_block[1] > 0:
        formatted_charge_block = [
            str(charge_block[0]),
            '{:>3}'.format(str(charge_block[1]))
        ]
        for block in charge_block[2:]:
            formatted_charge_block.append('{:>4}'.format(str(block)))
        mol_block.append(''.join(formatted_charge_block))
    mol_block.append('M  END')
    mol_block = [i + '\n' for i in mol_block]
    return mol_block


def get_disconnected_mol(mol, recenter=True):
    """
    Extract Disconnected Molecules from a Give Mol Object.
    """
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx())
    for bond in mol.GetBonds():
        e = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        graph.add_edge(*e)
    # return graph
    mols = []
    all_positions = mol.GetConformer().GetPositions()
    non_h_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() != 'H']
    non_h_coordinates = [mol.GetConformer().GetAtomPosition(atom.GetIdx()) for atom in non_h_atoms]
    center_of_mass = np.mean(non_h_coordinates, axis=0)

    for component in nx.connected_components(graph):
        atoms = [atom for atom in mol.GetAtoms() if atom.GetIdx() in component]
        # To compare the charge information in the number_dict
        # and determine if there are differences between the structure and the reference structure.
        atom_charges = [atom.GetSymbol() + str(atom.GetFormalCharge()) for atom in atoms if atom.GetFormalCharge() != 0]
        charge_counters = Counter(atom_charges)
        bonds = []
        for bond in mol.GetBonds():
            if bond.GetBeginAtomIdx() in component and bond.GetEndAtomIdx() in component:
                bonds.append(bond)
        # Get the positions of the atoms in the current component
        positions = [all_positions[atom.GetIdx()] for atom in atoms]
        if recenter:
            positions -= center_of_mass
        mols.append([atoms, bonds, positions, charge_counters])
    return mols


def split_host_guest(filepath, name, number_dict, outpath, recenter=True):
    mol = AllChem.MolFromPDBFile(filepath, sanitize=False, removeHs=False)
    file_header = filepath.split('/')[-1].split('.')[0]
    host_name = os.path.join(outpath, 'Host', name, f'{file_header}_host.mol')
    guest_name = os.path.join(outpath, 'Guest', name, f'{file_header}_guest.mol')
    if os.path.exists(host_name) and os.path.exists(guest_name):
        return None
    components = get_disconnected_mol(mol, recenter=recenter)
    host_number = number_dict[name]['HostNumber']
    guest_number = number_dict[name]['GuestNumber']
    host_charge = number_dict[name]['HostCharge']
    guest_charge = number_dict[name]['GuestCharge']
    if len(components[0][0]) == host_number and len(components[1][0]) == guest_number:
        host_info, guest_info = components[0], components[1]
    elif len(components[1][0]) == host_number and len(components[0][0]) == guest_number:
        host_info, guest_info = components[1], components[0]
    else:
        print(f"======Wrong Atom Number of {filepath.split('/')[-1]}======\n")
        raise ValueError(f"======Wrong Atom Number of {filepath.split('/')[-1]}======\n")
    if host_info[3] != host_charge or guest_info[3] != guest_charge:
        print(f"======Error Atom Charge for {filepath.split('/')[-1]}======\n")
        raise ValueError(f"======Error Atom Charge for {filepath.split('/')[-1]}======\n")
    host_block = generate_mol_block(host_info[0], host_info[1], host_info[2])
    guest_block = generate_mol_block(guest_info[0], guest_info[1], guest_info[2])
    with open(host_name, 'w') as f1:
        f1.writelines(host_block)
    with open(guest_name, 'w') as f2:
        f2.writelines(guest_block)
    return None


def cal_rmsd(molfile1, molfile2, rmsd_path):
    output, error = run_command(
        f"{rmsd_path} --no-hydrogen -r Kabsch {molfile1} {molfile2} "
    )
    return output, error


class RunMultisim:
    def __init__(self, schrodinger, obabel=None):
        self.schrodinger = schrodinger
        self.obabel = obabel

    def convert_mol_to_mae(self, mol_file, mae_file):
        output, error = run_command(f"{self.schrodinger}/utilities/structconvert {mol_file} {mae_file}")
        return output, error

    def check_out_file(self, out_file, wait_time=5):
        start_time = time.time()
        while True:
            if os.path.isfile(out_file):
                return True
            elif time.time() - start_time > wait_time:
                print("Timeout reached. Exiting loop.")
                return False
            else:
                time.sleep(1)  # Check the outfile every 1 second

    def system_builder(self, job_name, msjdir, mae_file, outdir, waittime=150):
        outfile_name = os.path.join(outdir, job_name, f'{job_name}_setup.cms')
        if os.path.exists(outfile_name):
            print(f"SYSTEM BUILDER: {job_name} has finished, skip!")
            return True
        else:
            sys_output, sys_error = run_command(
                f"{self.schrodinger}/utilities/multisim -HOST localhost -maxjob 10 -JOBNAME {job_name}_setup -m {msjdir}/desmond_setup.msj {mae_file} -o "
                f"{outfile_name}")
            timeout = time.time() + waittime
            while True:
                log_file = os.path.join(outdir, job_name, f'{job_name}_setup_multisim.log')
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        log_contents = f.readlines()
                    if contains_string_or_substring(log_contents, 'Multisim failed'):
                        print(f"======Setup job {job_name} failed.======")
                        print(f"Output in Setup: {sys_output.decode()}",
                              f"Error in Setup: {sys_error.decode()}\n")
                        return None
                    elif contains_string_or_substring(log_contents, 'Multisim completed'):
                        print(f"Setup job {job_name} completed.")
                        return True
                if time.time() > timeout:
                    print(f"Setup ======{job_name}====== Timeout Reached.\n")
                    return None
                time.sleep(1)

    def minimization(self, job_name, msjdir, outdir, waittime=900):
        outfile_name = os.path.join(outdir, job_name, f'{job_name}_min_out.cms')
        if os.path.exists(outfile_name):
            print(f"MINIMIZATION: {job_name} has finished, skip!")
            return True
        else:
            min_output, min_error = run_command(
                f"{self.schrodinger}/utilities/multisim -VIEWNAME desmond_minimization_gui.MiniApp -JOBNAME {job_name}_min "
                f"-HOST localhost -maxjob 10 -cpu 1 -m {msjdir}/desmond_min_job.msj -c {msjdir}/desmond_min_job.cfg "
                f"-description Minimization {os.path.join(outdir, job_name, f'{job_name}_setup.cms')} -mode umbrella "
                f"-o {outfile_name} -lic DESMOND_GPGPU:16")
            timeout = time.time() + waittime
            while True:
                log_file = os.path.join(outdir, job_name, f'{job_name}_min_multisim.log')
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        log_contents = f.readlines()
                    if contains_string_or_substring(log_contents, 'Multisim failed'):
                        print(f"======Minimization job {job_name} failed.======")
                        print(f"Output in Minimization: {min_output.decode()}",
                              f"Error in Minimization: {min_error.decode()}\n")
                        return None
                    elif contains_string_or_substring(log_contents, 'Multisim completed'):
                        print(f"Minimization job {job_name} completed.")
                        return True
                if time.time() > timeout:
                    print(f"Minimization ======{job_name}====== Timeout Reached.\n")
                    return None
                time.sleep(1)

    def molecular_dynamics(self, job_name, msjdir, outdir, waittime=1800):
        outfile_name = os.path.join(outdir, job_name, f'{job_name}_md_out.cms')
        if os.path.exists(outfile_name):
            print(f"MOLECULAR DYNAMICS: {job_name} has finished, skip!")
            return True
        else:
            md_output, md_error = run_command(
                f"{self.schrodinger}/utilities/multisim -VIEWNAME desmond_molecular_dynamics_gui.MDApp "
                f"-JOBNAME {job_name}_md -HOST localhost -maxjob 10 -cpu 1 -m {msjdir}/desmond_md_job.msj "
                f"-c {msjdir}/desmond_md_job.cfg -description 'Molecular Dynamics' {os.path.join(outdir, job_name, f'{job_name}_min_out.cms')} "
                f"-mode umbrella -o {outfile_name} -lic DESMOND_GPGPU:16")
            timeout = time.time() + waittime
            while True:
                log_file = os.path.join(outdir, job_name, f'{job_name}_md_multisim.log')
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        log_contents = f.readlines()
                    if contains_string_or_substring(log_contents, 'Multisim failed'):
                        print(f"======Molecular Dynamics job {job_name} failed.======")
                        print(f"Output in Molecular Dynamics: {md_output.decode()}",
                              f"Error in Molecular Dynamics: {md_error.decode()}\n")
                        return None
                    elif contains_string_or_substring(log_contents, 'Multisim completed'):
                        print(f"Molecular Dynamics job {job_name} completed.")
                        return True
                if time.time() > timeout:
                    print(f"Molecular Dynamics ======{job_name}====== Timeout Reached.\n")
                    return None
                time.sleep(1)

    def run_multisim(self, job_name, msjdir, structdir, outdir):
        """
        :param job_name: name of the simulation job
        :param msjdir: directory to contain the .msj files
        :param structdir: directory to contain the structure files (in .mol format)
        :param outdir: directory to contain the schrodinger output files
        """
        mol_file = os.path.join(structdir, f'{job_name}_3.mol')
        if not os.path.exists(os.path.join(outdir, job_name)):
            os.makedirs(os.path.join(outdir, job_name), exist_ok=True)
        os.chdir(os.path.join(outdir, job_name))
        mae_file = os.path.join(outdir, job_name, f'{job_name}.mae')
        conv_output, conv_error = self.convert_mol_to_mae(mol_file, mae_file)
        if conv_error:
            print(f"convert error: {conv_error.decode()}")
            return None
        # Running system builder.
        sys_builder = self.system_builder(job_name, msjdir, mae_file, outdir)
        if not sys_builder:
            return None
        minimization = self.minimization(job_name, msjdir, outdir)
        if not minimization:
            return None
        molecular_dynamics = self.molecular_dynamics(job_name, msjdir, outdir)
        if not molecular_dynamics:
            return None
        return True

    def trj_cluster(self, iter_list):
        """
        'out_name' should follow the form provided in the script '3.use_deepdock.ClusterTrajs.py'
        """
        cms_file, trj_file, out_name, num_clusters = iter_list[0], iter_list[1], iter_list[2], iter_list[3]
        check_path = out_name.split('/')
        if os.path.exists('/'.join(check_path[:-1])):
            print(f"Cluster of =={check_path[-3]}== has been done, skip")
            return out_name, None, None
        else:
            os.makedirs('/'.join(check_path[:-1]))
            print(f"Cluster of {check_path[-3]}")
            output, error = run_command(
                f"{self.schrodinger}/run trj_cluster.py {cms_file} {trj_file} {out_name} -rmsd-asl all -n {num_clusters} -split-trj")
            return out_name, output, error

    def convert_cms_to_pdb(self, cms_file, trj_file, outfilename):
        output, error = run_command(
            f"{self.schrodinger}/run trj2mae.py {cms_file} "
            f"{trj_file} {outfilename} -extract-asl all -separate -out-format PDB -align-asl all")
        return output, error

    def convert_pdb_to_mol(self, pdb_file, mol_file):
        output, error = run_command(f"{self.obabel} {pdb_file} -O {mol_file}")
        return output, error

    def macromodel_mini(self, job_name, mae_file, com_file, out_dir):
        if os.path.exists(out_dir):
            print(f"{job_name} has been done, skip")
        else:
            os.makedirs(out_dir)
        copied_mae = shutil.copy(mae_file, out_dir)

        copied_mae_file = os.path.abspath(copied_mae)
        used_com_file = os.path.join(out_dir, job_name + '.com')
        out_file_name = os.path.join(out_dir, job_name + '-out.maegz')

        with open(com_file, 'r') as f:
            mm_mini_com_file_contents = f.readlines()
        mm_mini_com_file_contents.insert(0, out_file_name + '\n')
        mm_mini_com_file_contents.insert(0, copied_mae_file + '\n')

        with open(used_com_file, 'w') as f:
            f.writelines(mm_mini_com_file_contents)

        output, error = run_command(
            f"{self.schrodinger}/bmin {job_name}"
        )
        return output, error
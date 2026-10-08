from DeepHostGuest.data_augmentation.run_multisim import run_command
import shutil
import os


class ESP:
    """
    Generate molden.input and convert it into mesh file.

    Example: (see /examples/2.DataAugmentation)
    -------------------------------------------------
    1. Execute xtb to generate molden.input
        from DeepHostGuest.data_augmentation.generate_mesh import *
        from tqdm import tqdm

        xtb = ESP('/path/to/xtb', '/path/to/Multiwfn)
        host_path = '/path/to/Host'
        guest_path = '/path/to/Guest'
        names = os.listdir(host_path)
        outdir = '/path/to/xtb_output'

        for name in tqdm(names):
            print(f"======Processing {name}======")
            host_files = [i for i in os.listdir(os.path.join(host_path, name)) if i.endswith('.xyz')]
            files_prefix = [os.path.splitext(i)[0] for i in host_files]
            for prefix in tqdm(files_prefix):
                try:
                    host_xtb, _ = xtb.run_xtb(
                        os.path.join(host_path, name, f"{prefix}_host.xyz"),
                        outpath=outdir,
                        name=name)
                    guest_xtb, _ = xtb.run_xtb(
                        os.path.join(guest_path, name, f"{prefix}_guest.xyz"),
                        outpath=outdir,
                        name=name)
                except Exception as e:
                    print(e)

    """

    def __init__(self, xtb, multiwfn):
        self.xtb = xtb
        self.multiwfn = multiwfn
        self.multiwfn_settings = multiwfn.rstrip('Multiwfn') + 'settings.ini'

    def run_xtb(self, xyzpath, outpath=None, name=None):
        """
        :param xyzpath: path of the .xyz file
        :param outpath: outdir of the calculation
        :param name: CCDC RefCode of the xyz file

        The process working directory is restored on exit, so this method is safe
        to call repeatedly (including from a multiprocessing pool).
        """
        filename = os.path.basename(xyzpath)
        # `rstrip('.xyz')` strips the character set {'.', 'x', 'y', 'z'}; use
        # os.path.splitext so names such as 'complex_foxy.xyz' stay intact.
        stem = os.path.splitext(filename)[0]
        outpath = os.path.abspath(outpath)
        target_path = os.path.join(outpath, name, stem)
        os.makedirs(target_path, exist_ok=True)
        if 'molden.input' in os.listdir(target_path):
            print(f"{filename} has been calculated!!")
            return 0, 0

        current_directory = os.getcwd()
        try:
            if not os.path.exists(os.path.join(target_path, filename)):
                shutil.copy(xyzpath, target_path)
            os.chdir(target_path)
            out, errors = run_command(f"{self.xtb} {os.path.join(target_path, filename)} --molden --esp")
        finally:
            os.chdir(current_directory)
        return out, errors

    def run_xtb_single(self, xyzpath, outpath=None, esp=False, opt=False, others=' --iterations 9999'):
        """
        :param xyzpath: path of the .xyz file
        :param outpath: outdir of the calculation
        :param esp: Whether calculate esp or not
        :param opt: Whether calculate opt or not
        :param others: Other key words for xtb. In the form of " --iterations 1000"
        """
        filename = os.path.basename(xyzpath)
        outpath = os.path.abspath(outpath)
        os.makedirs(outpath, exist_ok=True)
        if 'molden.input' in os.listdir(outpath):
            print(f"{filename} has been calculated!!")
            return 0, 0

        command_input = f"{self.xtb} {os.path.join(outpath, filename)} --molden"
        if esp:
            command_input += " --esp"
        if opt:
            command_input += " --opt"
        if others:
            command_input += others

        current_directory = os.getcwd()
        try:
            os.chdir(outpath)
            out, errors = run_command(command_input)
        finally:
            os.chdir(current_directory)
        return out, errors

    def run_molden_to_fch(self, molden_path, workdir):
        """
        Convert molden.input to molden.fch file.

        The molden file should be the default value "molden.input"

        :param molden_path: the path of molden.input
        :param workdir: the working directory of Multiwfn runs.
                        it should be the folder of molden_path.
        """
        current_directory = os.getcwd()
        workdir = os.path.abspath(workdir)
        os.makedirs(workdir, exist_ok=True)
        try:
            os.chdir(workdir)
            if 'molden.fch' in os.listdir(workdir):
                print(f"{molden_path} has been converted!!")
                return 0, 0
            molden_to_fch_txt = ['\n', '100\n', '2\n', '7\n', '\n']
            with open(os.path.join(workdir, 'molden2fch.txt'), 'w') as f:
                f.writelines(molden_to_fch_txt)
            if 'molden.input' not in os.listdir(workdir):
                shutil.copy(molden_path, workdir)
            if 'settings.ini' not in os.listdir(workdir):
                shutil.copy(self.multiwfn_settings, workdir)
            out, errors = run_command(f"{self.multiwfn} {molden_path} < molden2fch.txt |tee "
                                      f"molden2fch.log")
            for scratch in ('molden2fch.txt', 'settings.ini'):
                if os.path.exists(scratch):
                    os.remove(scratch)
        finally:
            os.chdir(current_directory)
        return out, errors

    def run_fch_to_esp(self, fchpath, workdir, grid_points_spacing=0.25, rename=False):
        """
        The .fch file name shoule be the default value "molden.fch".
        """
        current_directory = os.getcwd()
        workdir = os.path.abspath(workdir)
        os.makedirs(workdir, exist_ok=True)
        try:
            os.chdir(workdir)
            existing = 'esp.pdb' if rename else 'vtx.pdb'
            if existing in os.listdir(workdir):
                print(f"{fchpath} has been calculated to {existing}!!")
                return 0, 0
            fch_to_esp_txt = ['12\n', '3\n', f'{grid_points_spacing}\n', '0\n', '-2\n', '\n', '66\n', '\n']
            with open(os.path.join(workdir, 'fch2esp.txt'), 'w') as f:
                f.writelines(fch_to_esp_txt)
            if 'molden.fch' not in os.listdir(workdir):
                shutil.copy(fchpath, workdir)
            if 'settings.ini' not in os.listdir(workdir):
                shutil.copy(self.multiwfn_settings, workdir)
            out, errors = run_command(f"{self.multiwfn} {fchpath} < fch2esp.txt |tee fch2esp.log")
            for scratch in ('fch2esp.txt', 'settings.ini'):
                if os.path.exists(scratch):
                    os.remove(scratch)
            if rename and os.path.exists('vtx.pdb'):
                shutil.move('vtx.pdb', 'esp.pdb')
        finally:
            os.chdir(current_directory)
        return out, errors

    def run_fch_to_ed(self, fchpath, workdir, isovalue=0.001, rename=False):
        """
        The .fch file name shoule be the default value "molden.fch".
        """
        current_directory = os.getcwd()
        workdir = os.path.abspath(workdir)
        os.makedirs(workdir, exist_ok=True)
        try:
            os.chdir(workdir)
            existing = 'ed.pdb' if rename else 'vtx.pdb'
            if existing in os.listdir(workdir):
                print(f"{fchpath} has been calculated to {existing}!!")
                return 0, 0
            fch_to_ed_txt = ['12\n', '1\n', '1\n', f'{isovalue}\n', '6\n', '66\n', '\n']
            with open(os.path.join(workdir, 'fch2ed.txt'), 'w') as f:
                f.writelines(fch_to_ed_txt)
            if 'molden.fch' not in os.listdir(workdir):
                shutil.copy(fchpath, workdir)
            if 'settings.ini' not in os.listdir(workdir):
                shutil.copy(self.multiwfn_settings, workdir)
            out, errors = run_command(f"{self.multiwfn} {fchpath} < fch2ed.txt |tee fch2ed.log")
            for scratch in ('fch2ed.txt', 'settings.ini'):
                if os.path.exists(scratch):
                    os.remove(scratch)
            if rename and os.path.exists('vtx.pdb'):
                shutil.move('vtx.pdb', 'ed.pdb')
        finally:
            os.chdir(current_directory)
        return out, errors

    def run_fch_to_pdb(self, fch_path, workdir):
        """
        Convert a .fch file into a .pdb file with Multiwfn.
        """
        current_directory = os.getcwd()
        workdir = os.path.abspath(workdir)
        os.makedirs(workdir, exist_ok=True)
        try:
            os.chdir(workdir)

            fch_to_pdb_txt = ['100\n', '2\n', '1\n', '\n']
            with open(os.path.join(workdir, 'fch2pdb.txt'), 'w') as f:
                f.writelines(fch_to_pdb_txt)
            if 'molden.fch' not in os.listdir(workdir):
                shutil.copy(fch_path, workdir)
            if 'settings.ini' not in os.listdir(workdir):
                shutil.copy(self.multiwfn_settings, workdir)
            out, errors = run_command(f"{self.multiwfn} {fch_path} < fch2pdb.txt |tee "
                                      f"fch2pdb.log")
            for scratch in ('fch2pdb.txt', 'settings.ini'):
                if os.path.exists(scratch):
                    os.remove(scratch)
        finally:
            os.chdir(current_directory)
        return out, errors

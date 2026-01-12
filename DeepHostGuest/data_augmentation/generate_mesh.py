from DeepHostGuest.data_augmentation.run_multisim import run_command
import shutil
import os


class ESP:
    """
    Generate molden.input and convert it into mesh file.

    Example: (Can be seen in /examples/2.train_deepdock.preprocessing)
    -------------------------------------------------
    1.preprocessing.generate_structural_data. Execute xtb to generate molden.input
        from DeepDockHostGuest.1.preprocessing.preprocessing.generate_mesh import *
        from tqdm import tqdm

        xtb = ESP('/path/to/xtb', '/path/to/Multiwfn)
        host_path = '/path/to/Host'
        guest_path = '/path/to/Guest'
        names = os.listdir(host_path)
        outdir = '/path/to/xtb_output'

        for name in tqdm(names):
            print(f"======Processing {name}======")
            host_files = [i for i in os.listdir(os.path.join(host_path, name)) if i.endswith('.xyz')]
            files_prefix = [i.rstrip('_host.xyz') for i in host_files]
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
        """
        filename = xyzpath.split('/')[-1]
        target_path = os.path.join(outpath, name, filename.rstrip('.xyz'))
        os.makedirs(target_path, exist_ok=True)
        if 'molden.input' in os.listdir(target_path):
            print(f"{filename} has been calculated!!")
            return 0, 0
        else:
            shutil.copy(xyzpath, target_path)
            os.chdir(target_path)
            out, errors = run_command(f"{self.xtb} {os.path.join(target_path, filename)} --molden --esp")
            return out, errors

    def run_xtb_single(self, xyzpath, outpath=None, esp=False, opt=False, others=' --iterations 9999'):
        """
        :param xyzpath: path of the .xyz file
        :param outpath: outdir of the calculation
        :param esp: Whether calculate esp or not
        :param opt: Whether calculate opt or not
        :param others: Other key words for xtb. In the form of " --iterations 1000"
        """
        filename = xyzpath.split('/')[-1]
        os.makedirs(outpath, exist_ok=True)
        if 'molden.input' in os.listdir(outpath):
            print(f"{filename} has been calculated!!")
            return 0, 0
        else:
            command_input = f"{self.xtb} {os.path.join(outpath, filename)} --molden"
            os.chdir(outpath)
            if esp:
                command_input += " --esp"
            if opt:
                command_input += " --opt"
            if others:
                command_input += others
            out, errors = run_command(command_input)
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
        if not os.path.exists(workdir):
            os.makedirs(workdir)
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
        os.remove('molden2fch.txt')
        os.remove('settings.ini')
        os.chdir(current_directory)
        return out, errors

    def run_fch_to_esp(self, fchpath, workdir, grid_points_spacing=0.25, rename=False):
        """
        The .fch file name shoule be the default value "molden.fch".
        """
        current_directory = os.getcwd()
        os.chdir(workdir)
        if not os.path.exists(workdir):
            os.makedirs(workdir)
        if not rename and 'vtx.pdb' in os.listdir(workdir) or rename and 'esp.pdb' in os.listdir(workdir):
            print(f"{fchpath} has been calculated to vtx.pdb!!")
            return 0, 0
        fch_to_esp_txt = ['12\n', '3\n', f'{grid_points_spacing}\n', '0\n', '-2\n', '\n', '66\n', '\n']
        with open(os.path.join(workdir, 'fch2esp.txt'), 'w') as f:
            f.writelines(fch_to_esp_txt)
        if 'molden.fch' not in os.listdir(workdir):
            shutil.copy(fchpath, workdir)
        os.chdir(workdir)
        if 'settings.ini' not in os.listdir(workdir):
            shutil.copy(self.multiwfn_settings, workdir)
        out, errors = run_command(f"{self.multiwfn} {fchpath} < fch2esp.txt |tee fch2esp.log")
        os.remove('fch2esp.txt')
        os.remove('settings.ini')
        if rename:
            shutil.move('vtx.pdb', 'esp.pdb')
        os.chdir(current_directory)
        return out, errors

    def run_fch_to_ed(self, fchpath, workdir, isovalue=0.001, rename=False):
        """
        The .fch file name shoule be the default value "molden.fch".
        """
        current_directory = os.getcwd()
        os.chdir(workdir)
        if not os.path.exists(workdir):
            os.makedirs(workdir)
        if not rename and 'vtx.pdb' in os.listdir(workdir) or rename and 'ed.pdb' in os.listdir(workdir):
            print(f"{fchpath} has been calculated to vtx.pdb!!")
            return 0, 0
        fch_to_ed_txt = ['12\n', '1\n', '1\n', f'{isovalue}\n', '6\n', '66\n', '\n']
        with open(os.path.join(workdir, 'fch2ed.txt'), 'w') as f:
            f.writelines(fch_to_ed_txt)
        if 'molden.fch' not in os.listdir(workdir):
            shutil.copy(fchpath, workdir)
        os.chdir(workdir)
        if 'settings.ini' not in os.listdir(workdir):
            shutil.copy(self.multiwfn_settings, workdir)
        out, errors = run_command(f"{self.multiwfn} {fchpath} < fch2ed.txt |tee fch2ed.log")
        os.remove('fch2ed.txt')
        os.remove('settings.ini')
        if rename:
            shutil.move('vtx.pdb', 'ed.pdb')
        os.chdir(current_directory)
        return out, errors

    def run_fch_to_pdb(self, fch_path, workdir):
        """

        """
        current_directory = os.getcwd()
        if not os.path.exists(workdir):
            os.makedirs(workdir)
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
        os.remove('fch2pdb.txt')
        os.remove('settings.ini')
        os.chdir(current_directory)
        return out, errors

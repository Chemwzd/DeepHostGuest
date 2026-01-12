from DeepHostGuest.data_augmentation.run_multisim import run_command
import shutil
import os


class Multiwfn:
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

    def __init__(self, multiwfn):
        self.multiwfn = multiwfn
        self.multiwfn_settings = multiwfn.rstrip('Multiwfn') + 'settings.ini'

    def convert_to_fch(self, input_file, workdir, formchk_path=None):
        """
        Convert molden.input to molden.fch file.

        The molden file should be the default value "molden.input"

        :param input_file: the path of input file
        :param workdir: the working directory of Multiwfn runs.
                        it should be the folder of molden_path.
        """
        current_directory = os.getcwd()
        if not os.path.exists(workdir):
            os.makedirs(workdir)
        os.chdir(workdir)
        if formchk_path:
            out, errors = run_command(f"{formchk_path} {input_file}")
        else:
            convert_to_fch_txt = ['\n', '100\n', '2\n', '7\n', '\n']
            with open(os.path.join(workdir, 'convert2fch.txt'), 'w') as f:
                f.writelines(convert_to_fch_txt)
            if 'settings.ini' not in os.listdir(workdir):
                shutil.copy(self.multiwfn_settings, workdir)
            out, errors = run_command(f"{self.multiwfn} {input_file} < convert2fch.txt |tee "
                                      f"convert2fch.log")
            os.remove('convert2fch.txt')
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

    @staticmethod
    def get_surface_info(log_file, decimals=4):
        """
        Read surface analysis results from the log file after converting the ESP file with Multiwfn.

            volume (Angstrom^3): Van der Waals volume of the molecule.
            min_esp_value (kcal/mol): Minimum value of ESP.
            max_esp_value (kcal/mol): Maximum value of ESP.
            overall_sa (Angstrom^2): Overall surface area of the molecule.
            positive_sa (Angstrom^2): Surface area of regions with positive ESP values.
            negative_sa (Angstrom^2): Surface area of regions with positive ESP values.
            overall_average (kcal/mol)：Average ESP value of the overall region.
            positive_average (kcal/mol)：Average ESP value of the positive region.
            negative_average (kcal/mol)：Average ESP value of the negative region.
            overall_variance ((kcal/mol)^2): Variance of ESP in the overall region.
            positive_variance ((kcal/mol)^2): Variance of ESP in the positive region.
            negative_variance ((kcal/mol)^2): Variance of ESP in the negative region.
            molecular_polarity_index (kcal/mol): Molecular polarity index (MPI).
            nonpolar_sa (Angstrom^2): Polar surface area (|ESP| > 10 kcal/mol).
            polar_sa (Angstrom^2): Nonpolar surface area (|ESP| <= 10 kcal/mol).
        """
        with open(log_file, 'r') as f:
            log_contents = f.readlines()

        key_matching_dict = {
            ' Volume:': None,
            ' Minimal value:': [None, None],
            ' Overall surface area:': None,
            ' Positive surface area:': None,
            ' Negative surface area:': None,
            ' Overall average value:': None,
            ' Positive average value:': None,
            ' Negative average value:': None,
            ' Overall variance': None,
            ' Positive variance:': None,
            ' Negative variance:': None,
            ' Molecular polarity index (MPI):': None,
            ' Nonpolar surface area (|ESP| <= 10 kcal/mol):': None,
            ' Polar surface area (|ESP| > 10 kcal/mol):': None,
        }

        for log_line in log_contents:
            for key, value in key_matching_dict.items():
                if log_line.startswith(key):
                    if key in [' Nonpolar surface area (|ESP| <= 10 kcal/mol):',
                               ' Polar surface area (|ESP| > 10 kcal/mol):']:
                        splited_line = log_line.split('(')[-2].split()
                        key_matching_dict[key] = round(float(splited_line[-2]), decimals)
                    elif key == ' Minimal value:':
                        key_matching_dict[key] = [round(float(log_line.split()[2]), decimals),
                                                  round(float(log_line.split()[6]), decimals)]
                    else:
                        key_matching_dict[key] = round(float(log_line.split()[-2]), decimals)

        return {'volume': key_matching_dict[' Volume:'],
                'min_esp_value': key_matching_dict[' Minimal value:'][0],
                'max_esp_value': key_matching_dict[' Minimal value:'][1],
                'overall_sa': key_matching_dict[' Overall surface area:'],
                'positive_sa': key_matching_dict[' Positive surface area:'],
                'negative_sa': key_matching_dict[' Negative surface area:'],
                'overall_average': key_matching_dict[' Overall average value:'],
                'positive_average': key_matching_dict[' Positive average value:'],
                'negative_average': key_matching_dict[' Negative average value:'],
                'overall_variance': key_matching_dict[' Overall variance'],
                'positive_variance': key_matching_dict[' Positive variance:'],
                'negative_variance': key_matching_dict[' Negative variance:'],
                'molecular_polarity_index': key_matching_dict[' Molecular polarity index (MPI):'],
                'nonpolar_sa': key_matching_dict[' Nonpolar surface area (|ESP| <= 10 kcal/mol):'],
                'polar_sa': key_matching_dict[' Polar surface area (|ESP| > 10 kcal/mol):']}

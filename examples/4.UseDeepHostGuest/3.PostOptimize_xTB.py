"""
Post-optimize a predicted host-guest complex with GFN2-xTB.

This performs the local geometry refinement that follows DeepHostGuest pose
prediction, matching the protocol described in the manuscript:

    "All DeepHostGuest-predicted host-guest complexes were optimized with
     GFN2-xTB using default settings. For charged host-guest complexes ...
     the analytical linearized Poisson-Boltzmann (ALPB) implicit water
     solvent model was incorporated."

Geometry optimization REQUIRES the ``--opt`` flag. Without it xTB performs a
single-point calculation and ``xtbopt.mol`` is never written.

Usage
-----
    python 3.PostOptimize_xTB.py --host vuqzal_1.mol --guest vuqzal_2_pre.mol

    # charged complex (total charge = host + guest); ALPB water screening
    python 3.PostOptimize_xTB.py --host host.mol --guest guest_pre.mol \
        --charge -4 --solvent water --outdir ./postopt

Outputs (in --outdir, default: ./postopt)
-----------------------------------------
    <host>_<guest>_complex.mol   combined input complex
    xtbopt.mol                   raw xTB output geometry
    <host>_<guest>_opt.mol       final optimized complex

Requirements
------------
    xTB (tested with 6.4.1) and Open Babel in PATH, or pass --xtb/--obabel.
"""
import argparse
import os
import shutil
import subprocess
import sys

from rdkit import Chem
from rdkit.Chem import rdmolops


def run_command(cmd, cwd=None):
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    ap = argparse.ArgumentParser(description="GFN2-xTB post-optimization of a predicted host-guest complex.")
    ap.add_argument("--host", required=True, help="host molecule (.mol) used in the prediction")
    ap.add_argument("--guest", required=True, help="predicted guest pose (.mol), e.g. <name>_2_pre.mol")
    ap.add_argument("--charge", type=int, default=0, help="TOTAL charge of the complex (host + guest), default 0")
    ap.add_argument("--solvent", default=None, help="ALPB implicit solvent, e.g. water (recommended when --charge != 0)")
    ap.add_argument("--outdir", default="./postopt", help="output directory (default: ./postopt)")
    ap.add_argument("--xtb", default=None, help="path to the xTB executable (default: 'xtb' from PATH)")
    ap.add_argument("--obabel", default=None, help="path to the Open Babel executable (default: 'obabel' from PATH)")
    ap.add_argument("--level", default="normal", help="xTB optimization level (default: normal)")
    ap.add_argument("--gfnff-fallback", action="store_true",
                    help="fall back to GFN-FF optimization if GFN2-xTB fails")
    args = ap.parse_args()

    xtb = args.xtb or shutil.which("xtb")
    if not xtb:
        sys.exit("xTB executable not found. Install xTB (https://github.com/grimme-lab/xtb) or pass --xtb /path/to/xtb.")
    obabel = args.obabel or shutil.which("obabel")

    host_mol = Chem.MolFromMolFile(args.host, removeHs=False, sanitize=True)
    guest_mol = Chem.MolFromMolFile(args.guest, removeHs=False, sanitize=True)
    if host_mol is None or guest_mol is None:
        sys.exit("Could not read the host or guest .mol file with RDKit. Check the file format.")

    os.makedirs(args.outdir, exist_ok=True)
    args.outdir = os.path.abspath(args.outdir)
    stem = f"{os.path.splitext(os.path.basename(args.host))[0]}_{os.path.splitext(os.path.basename(args.guest))[0]}"
    complex_path = os.path.join(args.outdir, f"{stem}_complex.mol")
    opt_path = os.path.join(args.outdir, f"{stem}_opt.mol")

    complex_mol = rdmolops.CombineMols(host_mol, guest_mol)
    Chem.MolToMolFile(complex_mol, complex_path)
    print(f"[1/3] combined complex written: {complex_path}")

    if obabel:
        run_command([obabel, complex_path, "-O", complex_path])
    else:
        print("[1/3] Open Babel not found - skipping format normalisation (xTB will read the RDKit .mol directly)")

    xtbopt = os.path.join(args.outdir, "xtbopt.mol")

    def xtb_command(gfnff=False):
        cmd = [xtb, complex_path, "-c", str(args.charge), "-P", "32"]
        if gfnff:
            cmd.append("--gfnff")
        cmd += ["--opt", args.level]
        if args.solvent:
            cmd += ["--alpb", args.solvent]
        cmd.append("-v")
        return cmd

    if os.path.exists(xtbopt):
        os.remove(xtbopt)

    print(f"[2/3] GFN2-xTB geometry optimization (charge={args.charge}, solvent={args.solvent or 'none'})")
    try:
        run_command(xtb_command(gfnff=False), cwd=args.outdir)
    except subprocess.CalledProcessError:
        print("GFN2-xTB optimization failed.")
        if args.gfnff_fallback:
            print("Falling back to GFN-FF optimization ...")
            run_command(xtb_command(gfnff=True), cwd=args.outdir)

    if not os.path.exists(xtbopt):
        sys.exit("xTB did not produce xtbopt.mol - optimization did not complete. "
                 "Check the xTB output above (try --gfnff-fallback for difficult systems).")

    shutil.copy(xtbopt, opt_path)
    print(f"[3/3] optimized complex written: {opt_path}")
    print("Next step: evaluate the guest RMSD against the experimental reference "
          "(host-atom alignment followed by symmetry-corrected guest RMSD, e.g. pydockrmsd/dockrmsd).")


if __name__ == "__main__":
    main()

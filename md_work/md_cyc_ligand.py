################################################################################################
# This code is essentially adapted (taken) from the blog of Corin Wagen
# I have made some modifications to it, but you should cite the following blog post
# if you use it. Thanks!
# https://corinwagen.github.io/public/blog/20240613_simple_md.html
################################################################################################

import os
from openff.toolkit import Molecule, Topology

from openmm import *
from openmm.app import *

import mdtraj
import matplotlib.pyplot as plt
import numpy as np
import openmoltools
import tempfile
import cctk
import openmmtools
import math
import tqdm
from random import random, randint

from sys import stdout
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

from openmmforcefields.generators import SMIRNOFFTemplateGenerator

#########################################################
# User inputs below
# vvvv
#########################################################

smiles = "CNCc1c(O)ccc(c1)C" # !!! CHANGE !!!
results_dir = '/home/bfd21/rds/hpc-work/tbg/md_work/test_results' # !!! CHANGE !!!
seeds = range(10) # !!! CHANGE !!!

#########################################################
# ^^^^
# User inputs above
#########################################################

os.makedirs(results_dir, exist_ok=True)

def generate_forcefield(smiles: str) -> ForceField:
    """ Creates an OpenMM ForceField object that knows how to handle a given SMILES string """
    molecule = Molecule.from_smiles(smiles)
    smirnoff = SMIRNOFFTemplateGenerator(molecules=molecule)
    forcefield = ForceField(
      'amber/protein.ff14SB.xml',
      'amber/tip3p_standard.xml',
      'amber/tip3p_HFE_multivalent.xml'
     )
    forcefield.registerTemplateGenerator(smirnoff.generator)
    return forcefield

def generate_initial_pdb(
    smiles: str,
    min_side_length: int = 25, # Å
    solvent_smiles = "O",
) -> PDBFile:
    """ Creates a PDB file for a solvated molecule, starting from two SMILES strings. """

    # do some math to figure how big the box needs to be
    solute = cctk.Molecule.new_from_smiles(smiles)
    solute_volume = solute.volume(qhull=True)
    solvent = cctk.Molecule.new_from_smiles(solvent_smiles)
    solvent_volume = solvent.volume(qhull=False)

    total_volume = 50 * solute_volume # seems safe?
    min_allowed_volume = min_side_length ** 3
    total_volume = max(min_allowed_volume, total_volume)

    total_solvent_volume = total_volume - solute_volume
    n_solvent = int(total_solvent_volume // solvent_volume)
    box_size = total_volume ** (1/3)

    # build pdb
    with tempfile.TemporaryDirectory() as tempdir:
        solute_fname = f"{tempdir}/solute.pdb"
        solvent_fname = f"{tempdir}/solvent.pdb"
        system_fname = f"system.pdb"

        smiles_to_pdb(smiles, solute_fname)
        smiles_to_pdb(solvent_smiles, solvent_fname)
        traj_packmol = openmoltools.packmol.pack_box(
          [solute_fname, solvent_fname],
          [1, n_solvent],
          box_size=box_size
         )
        traj_packmol.save_pdb(system_fname)

        return PDBFile(system_fname)

def smiles_to_pdb(smiles: str, filename: str) -> None:
    """ Turns a SMILES string into a PDB file (written to current working directory). """
    m = Chem.MolFromSmiles(smiles)
    mh = Chem.AddHs(m)
    AllChem.EmbedMolecule(mh)
    Chem.MolToPDBFile(mh, filename)

forcefield = generate_forcefield(smiles)
pdb = generate_initial_pdb(smiles, solvent_smiles="O")

system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=PME,
    nonbondedCutoff=1*unit.nanometer,
)

for seed in tqdm.tqdm(seeds):
    traj_file = os.path.join(results_dir, f"traj_seed_{seed}.dcd")
    csv_file = os.path.join(results_dir, f"scalars_seed_{seed}.csv")

    # initialize Langevin integrator and minimize
    integrator = LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 1 * unit.femtoseconds)
    integrator.setRandomNumberSeed(seed)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()

    # we'll make this an NPT simulation now
    system.addForce(MonteCarloBarostat(1*unit.bar, 300*unit.kelvin))
    simulation.context.reinitialize(preserveState=True)

    checkpoint_interval = 100
    printout_interval = 10000

    # set the reporters collecting the MD output.
    simulation.reporters = []
    simulation.reporters.append(DCDReporter(traj_file, checkpoint_interval))
    simulation.reporters.append(
        StateDataReporter(
            stdout,
            printout_interval,
            step=True,
            temperature=True,
            elapsedTime=True,
            volume=True,
            density=True
        )
    )

    simulation.reporters.append(
        StateDataReporter(
            csv_file,
            checkpoint_interval,
            time=True,
            potentialEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
        )
    )

    # actually run the MD
    simulation.step(500000) # this is the number of steps, you may want fewer to test quickly
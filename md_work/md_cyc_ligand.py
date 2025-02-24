################################################################################################
# This code is essentially adapted (taken) from the blog of Corin Wagen
# I have made some modifications to it, but you should cite the following blog post
# if you use it. Thanks!
# https://corinwagen.github.io/public/blog/20240613_simple_md.html
################################################################################################

from openff.toolkit import Molecule
from openmm import *
from openmm.app import *
import matplotlib.pyplot as plt
import numpy as np
import mdtraj
import openmoltools
import openmmtools
import math
import os
from random import randint
from sys import stdout
from rdkit import Chem
from rdkit.Chem import AllChem
from openmmforcefields.generators import SMIRNOFFTemplateGenerator

#######################################
# User-defined settings
#######################################
output_dir = "my_simulation_results"
seeds = range(10)  # List of seeds for independent runs
pdb_filename = "input.pdb"  # Input PDB file
simulation_steps = 500000  # Adjust for testing
checkpoint_interval = 100  # Save trajectory every N steps
printout_interval = 10000  # Print simulation state every N steps

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

##################################################
# Generate force field once (avoids recomputation)
##################################################

def generate_forcefield(pdb_file: str) -> tuple[ForceField, PDBFile, str]:
    """Creates an OpenMM ForceField object for a given PDB file and extracts its SMILES string."""
    
    # Load pdb file
    pdb = PDBFile(pdb_file)
    rdkit_mol = Chem.MolFromPDBFile(pdb_file)
    
    # Create OpenFF molecule
    molecule = Molecule.from_rdkit(rdkit_mol)

    # Generate SMIRNOFF force field
    smirnoff = SMIRNOFFTemplateGenerator(molecules=molecule)
    forcefield = ForceField(
        'amber/protein.ff14SB.xml',
        'amber/tip3p_standard.xml',
        'amber/tip3p_HFE_multivalent.xml'
    )
    forcefield.registerTemplateGenerator(smirnoff.generator)

    return forcefield, pdb

forcefield, pdb = generate_forcefield(pdb_filename)

# Create the system once
system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=PME,
    nonbondedCutoff=1 * unit.nanometer,
)

# Add barostat once before entering the loop
system.addForce(MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin))

#######################################
# Run simulations for different seeds
#######################################
for seed in seeds:
    # Initialize integrator with unique seed
    integrator = LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 1 * unit.femtoseconds)
    integrator.setRandomNumberSeed(seed)

    # Create a new simulation object (resets context)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()
    
    simulation.context.reinitialize(preserveState=True)

    # Define unique filenames per seed
    traj_file = os.path.join(output_dir, f"traj_seed_{seed}.dcd")
    csv_file = os.path.join(output_dir, f"scalars_seed_{seed}.csv")

    # Set up reporters
    simulation.reporters = [
        DCDReporter(traj_file, checkpoint_interval),
        StateDataReporter(
            stdout,
            printout_interval,
            step=True,
            temperature=True,
            elapsedTime=True,
            volume=True,
            density=True
        ),
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
    ]

    # Run MD simulation
    simulation.step(simulation_steps)  # This dominates computational time

import argparse
import os
import logging
import sys

import time
from datetime import datetime

from openfermion import MolecularData
from openfermion import geometry_from_pubchem, count_qubits
from openfermionpsi4 import run_psi4
from openfermion.transforms import get_fermion_operator, jordan_wigner

from .molecule_repository import molecule_to_multiplicity, molecule_to_charge
from .molecule_repository import linear_molecules, linear_stubs, carleo_geometries

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
STR_TO_LOGGING_LEVEL = {
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
}
DEFAULT_MOLECULE_ROOT = './molecules'
DEFAULT_MOLECULE_FILENAME = 'molecular_data.hdf5'


# Creates basic parser
def create_parser():
    parser = argparse.ArgumentParser(description='Obtain required directories')
    parser.add_argument('--molecule_root',
                        default=DEFAULT_MOLECULE_ROOT,
                        help='Path to the molecules .hdf5 collection')
    parser.add_argument('--molecule_name',
                        required=False,
                        default='LiH',
                        help='Name of the molecule (either mundane or expressed with the formula')
    parser.add_argument('--molecule_type',
                        required=False,
                        choices=['carleo', 'pubchem', 'barrett', 'linear'],
                        default='carleo',
                        help='Type of molecule geometry')
    parser.add_argument('--internuclear_distance',
                        required=False,
                        type=float,
                        default=None,
                        help='Internuclear distance (only together with molecule_type = linear)')
    parser.add_argument('--basis',
                        choices=['sto-3g'],
                        default='sto-3g',
                        help='Basis for quantum chemistry calculations')
    parser.add_argument('--log_filename',
                        required=False,
                        default=None,
                        help='Name of log file')
    parser.add_argument('--logging_level',
                        required=False,
                        default='INFO',
                        help='Level of logging')
    parser.add_argument('--run_mp2',
                        type=bool,
                        default=1,
                        help='Perform MP2 calculations')
    parser.add_argument('--run_cisd',
                        type=bool,
                        default=1,
                        help='Perform CISD calculations')
    parser.add_argument('--run_ccsd',
                        type=bool,
                        default=1,
                        help='Perform CCSD calculations')
    parser.add_argument('--run_fci',
                        type=bool,
                        default=1,
                        help='Perform FCI calculations')

    return parser


def calc_molecule_dir(args):
    molecule_dir = os.path.join(args.molecule_root,
                                args.molecule_name,
                                args.basis,
                                args.molecule_type,
                                f'{args.internuclear_distance}')

    return molecule_dir


def create_logger(args):
    molecule_dir = calc_molecule_dir(args)
    log_dir = os.path.join(molecule_dir, f'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = args.log_filename if args.log_filename is not None else f"{datetime.now().strftime(DATE_FORMAT)}.txt"
    log_filename = os.path.join(log_dir, log_filename)

    logger = logging.getLogger('nnqs')
    logger.setLevel(STR_TO_LOGGING_LEVEL[args.logging_level])

    # Remove all previously created handlers (if any)
    logger.handlers = []

    fh = logging.FileHandler(log_filename)
    fh.setLevel(STR_TO_LOGGING_LEVEL[args.logging_level])
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                       datefmt=DATE_FORMAT)
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(STR_TO_LOGGING_LEVEL[args.logging_level])
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def create_molecule(args, logger):
    molecule_dir = calc_molecule_dir(args)
    molecule_filename = os.path.join(molecule_dir, DEFAULT_MOLECULE_FILENAME)

    molecule = None
    if os.path.exists(molecule_filename):
        logger.info(f'For {args.molecule_name} molecule .hdf5 was found, loading the molecule...')
        molecule = MolecularData(filename=molecule_filename)
        logger.info(f'The molecule was successfully loaded!')
    else:
        molecule_geometry = None
        if args.molecule_type == 'pubchem':
            molecule_geometry = geometry_from_pubchem(args.molecule_name)
            if molecule_geometry is not None:
                logger.info(f'Geometry of {args.molecule_name} molecule was successfully '
                            f'downloaded from Pubchem')
            else:
                msg = f'Geometry of {args.molecule_name} molecule was not successfully loaded'
                logger.error(msg)
                raise RuntimeError(msg)
        elif args.molecule_type == 'carleo':
            if args.molecule_name in carleo_geometries:
                molecule_geometry = carleo_geometries[args.molecule_name]
                logger.info(f'Geometry of {args.molecule_name} molecule was set according '
                            f'to the Carleo paper')
            else:
                msg = f'Failed attempt to load Carleo\'s geometry for '\
                      f'{args.molecule_name} molecule. '\
                      f'Available molecules are {carleo_geometries.keys()}.'
                logger.error(msg)
                raise ValueError(msg)
        elif args.molecule_type == 'linear':
            assert args.molecule_name in linear_molecules
            assert args.internuclear_distance is not None
            molecule_geometry = linear_stubs[args.molecule_name]
            molecule_geometry[1][1][2] = args.internuclear_distance
            logger.info(f'Geometry of {args.molecule_name} molecule was set to {molecule_geometry} '
                        f'as the former was asked to be linear with internuclear distance '
                        f'{args.internuclear_distance}')
        molecule = MolecularData(molecule_geometry,
                                 args.basis,
                                 molecule_to_multiplicity[args.molecule_name],
                                 molecule_to_charge[args.molecule_name],
                                 filename=molecule_filename)
        logger.info(f'Running psi4 calculations...')
        start = time.time()
        molecule = run_psi4(molecule,
                            run_mp2=args.run_mp2,
                            run_cisd=args.run_cisd,
                            run_ccsd=args.run_ccsd,
                            run_fci=args.run_fci)
        logger.info(f'The molecule was successfully calculated in {time.time() - start} seconds!')

    return molecule


def create_of_qubit_operator(molecule):
    of_fermion_operator = get_fermion_operator(molecule.get_molecular_hamiltonian())
    of_qubit_operator = jordan_wigner(of_fermion_operator)
    qubit_num = count_qubits(of_qubit_operator)

    return of_qubit_operator, qubit_num


def create_of_hamiltonian(molecule):
    of_fermion_operator = get_fermion_operator(molecule.get_molecular_hamiltonian())
    of_hamiltonian = jordan_wigner(of_fermion_operator)

    return of_hamiltonian


def load_molecule(molecule_root: str = '../molecules',
                  molecule_name: str = None,
                  molecule_type: str = 'carleo',
                  internuclear_distance: float = 1.26,
                  create_hamiltonian: bool = False):
    parser = create_parser()
    args = parser.parse_args([])

    args.molecule_root = molecule_root
    args.molecule_name = molecule_name
    args.molecule_type = molecule_type
    args.internuclear_distance = internuclear_distance

    logger = create_logger(args)
    molecule_dir = calc_molecule_dir(args)
    molecule = create_molecule(args, logger)

    of_hamiltonian = None
    if create_hamiltonian:
        of_hamiltonian = create_of_hamiltonian(molecule)

    return molecule, molecule_dir, of_hamiltonian, logger

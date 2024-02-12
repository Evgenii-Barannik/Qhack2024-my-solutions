import json
import pennylane as qml
import pennylane.numpy as np

def potential_energy_surface(symbols, bond_lengths):
    """Calculates the molecular energy over various bond lengths (AKA the 
    potential energy surface) using the Hartree Fock method.
    
    Args:
        symbols (list(string)): 
            A list of atomic symbols that comprise the diatomic molecule of interest.
        bond_lengths (numpy.tensor): Bond lengths to calculate the energy over.

        
    Returns:
        hf_energies (numpy.tensor): 
            The Hartree Fock energies at every bond length value.
    """


    hf_energies = []

    # Put your code here #

    for l in bond_lengths:
        geometry = np.array([[l/2, 0.0, 0.0], [-l/2, 0.0, 0.0]])
        mol = qml.qchem.Molecule(symbols, geometry)
        hf_energy = qml.qchem.hf_energy(mol)(geometry) 
        # print("Geometry: {}, Energy: {}".format (geometry, hf_energy))
        hf_energies.append(hf_energy)

    return np.array(hf_energies)


def ground_energy(hf_energies):
    """Finds the minimum energy of a molecule given its potential energy surface.
    
    Args: 
        hf_energies (numpy.tensor): 

    Returns:
        (float): The minumum energy in units of hartrees.
    """

    ind = np.argmin(hf_energies)
    return hf_energies[ind]

def reaction():
    """Calculates the energy of the reactants, the activation energy, and the energy of 
    the products in that order.

    Returns:
        (numpy.tensor): [E_reactants, E_activation, E_products]
    """
    molecules = {
        "H2": 
            {"symbols": ["H", "H"], "E0": 0, "E_dissociation": 0, "bond lengths": np.arange(0.5, 9.3, 0.3)}, 
        "Li2": 
            {"symbols": ["Li", "Li"], "E0": 0, "E_dissociation": 0, "bond lengths": np.arange(3.5, 8.3, 0.3)}, 
        "LiH": 
            {"symbols": ["Li", "H"], "E0": 0, "E_dissociation": 0, "bond lengths": np.arange(2.0, 6.6, 0.3)}
    }


    for molecule in molecules.keys():
        symbols = molecules[molecule]["symbols"]
        bond_lengths = molecules[molecule]["bond lengths"]
        print("Molecule: {}\nBond lengths: {}\n".format(symbols, bond_lengths))
        
        hf_energies = potential_energy_surface(symbols, bond_lengths)

        molecules[molecule]["E0"] = ground_energy(hf_energies)
        print("Ground Energy: ", molecules[molecule]["E0"])

        molecules[molecule]["E_dissociation"] = np.abs(ground_energy(hf_energies) - hf_energies[-1])
        print("Dissociation Energy: ", molecules[molecule]["E_dissociation"])
        # Put your code here #
        # populate each molecule's E0 and E_dissociation values

    # Calculate the following and don't forget to balance the chemical reaction!
    E_reactants = molecules["H2"]["E0"] + molecules["Li2"]["E0"]
    E_dissociation = molecules["H2"]["E_dissociation"] + molecules["Li2"]["E_dissociation"]
    E_activation = E_reactants + E_dissociation
    E_products = 2*molecules["LiH"]["E0"] 
    print(np.array([E_reactants, E_activation, E_products]))
    return np.array([E_reactants, E_activation, E_products])


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    output = reaction().tolist()
    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)

    assert np.allclose(solution_output, expected_output, rtol=1e-3)

reaction()
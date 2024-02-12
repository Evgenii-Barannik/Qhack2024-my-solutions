import json
import pennylane as qml
import pennylane.numpy as np

# Write any helper functions you need here


dev = qml.device('default.qubit', wires=[0,1,2])

@qml.qnode(dev)
def cloning_machine(coefficients, wire):
    
    """
    Returns the reduced density matrix on a wire for the cloning machine circuit.
    
    Args:
        - coefficients (np.array(float)): an array [c0,c1] containing the coefficients parametrizing
        the input state fed into the middle and bottom wires of the cloning machine.
        wire (int): The wire on which we calculate the reduced density matrix.

    Returns:
        - np.tensor(complex): The reduced density matrix on wire = wire, as returned by qml.density_matrix.
    
    """
    


    # Put your code here
    c0, c1 = coefficients
    alpha = (c0 + c1)/np.sqrt(2)
    beta = c1/np.sqrt(2)
    gamma = c0/np.sqrt(2)

    phi1 = 2*np.arccos(np.sqrt(alpha**2 + beta**2))
    phi2 = 2*np.arccos(alpha/np.sqrt(alpha**2 + beta**2))

    qml.RY(phi1, wires=1)
    qml.RY(phi2, wires=2)
    qml.CNOT([1,2])
    qml.RY(-(np.pi-phi2)/2, wires=2)
    qml.CNOT([1,2])
    qml.RY((np.pi-phi2)/2, wires=2)

    qml.CNOT(wires=[0,1])
    qml.CNOT(wires=[0,2])
    qml.CNOT(wires=[1,0])
    qml.CNOT(wires=[2,0])

    # Return the reduced density matrix
    return qml.density_matrix(wires=wire)


def fidelity(coefficients):
    
    """
    Calculates the fidelities between the reduced density matrices in wires 0 and 1 and the input state |0>.
    
    Args:
        - coefficients (np.array(float)): an array [c0,c1] containing the coefficients parametrizing
        the input state fed into the middle and bottom wires of the cloning machine.
    Returns:
        - (np.array(float)): An array whose elements are:
            - 0th element:  The fidelity between the output reduced state on wire 0 and the state |0>.
            - 1st element:  The fidelity between the output reduced state on wire 1 and the state |0>.    
    """
    


    # Put your code here
    dm0 = cloning_machine(coefficients, 0)
    dm1 = cloning_machine(coefficients, 1)

    f0 = qml.math.fidelity(dm0, np.outer([1, 0], [1, 0]))
    f1 = qml.math.fidelity(dm1, np.outer([1, 0], [1, 0]))
    return np.array([f0, f1])

# These functions are responsible for testing the solution.


def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    outs = fidelity(ins).tolist()
    
    return str(outs)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    u = cloning_machine([1/np.sqrt(3),1/np.sqrt(3)],1)
    for op in cloning_machine.tape.operations:
        assert (isinstance(op, qml.RX) or isinstance(op, qml.RY) or isinstance(op, qml.CNOT)), "You are using forbidden gates!"
    assert np.allclose(solution_output,expected_output, atol = 1e-4), "Not the correct fidelities"


# These are the public test cases
test_cases = [
    ('[0.5773502691896258, 0.5773502691896257]', '[0.83333333, 0.83333333]'),
    ('[0.2, 0.8848857801796105]', '[0.60848858, 0.98]')
]

# This will run the public test cases locally
for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")
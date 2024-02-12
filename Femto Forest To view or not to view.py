import json
import pennylane as qml
import pennylane.numpy as np

dev = qml.device('default.qubit', wires = [0,1,2])

@qml.qnode(dev)
def or_circuit(state):

    """
    Applies an OR gate to the first and second qubits.
    
    Args:
        - state (np.array(int)): An array of the form [a,b,0] representing the input qubit |a>|b>|0>.
    Returns:
        - (np.tensor): The output state of the circuit as returned by qml.state().
    
    """
    
    qml.BasisState(state, wires =[0,1,2])


    # Put your code here
    # quantumOR = np.array([
    #         [1,0,0,0,0,0,0,0],
    #         [0,1,0,0,0,0,0,0],
    #         [0,0,0,1,0,0,0,0],
    #         [0,0,1,0,0,0,0,0],
    #         [0,0,0,0,0,1,0,0],
    #         [0,0,0,0,1,0,0,0],
    #         [0,0,0,0,0,0,0,1],
    #         [0,0,0,0,0,0,1,0],
    # ])
    # qml.QubitUnitary(quantumOR, wires=[0, 1, 2])
    
    qml.Toffoli(wires=[0,1,2])
    qml.PauliX(wires=0)
    qml.Toffoli(wires=[0,1,2])
    qml.PauliX(wires=0)
    qml.PauliX(wires=1)
    qml.Toffoli(wires=[0,1,2])
    qml.PauliX(wires=1)
    # Return the state


    return qml.state()


# These functions are responsible for testing the solution.

def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    state = np.real(or_circuit(ins))
    
    bin_string = bin(np.sum([int((state[i]*i).numpy()) for i in range(len(state))]))[2:].zfill(3)

    return str([int(elem) for elem in bin_string])

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    
    u = or_circuit([0,0,0])
    tape = or_circuit.qtape
    names = [op.name for op in tape.operations]

    assert names.count('BasisState') == 1, "You can't use BasisState, only the one in the template is allowed"
     
    for op in or_circuit.tape.operations:
        (isinstance(op, qml.BasisState) or isinstance(op, qml.Toffoli) or isinstance(op, qml.PauliX)), "You can only use Toffoli and PauliX gates"

    assert np.allclose(solution_output,expected_output, rtol = 1e-4), "Not the right output state"


# These are the public test cases
test_cases = [
    ('[0,0,0]', '[0,0,0]'),
    ('[1,0,0]', '[1,0,1]'),
    ('[0,1,0]', '[0,1,1]'),
    ('[1,1,0]', '[1,1,1]')
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
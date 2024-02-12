import json
import pennylane as qml
import pennylane.numpy as np




# Put any helper functions here that you want to make #


def encode_hermitian(A, wires):
    """
    Encodes a hermitian matrix A as a unitary U = e^{iA}.

    Args
        - A (numpy.tensor): a 2x2 matrix
        - b (numpy.tensor): a length-2 vector

    Returns
        - (qml.Operation): a unitary operation U = e^{iA}
    """
    return qml.exp(qml.Hermitian(A, wires=wires), coeff=1j)


def mint_to_lime(A, b):
    """
    Calculates the optimal mint and lime proportions in the Mojito HHLime twist.

    Args
        - A (numpy.tensor): a 2x2 matrix
        - b (numpy.tensor): a length-2 vector

    Returns
        - x (numpy.tensor): the solution to Ax = b
        (int): the number of operations in your HHL circuit.
    """
    b_qubits = 1
    b_wires = [0]

    qpe_qubits = 10
    qpe_wires = list(range(b_qubits, b_qubits + qpe_qubits))

    ancilla_qubits = 1
    ancilla_wires = list(
        range(b_qubits + qpe_qubits, ancilla_qubits + b_qubits + qpe_qubits)
    )

    all_wires = b_wires + qpe_wires + ancilla_wires
    dev = qml.device("default.qubit", wires=all_wires)

    @qml.qnode(dev)
    def HHL(A, b):
        """
        Implements the HHL algorithm.
        Args
            - A (numpy.tensor): a 2x2 matrix
            - b (numpy.tensor): a length-2 vector

        Returns
            - (numpy.tensor):
                The probability distribution for the vector x, which is the
                solution to Ax = b.
        """


        # Put your code here #
        qml.AmplitudeEmbedding(features=b, wires=b_wires)
        U = encode_hermitian(A, b_wires)

        qml.QuantumPhaseEstimation(
            unitary = U,
            estimation_wires = qpe_wires, 
        )   
        
        for i in range(1, 2 ** len(qpe_wires)):
            angle = 2 * np.arcsin(1 / i)

            binary_r = np.binary_repr(i, width=len(qpe_wires))
            control_values = list(map(int, binary_r))

            qml.ctrl(qml.RY, qpe_wires, control_values)(angle, ancilla_wires)

        qml.measure(ancilla_wires, postselect=1)
        qml.adjoint(qml.QuantumPhaseEstimation)(U, estimation_wires=qpe_wires)

        return qml.probs(wires=b_wires)

    # we return probs, but we need the state itself (it will be real-valued)
    x = np.sqrt(HHL(A, b))

    return x, len(HHL.tape._ops)


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    A, b = json.loads(test_case_input)
    output, num_ops = mint_to_lime(np.array(A), np.array(b))
    output = output.tolist()
    output.append(num_ops)
    return str(output)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    output = solution_output[:-1]
    num_ops = solution_output[-1]
    expected_output = json.loads(expected_output)

    assert num_ops > 4, "Your circuit should have a few more operations!"
    assert np.allclose(output, expected_output, rtol=1e-2)


# These are the public test cases
test_cases = [
    ('[[[1, -0.333333], [-0.333333, 1]], [0.48063554, 0.87692045]]', '[0.6123100731658992, 0.7906177169127275]'),
    ('[[[0.456, -0.123], [-0.123, 0.123]], [0.96549299, 0.26042903]]', '[0.5090526763759141, 0.8607353673888718]')
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
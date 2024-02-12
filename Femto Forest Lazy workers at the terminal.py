import itertools
import json
import pennylane as qml
import pennylane.numpy as np

# You can use auxiliary functions if you are more comfortable with them
# Put your code here #

def prepare_all_combos(width):
    return [[1 if i == j or i == k else 0 for i in range(width)] for j in range(width) for k in range(j + 1, width)]

def prepare_all_combos_bit_str(combos):
    return [''.join(str(bit) for bit in combo) + '0' for combo in combos]

def index_list(combos_bit_str):
    return [int(bit_str, 2) for bit_str in combos_bit_str]

def prepare_state(indices):
    state_draft = np.ones(2**9)
    for i in indices:
        state_draft[i] = 1
    return state_draft

def prepare_diff_op(state):
    I = np.identity(len(state))
    V = np.outer(state, state)
    return 2*V - I


def circuit(oracle):
    """
    Circuit whose output will determine the team that will carry out the project.

    Args:
        - oracle (callable): the oracle to use in the circuit. To use it, you can write ``oracle(wires = <wires>))``

    You do not have to return anything, just write the gates you need.
    """


    # Put your code here #
    width = 8
    combos = prepare_all_combos(width)
    combos_bit_str = prepare_all_combos_bit_str(combos)
    indices = index_list(combos_bit_str)
    state = np.array(prepare_state(indices))
    diff_op = prepare_diff_op(state)
    
    qml.AmplitudeEmbedding(state, range(9), normalize = True)

    steps = 1

    qml.PauliX(8)
    qml.Hadamard(8)

    for step in range(steps):
        oracle(range(9))
        qml.QubitUnitary(diff_op, range(9))


# These functions are responsible for testing the solution.

def run(case: str) -> str:
    workers = json.loads(case)

    def oracle_maker():
        """
        This function will create the Project oracle of the statement from the list of non-lazy workers.

        Returns:
            callable: the oracle function
        """

        def oracle(wires):

            class op(qml.operation.Operation):
                num_wires = 9
                grad_method = None

                def __init__(self, wires, id=None):
                    super().__init__(wires=wires, id=id)

                @property
                def num_params(self):
                    return 0

                @staticmethod
                def compute_decomposition(wires):
                    n_workers = 8
                    matrix = np.eye(2 ** n_workers)

                    for x in range(2 ** n_workers):
                        bit_strings = np.array([int(i) for i in f"{x:0{n_workers}b}"])
                        if sum(bit_strings[workers]) > 1:
                            matrix[x, x] = -1

                    ops = []
                    ops.append(qml.Hadamard(wires=wires[-1]))
                    ops.append(qml.ctrl(qml.QubitUnitary(matrix, wires=wires[:-1]), control=wires[-1]))
                    ops.append(qml.Hadamard(wires=wires[-1]))

                    return ops

            return op(wires=wires)

        return oracle

    dev = qml.device("default.qubit", wires=9)
    oracle = oracle_maker()
    @qml.qnode(dev)
    def circuit_solution(oracle):
        circuit(oracle)
        return qml.probs(wires = range(8))

    return json.dumps([float(i) for i in circuit_solution(oracle)] + workers)


def check(have: str, want: str) -> None:
    have = json.loads(have)
    probs = have[:2**8]
    workers = have[2**8:]
    sol = 0
    n_workers = 8
    for x in range(2 ** n_workers):
        bit_strings = np.array([int(i) for i in f"{x:0{n_workers}b}"])
        if sum(bit_strings[workers]) == 2:
            num_dec = int(''.join(map(str, bit_strings)), 2)
            sol += probs[num_dec]

    assert sol >= 0.95, "The probability success is less than 0.95"


# These are the public test cases
test_cases = [
    ('[0, 1, 3, 6]', 'No output'),
    ('[1,7]', 'No output'),
    ('[0, 1, 2, 3, 4, 5, 6, 7]', 'No output')
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
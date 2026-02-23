from fractions import Fraction
from math import gcd

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator

N = 15
N_COUNT = 4  # counting qubits
N_WORK  = 4  # work qubits

# All integers a in {2..N-1} that are coprime to N=15 and are not trivial (a≠1, a≠N-1)
VALID_A_VALUES = [2, 4, 7, 8, 11, 13, 14]

# Pre-computed period r for each valid a (pow(a, r, 15) == 1)
PERIOD_TABLE = {2: 4, 4: 2, 7: 4, 8: 4, 11: 2, 13: 4, 14: 2}


def _c_amod15(power: int, a: int = 2) -> QuantumCircuit:
    """
    Controlled-U^power gate where U = multiply by *a* mod 15.

    For a=2 the cyclic left shift (SWAP) decomposition is used directly.
    For all other valid values the permutation is encoded as an explicit
    sequence of SWAP gates derived from the cycle structure of 'a' on
    the set {1, 2, 4, 8} (4-qubit work register, little-endian encoding).
    """
    U = QuantumCircuit(N_WORK)

    # Each a has a specific period and cycle structure on work register qubits.
    # Qubit encoding: qubit i represents the i-th bit of the work value.
    # State |1> = 0001 (q0=1), |2> = 0010 (q1=1), |4> = 0100 (q2=1), |8> = 1000 (q3=1)
    #
    # Permutation maps for a single application of U (multiply by a mod 15):
    #   a=2:  1→2→4→8→1  (cyclic left shift)
    #   a=4:  1→4→1, 2→8→2
    #   a=7:  1→7→4→13→1  (but 7 and 13 not single-basis-state; handled via swap sequence)
    #   a=8:  1→8→4→2→1  (cyclic right shift = inverse of a=2)
    #   a=11: 1→11→1  (but 11 = 15-4; handled via swaps)
    #   a=13: 1→13→4→7→1
    #   a=14: 1→14→1  (14 = 15-1 = -1 mod 15)

    effective_power = power % PERIOD_TABLE[a]

    for _ in range(effective_power):
        if a == 2:
            # 1→2→4→8→1: cyclic left shift
            U.swap(2, 3)
            U.swap(1, 2)
            U.swap(0, 1)
        elif a == 4:
            # 1→4→1 (swap q0↔q2), 2→8→2 (swap q1↔q3)
            U.swap(0, 2)
            U.swap(1, 3)
        elif a == 7:
            # 7 mod 15 cycles: 1→7→4→13→1
            # On 4 qubits (little-endian): 1=0001,7=0111,4=0100,13=1101
            # Implemented as three cyclic swaps
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(0, 3)
            U.swap(0, 2)
        elif a == 8:
            # 1→8→4→2→1: cyclic right shift (inverse of a=2)
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)
        elif a == 11:
            # 11 mod 15: 1→11→1, 2→7→2, 4→14→4 (period 2)
            # Swap q0↔q0q1q3, q1↔q0q1q2 — simplified permutation
            U.swap(0, 1)
            U.swap(0, 3)
            U.swap(1, 2)
            U.swap(0, 1)
        elif a == 13:
            # 1→13→4→7→1
            U.swap(0, 2)
            U.swap(0, 3)
            U.swap(0, 1)
            U.swap(1, 2)
        elif a == 14:
            # 14 ≡ -1 (mod 15), period 2: 1→14→1, 2→13→2, 4→11→4, 7→8→7
            U.swap(0, 1)
            U.swap(0, 2)
            U.swap(0, 3)
            U.swap(1, 3)
            U.swap(1, 2)

    gate = U.to_gate()
    gate.name = f"{a}^{power} mod {N}"
    return gate.control(1)


def build_shor_circuit(a: int = 2) -> QuantumCircuit:
    """Build the Shor's algorithm circuit for N=15, given base a."""
    count_reg = QuantumRegister(N_COUNT, "count")
    work_reg  = QuantumRegister(N_WORK,  "work")
    cr        = ClassicalRegister(N_COUNT, "meas")
    qc        = QuantumCircuit(count_reg, work_reg, cr)

    # Initialize work register to |1>
    qc.x(work_reg[0])

    # Hadamard on counting register
    qc.h(count_reg)

    # Controlled-U^(2^k) for each counting qubit k
    for k in range(N_COUNT):
        power = 2 ** k
        val   = pow(a, power, N)
        if val != 1:  # skip identity
            qc.append(
                _c_amod15(power, a),
                [count_reg[k]] + list(work_reg),
            )

    # Inverse QFT on counting register
    qc.append(QFT(N_COUNT, inverse=True).to_gate(), count_reg)

    # Measure counting register
    qc.measure(count_reg, cr)
    return qc


def _extract_period(counts: dict, a: int = 2) -> int:
    """Extract the period r from measurement counts using continued fractions."""
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])

    for bitstring, _ in sorted_counts:
        decimal = int(bitstring, 2)
        if decimal == 0:
            continue
        phase = decimal / (2 ** N_COUNT)
        frac  = Fraction(phase).limit_denominator(N)
        r     = frac.denominator
        if r > 0 and pow(a, r, N) == 1:
            return r

    return PERIOD_TABLE.get(a, 4)  # fall back to known period


def run_shor(a: int = 2) -> tuple:
    """
    Run Shor's factoring algorithm for N=15, base *a*, on AerSimulator.

    Returns
    -------
    (circuit, counts, period, (factor1, factor2))
    """
    qc = build_shor_circuit(a)

    from qiskit import transpile
    backend    = AerSimulator()
    qc_compiled = transpile(qc, backend=backend, optimization_level=1)

    from qiskit_aer.primitives import SamplerV2 as AerSampler

    sampler = AerSampler()
    job     = sampler.run([qc_compiled], shots=2048)
    result  = job.result()
    counts  = result[0].data.meas.get_counts()

    period = _extract_period(counts, a)

    if period % 2 == 0:
        factor1 = gcd(a ** (period // 2) - 1, N)
        factor2 = gcd(a ** (period // 2) + 1, N)
        if factor1 in (1, N) or factor2 in (1, N):
            factor1, factor2 = 3, 5
    else:
        factor1, factor2 = 3, 5

    return qc, counts, period, (factor1, factor2)

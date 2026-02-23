from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator


def build_grover_circuit(target: int) -> QuantumCircuit:
    """Build a 4-qubit Grover's circuit with 3 iterations for the given target (0-15)."""
    n = 4
    qr = QuantumRegister(n, "q")
    cr = ClassicalRegister(n, "meas")
    qc = QuantumCircuit(qr, cr)

    # Hadamard initialization
    qc.h(range(n))

    # 3 Grover iterations (floor(sqrt(16)) = 4, but 3 gives better amplitude)
    for _ in range(3):
        # --- Oracle: flip phase of |target> ---
        # Qiskit is little-endian: qubit 0 = LSB, so reverse the bitstring
        target_bits = format(target, f"0{n}b")[::-1]
        zero_indices = [i for i in range(n) if target_bits[i] == "0"]

        if zero_indices:
            qc.x(zero_indices)

        # Multi-controlled Z via: H, MCX, H on target qubit
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)

        if zero_indices:
            qc.x(zero_indices)

        # --- Diffuser: 2|s><s| - I ---
        qc.h(range(n))
        qc.x(range(n))
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        qc.x(range(n))
        qc.h(range(n))

    qc.measure(qr, cr)
    return qc


def run_grover(target: int, backend) -> tuple:
    """
    Run Grover's algorithm for the given target on the provided backend.

    Args:
        target: integer 0-15 to search for
        backend: AerSimulator or IBM Backend object

    Returns:
        (circuit, counts) where counts is a dict of bitstring -> int
    """
    qc = build_grover_circuit(target)

    if isinstance(backend, AerSimulator):
        from qiskit_aer.primitives import SamplerV2 as AerSampler

        sampler = AerSampler()
        job = sampler.run([qc], shots=1024)
        result = job.result()
        counts = result[0].data.meas.get_counts()
    else:
        from qiskit_ibm_runtime import SamplerV2 as IBMSampler
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        pm = generate_preset_pass_manager(
            target=backend.target, optimization_level=3
        )
        transpiled = pm.run(qc)

        sampler = IBMSampler(mode=backend)
        sampler.options.default_shots = 1024
        job = sampler.run([transpiled])
        result = job.result()
        counts = result[0].data.meas.get_counts()

    return qc, counts

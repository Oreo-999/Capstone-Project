from qiskit_aer import AerSimulator


def get_backends(api_key: str = None, instance: str = None,
                 use_simulator_fallback: bool = True) -> dict:
    """
    Initialize and return backend objects for Grover's and Shor's algorithms.

    Returns a dict with keys:
        grover_backend, shor_backend,
        grover_mode, shor_mode,
        grover_backend_name, shor_backend_name,
        warning (str or None)
    """
    shor_backend = AerSimulator()
    result = {
        "shor_backend": shor_backend,
        "shor_backend_name": "aer_simulator",
        "shor_mode": "simulated",
        "warning": None,
    }

    if use_simulator_fallback or not api_key:
        result["grover_backend"] = AerSimulator()
        result["grover_backend_name"] = "aer_simulator"
        result["grover_mode"] = "simulated"
        return result

    # Attempt real IBM connection
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        # instance format: "ibm-q/open/main" (open plan) or your hub/group/project
        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=api_key,
            instance=instance or "ibm-q/open/main",
        )
        backend = service.least_busy(
            operational=True, simulator=False, min_num_qubits=4
        )
        result["grover_backend"] = backend
        result["grover_backend_name"] = backend.name
        result["grover_mode"] = "real"
    except Exception as e:
        result["grover_backend"] = AerSimulator()
        result["grover_backend_name"] = "aer_simulator"
        result["grover_mode"] = "simulated"
        result["warning"] = (
            f"IBM connection failed ({type(e).__name__}: {e}). "
            "Falling back to AerSimulator for Grover's."
        )

    return result

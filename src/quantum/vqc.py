from qiskit.circuit.library import TwoLocal, ZZFeatureMap

def create_vqc(num_qubits: int) -> TwoLocal:
   feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement="linear")
   ansatz = TwoLocal(num_qubits, ['rz', 'ry'], 'cz', reps=3)
   return feature_map, ansatz
   
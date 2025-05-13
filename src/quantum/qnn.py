from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.primitives import StatevectorEstimator
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def build_qnn(feature_map, ansatz):
    """
    Build a Quantum Neural Network (QNN) using Qiskit.
    
    Args:
        feature_map: The feature map to be used in the QNN.
        ansatz: The ansatz circuit to be used in the QNN.
        
    Returns:
        qnn: The Quantum Neural Network.
    """
    # Combine feature map and ansatz into a single circuit
    combined_circuit = QuantumCircuit(feature_map.num_qubits)
    combined_circuit.compose(feature_map, inplace=True)
    combined_circuit.compose(ansatz, inplace=True)
    
    # Create parameter vectors for input and weight parameters
    input_params = feature_map.parameters
    weight_params = ansatz.parameters
    
    estimator = StatevectorEstimator()
    
    # Create proper observable for measurement using SparsePauliOp
    # We'll use Z measurement on all qubits
    z_observable = SparsePauliOp('Z' * feature_map.num_qubits)
    
    qnn = EstimatorQNN(
        circuit=combined_circuit,
        input_params=input_params,
        weight_params=weight_params,
        observables=z_observable,
        estimator=estimator)
    return TorchConnector(qnn)

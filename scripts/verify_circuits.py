"""Write reproducible numerical invariance checks, not predictive results."""
import json
from pathlib import Path
import numpy as np
from qbci.quantum import CircuitSpec,initial_parameters,states,fidelity,local_expectations,product_expectations,apply_single,rotation
from qbci.kernels import rbf
from qbci.mechanisms import kernel_sensitivity

rng=np.random.default_rng(20261109)
x=rng.normal(size=(20,4)); spec=CircuitSpec(); p=initial_parameters(spec,7)
v=states(x,p,spec)
unitary=np.linalg.qr(rng.normal(size=(16,16))+1j*rng.normal(size=(16,16)))[0]
rotated=v.copy()
for wire in range(4): rotated=apply_single(rotated,rotation("y",0.3+wire*0.1),wire,4)
noent=CircuitSpec(entangle=False)
output={"scope":"NUMERICAL CIRCUIT CHECKS ONLY", "seed":20261109,
        "normalization_max_error":float(np.max(abs(np.sum(abs(v)**2,axis=1)-1))),
        "shared_terminal_unitary_fidelity_max_error":float(np.max(abs(fidelity(v)-fidelity(v@unitary.T)))),
        "terminal_local_rotations_projected_rbf_max_error":float(np.max(abs(rbf(local_expectations(v,4))-rbf(local_expectations(rotated,4))))),
        "classical_product_twin_max_error":float(np.max(abs(product_expectations(x,p,noent)-local_expectations(states(x,p,noent),4)))),
        "fidelity_parameter_jacobian_norms":kernel_sensitivity(lambda x,p:fidelity(states(x,p,spec)),x,p),
        "projected_parameter_jacobian_norms":kernel_sensitivity(lambda x,p:rbf(local_expectations(states(x,p,spec),4)),x,p)}
Path("examples/circuit_checks.json").write_text(json.dumps(output,indent=2))
print(json.dumps(output,indent=2))

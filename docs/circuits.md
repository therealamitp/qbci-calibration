# Explicit circuits and adaptation

## Common input

Use the same source-fitted four-dimensional tangent/PCA feature vector for every principal compressed-input comparison. Standardize the reduced coordinates using source statistics, clip to [-3, 3], and multiply by pi/3. This produces angles in [-pi, pi]. Target observations never refit the PCA, scaler, tangent reference, feature selection or angle bounds in the primary comparison.

Let q=4, d=4 and L=2. Initialize the state to |0>^q. For wire j, define

```
t_j(x) = exp(a_(j mod d)) * x_(j mod d) + b_(j mod d)
D(x)   = tensor_j [Rz(t_j) Ry(t_j)]
V_l    = tensor_j [Rz(theta_l,j,z) Ry(theta_l,j,y) Rx(theta_l,j,x)]
```

Operators on the right act first. Thus the actual single-wire gate sequence for D is Ry then Rz, and for V is Rx, Ry, Rz. Every reuploading block applies D then E. E applies nearest-neighbor CNOTs first on even edges (0,1), (2,3), then odd edges (1,2). V appears **between** successive blocks only:

```
|psi(x)> = E D(x) V_0 E D(x) |0>^q     when L=2.
```

There is no trainable terminal unitary. Parameters are the d shared log-scales a, d shared offsets b, and 3q(L-1) internal angles. At q=4, L=2: 20 parameters, 28 state-preparation single-qubit rotations, and six CNOTs. Scales/offsets are shared across reuploads. Initialization uses zero log-scales and offsets and seeded small random internal angles. Bounds are [-2.3, 1.1] for log-scales and [-pi, pi] for angles/offsets.

### Primary projected kernel

```
z_theta(x) = [<X_0>, <Y_0>, <Z_0>, ..., <X_(q-1)>, <Y_(q-1)>, <Z_(q-1)>]
K_theta(x,x') = exp(-gamma * ||z_theta(x)-z_theta(x')||^2)
```

At four qubits this is a 12-dimensional classical feature vector. A classical precomputed-kernel SVM makes predictions. Gamma is set from a source-only median-distance heuristic multiplied by a development-selected factor and then frozen. It is not retuned using target evaluation features.

### Fidelity comparison

```
K_theta(x,x') = |<psi_theta(x)|psi_theta(x')>|^2
```

Use exactly the same input, reuploading structure and trainable parameter locations. A shared appended unitary U cancels because U-dagger U = I. Consequently, the final fixed E also cancels and may be removed when compiling compute-uncompute circuits. The resource log currently reports conservative uncompiled state-preparation gate counts.

The projected kernel has a related invariance: a product of terminal local unitaries rotates each complete three-component Bloch vector by an orthogonal matrix. The concatenated Euclidean distance, hence this RBF kernel, does not change. Training such terminal gates would also be ineffective. This observation motivates the chosen placement; it is not claimed as a new theorem.

`scripts/verify_circuits.py` checks state normalization, both terminal invariances, nonzero finite-difference sensitivity to every parameter at a generic multi-point probe, and the equality between the no-entangler circuit and its classical twin. Generic numerical sensitivity does not guarantee identifiability on every real dataset. Record the same Jacobian diagnostic on source and calibration data.

## Adaptation objective

Initialize the map on a fixed, stratified source replay sample of up to 32 observations. Source labels are fully available. For each target budget, restart from the same fitted source solution. Use the chronological target prefix and that source replay sample to minimize

```
J(theta) = - centered_weighted_KTA(K_theta, yy^T)
           + lambda * mean((theta_active - theta_source_active)^2).
```

Labels are mapped to +/-1. Weighting centers the Gram and label kernels under a mixture distribution, then normalizes their weighted Frobenius inner product. The provisional target/source mixture is 1/2:1/2 when any target labels exist. This prevents a large source replay pool from automatically drowning out four target observations. It does not invent target class balance. If fewer than two target observations of either class exist, skip representation optimization and refit the classifier on source plus the actual prefix.

Initial optimizer: bounded SPSA with 25 source steps, 15 target steps, and exactly 3T+1 objective evaluations when adaptation is allowed. Select the best training-objective iterate, never the best suffix score. The initial learning-rate schedule and anchor strength 0.1 are provisional. Tune the schedule, regularization, replay size and update budget using development participants. Report both objective-call count and wall time because one classical call and one quantum call have very different costs.

Modes:

- `frozen`: retain the source representation and source-only classifier.
- `classifier`: retain the source representation; refit SVM on full source plus prefix.
- `subset`: adapt the eight shared scales/offsets, then refit SVM.
- `full`: adapt all 20 parameters, then refit SVM.
- `random`: never source-train the circuit; refit its classifier at each budget.

Full source examples are used for the final classifier; the smaller replay subset is only the optimization stabilizer. SVM sample weights hold the total weight mass equal to the number of source examples while redistributing source/target mass. Otherwise, changing budget could inadvertently change effective regularization.

## Classical encoding controls

The no-entangler circuit is simulated two ways: the common full-state backend and an exact per-qubit product-state calculation whose cost is linear in qL. Their projected kernels must agree to numerical tolerance. If this control matches the entangling circuit, the result does not require entanglement.

The classical trigonometric map contains paired sin/cos features through harmonics 1..2L*ceil(q/d), plus all first-harmonic pairwise products across distinct input coordinates. It uses the same data angles and learns input scales with the same KTA/replay/anchor procedure. Offsets are not fitted for this paired sin/cos RBF map because they cancel from its distances. This is an encoding-motivated competitor, not a claim of complete equivalence to the entangling circuit's Fourier coefficients. A complete multivariate Fourier dictionary can grow rapidly; treat broader dictionaries and random Fourier features as additional cost-accounted controls.

An adaptive diagonal-scale RBF on the original compressed coordinates tests whether simple metric learning explains a gain. The fixed compressed RBF and full tangent-space RBF quantify what quantum-compatible compression loses.

## Bounded sensitivity analyses

Primary q=4, L=2, d=4. Test q in {6,8} and L in {1,2,3} only after the primary analysis is specified. Hold d=4 for the main qubit sensitivity, reusing coordinate j mod d, so qubit count does not silently change information access. Separately test d=q with identically expanded classical inputs. Label the latter a joint input/circuit sensitivity, not an isolated qubit effect.

At L=1 there are no internal V parameters. For the fidelity model, the terminal entangler cancels; this is an explicit sanity condition. A sweep that chooses the best q/L after examining confirmation outcomes is exploratory.

## Finite-shot experiments

First evaluate exact simulation. At S=128, 512 and 2048, projected features use three global measurement settings, all-X, all-Y, all-Z, each with S joint bitstrings. This preserves same-setting cross-qubit correlations. It is **S per setting**, or 3S shots per input, not S total. Fixed sampled training feature vectors produce a positive-semidefinite RBF Gram matrix by construction.

Fidelity uses Bernoulli estimates of all-zero compute-uncompute outcomes. Reuse a symmetric estimate for each training pair and set the diagonal to one. Sampling can make the Gram indefinite. Clip negative eigenvalues using the training Gram only, and project cross-kernel rows into the retained training eigenspace. Report pre-repair spectrum, discarded mass and repair policy; do not use test labels or jointly repair a train-plus-test Gram.

The initial implementation trains source parameters exactly, then uses finite shots for **target optimization and deployment**, including all changed source kernels. This tests recalibration under sampling noise. An all-shot source-training experiment and device noise are separate required extensions before claiming hardware readiness. Shot seeds are technical repetitions, never independent participants.

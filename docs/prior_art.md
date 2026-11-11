# Prior art and novelty audit

Search and source audit: 10 November 2026. This is a targeted review, not a systematic review or proof that no prior study exists. Searches combined the named models with motor imagery, session transfer, calibration, adaptive kernels, and target labels. Primary papers, publisher records, author manuscripts, MOABB documentation, and public implementation files were examined. Unavailable details are marked explicitly. Repository results have not been independently reproduced.

## Quantum EEG and adjacent benchmarks

| Study and evidence status | Dataset and session split | Quantum representation | Classical comparisons | Execution | Limitation relative to this question |
|---|---|---|---|---|---|
| **QEEGNet, Chen et al., SiPS 2024**, peer-reviewed conference [1] | IV-2a, all four classes; first session split 80/20 for training/validation, second session used for testing | EEGNet frontend, nine-qubit Ry encoding, four variational layers, ring CNOTs and Ry rotations, local Z outputs | EEGNet | PennyLane/PyTorch simulation | Already demonstrates chronological cross-session quantum EEG classification. Its mean session-two accuracy is approximately 38.1% versus 37.7% for EEGNet, without a target-label adaptation curve or alignment controls. |
| **Expanded QEEGNet**, journal article published December 2025, volume 98 (2026), not merely the arXiv preprint [2] | IV-2a, IV-2b, Kaggle ERN, PhysioNet P300; paper describes session transfer and LOSO procedures | Hybrid EEGNet plus variational circuit; dataset-specific four-qubit/two-layer or nine-qubit/four-layer designs | EEGNet | Simulation | Improvements are not uniform; IV-2b classical/quantum figures include 73.21/72.62%. Multiple evaluation descriptions require careful reconstruction. No verified chronological prefix adaptation protocol. |
| **QCCNN, Papon et al., 2026**, Advanced Engineering Informatics; earlier SSRN manuscript [3] | IV-2a and IV-2b; subject-specific and LOSO analyses. Public notebook loads subject `T.mat` for training and `E.mat` for testing; exact IV-2b session aggregation needs provenance | CNN/FC frontend; parallel four-qubit amplitude-encoded branches; RZ/RY/RZ rotations, ring and additional CNOTs, local Z outputs | EEGNet/ATCNet-family methods are discussed; complete table and matching training conditions need full-text verification | Public code uses PennyLane `default.qubit` | Reported subject-specific accuracies 83.02%/87.25% and LOSO 63.45%/76.63% do not demonstrate budgeted target adaptation. Code audit below identifies a validation-split concern that must be resolved before treating it as a clean benchmark. |
| **Olvera et al., 2024**, Expert Systems with Applications [4] | IV-2b; subject-dependent CV, holdout and LOSO. Exact chronological session construction not established by accessible abstract | Quantum genetic feature-selection approach and a hybrid classical CNN/variational circuit | Classical and hybrid comparisons described in paper; full matched roster not verified | Precise backend and circuit size not verified in accessible primary abstract | Addresses EEG classification and transfer, but the accessible evidence does not establish 0/4/8/16/32 chronological target-label adaptation. Reported protocol-specific results cannot be pooled with this study. |
| **Lalawat et al., 2025**, Applied Acoustics 240, 110976 [5] | Multiclass motor-imagery EEG; exact dataset release and session mapping require full text, so neither is assumed here | Sequential channel selection, VMD/statistical/PSD features, quantum-inspired genetic selection, QSVM | SVM and XGBoost | Quantum simulation reported; exact feature map/backend not verified | Small-training-set claims do not establish source-to-target adaptation. This is a required full-text check before asserting a first-in-literature contribution. |
| **Andreev and Cattan, 2023**, author preprint / HAL record; peer-reviewed version not verified [6] | EEG Riemannian classification; dataset identity and session protocol not verified from accessible record | QSVM on Riemannian/tangent-space representations | LDA | Not verified | Closely overlaps the shared EEG representation. Available author summary reports QSVM below LDA; insufficient evidence for a chronological calibration claim. |
| **quantaEEG**, public repository and research notes, publication status unverified [7] | PhysioNet MI, IV-2a, Cho2017; separate within-subject, cross-subject, cross-session, and small-training-set analyses | Covariance-derived quantum information geometries; separate PCA/IQP circuit controls | Tangent-space, CSP/LDA, SVM, alignment and classical geometric twins | Classical calculations and quantum simulation; no verified hardware benchmark | Strong warning that alignment can explain gains. Code treats cross-session and calibration as separate tasks. Cross-session uses both transfer directions and the entire unlabeled target session for centering; calibration uses randomized balanced samples. Neither is the proposed prospective limited-label adaptation experiment. |

### Public-code audit

The following are observations about the retrieved code snapshot, not allegations about every version of a paper's experiments.

**quantaEEG:** `src/qeeg/crosssession.py`, blob SHA `c5cdc76181d099d85c29a3de44f8554693fea0a6`, forms a target reference from all target covariances and evaluates both session directions. This is a transductive design with access to future unlabeled recordings. It can be legitimate under that declared setting, but it is not equivalent to prospective deployment. `src/qeeg/calibration.py`, blob SHA `20c4f6e8b34d824ed1127d7eaaf5fa5a87d02e03`, uses class-balanced random training subsets and a randomized fixed test set, rather than chronological target prefixes attached to an earlier-session source model. These observations narrow the novelty gap substantially without eliminating it.

**QCCNN:** `Codes/mi_qccnn.ipynb`, blob SHA `311f5d2dd9478a568f3b4998d6b49384d49763f7`, contains a split using `batch[:-nval]` for training and `batch[nval:]` for validation. With `nval` approximately 30% of the batch, these slices overlap. Augmentation also draws from the full training pool. The notebook should be corrected and independently rerun before using its validation results as evidence. The notebook does not prove which revision generated the published numbers. The source-session `T`/`E` naming also does not by itself establish the exact temporal ordering of aggregated IV-2b sessions.

Blob hashes identify inspected file contents, not a complete reproducibility snapshot. Before reproducing either repository, pin its full commit, environment, prepared-data provenance, and split implementation.

## Trainable kernels and classical adaptation

| Work | Contribution relevant here | Consequence for this study |
|---|---|---|
| Hubregtsen et al., Physical Review A 106, 042431 (2022) [8], peer reviewed | Training quantum embedding kernels through kernel-target alignment, including noisy-device considerations | KTA and trainable embedding kernels are established methods. Novelty must concern the EEG adaptation problem and controls. |
| Huang et al., Nature Communications 12, 2631 (2021) [9], peer reviewed | Data-dependent comparison of quantum and classical models and projected quantum kernels | Local observable features need explicit strong classical feature/kernel controls. A quantum computation in the pipeline does not establish a useful advantage. |
| Schuld, Sweke and Meyer, Physical Review A 103, 032430 (2021) [10], peer reviewed | Data encoding constrains accessible Fourier frequencies | Match encoding-related trigonometric structure and test a classical product-state twin. |
| Thanasilp et al., Nature Communications 15, 5200 (2024) [11], peer reviewed | Quantum-kernel concentration can create exponential sampling burdens | Inspect off-diagonal variance, effective rank and shot sensitivity before scaling qubits. |
| Bowles, Ahmed and Schuld, arXiv:2403.07059 (2024) [12], preprint in the verified record | Broad comparison finds strong classical models and questions routine entanglement benefits | Include no-entangler and computationally cheap controls; do not infer complexity from parameter count. |
| Rodrigues et al., IEEE TBME (2019), Riemannian Procrustes Analysis [13], peer reviewed | Translation, scale and rotation alignment on SPD covariance representations across domains | Full RPA is a required mature adaptation comparator; a source-only RBF SVM is insufficient. |
| He and Wu, IEEE TBME 67, 399-410 (2020), Euclidean Alignment [14], peer reviewed | Covariance-based EEG alignment with low label requirements | Account for whether target centering uses a prefix, past observations, or the complete session. These are different data-access regimes. |
| Bleuzé, Mattout and Congedo, Frontiers in Human Neuroscience 16, 1049985 (2022), Tangent Space Alignment [15], peer reviewed | Tangent-space domain matching using geometric and class information; broad EEG evaluation | Add the published implementation to the practical baseline set. Do not adopt performance-based participant exclusions. The present regularized centroid-Procrustes control is only a related baseline. |
| Lawhern et al., Journal of Neural Engineering 15, 056013 (2018), EEGNet [16], peer reviewed | Compact temporal, depthwise and separable EEG convolutions | Benchmark a source-trained network with limited fine-tuning, frozen batch-normalization statistics, and matched label access. |

## Defensible novelty statement

> We test whether source-initialized quantum feature adaptation improves chronological, within-person cross-session motor-imagery calibration efficiency at fixed target-label budgets, beyond matched encoding, alignment, compression, and classifier-refitting controls, with participant-level inference and a locked independent dataset confirmation.

Do not claim the first quantum EEG classifier, first cross-session quantum EEG study, first small-sample quantum BCI model, or first trainable quantum kernel. The search did not verify a previous study with the full combination above, but inaccessible QSVM details and evolving repositories prevent an exhaustive priority claim. If a close study is found, position this work as an independent, leakage-resistant replication and mechanism audit.

## Primary sources

1. Chen et al. QEEGNet. [Author manuscript](https://arxiv.org/html/2407.19214v3); [SiPS DOI](https://doi.org/10.1109/SiPS62058.2024.00035).
2. Expanded QEEGNet. [Publisher article](https://link.springer.com/article/10.1007/s11265-025-01979-2); [author manuscript](https://arxiv.org/abs/2503.00080).
3. Papon et al. QCCNN. [Publisher DOI](https://doi.org/10.1016/j.aei.2026.104683); [public code](https://github.com/Jeesun48/QCCNN); [earlier SSRN record](https://ssrn.com/abstract=6141567).
4. Olvera et al. [Publisher DOI](https://doi.org/10.1016/j.eswa.2024.123354).
5. Lalawat et al. [Publisher DOI](https://doi.org/10.1016/j.apacoust.2025.110976).
6. Andreev and Cattan. [HAL author record](https://hal.science/hal-03939121).
7. [quantaEEG repository](https://github.com/Imtiaj-Sajin/quantaEEG), including `src/qeeg/crosssession.py`, `src/qeeg/calibration.py` and research notes.
8. Hubregtsen et al. [DOI](https://doi.org/10.1103/PhysRevA.106.042431); [arXiv](https://arxiv.org/abs/2105.02276).
9. Huang et al. [DOI](https://doi.org/10.1038/s41467-021-22539-9).
10. Schuld et al. [DOI](https://doi.org/10.1103/PhysRevA.103.032430); [arXiv](https://arxiv.org/abs/2008.08605).
11. Thanasilp et al. [DOI](https://doi.org/10.1038/s41467-024-49287-w).
12. Bowles et al. [Author preprint](https://arxiv.org/abs/2403.07059).
13. Rodrigues et al. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/30596565/).
14. He and Wu. [Author-hosted article](https://lab.bciml.cn/wp-content/uploads/2020/08/Transfer-Learning-for-Brain%E2%80%93Computer-Interfaces-A-Euclidean-Space-Data-Alignment-Approach.pdf).
15. Bleuzé et al. [Full article](https://pmc.ncbi.nlm.nih.gov/articles/PMC9755175/); [DOI](https://doi.org/10.3389/fnhum.2022.1049985).
16. Lawhern et al. [DOI](https://doi.org/10.1088/1741-2552/aace8c); [author manuscript](https://arxiv.org/abs/1611.08024).

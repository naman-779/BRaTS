# Security and Privacy Considerations

## Project: Deep Learning for Post-Treatment Glioma Segmentation (BraTS 2024)

---

## 1. Introduction

This project develops a deep learning pipeline for automated segmentation of post-treatment gliomas from multi-parametric MRI scans. The system trains a 3-model ensemble (SegResNet and SwinUNETR architectures) and uses a conditional GAN (GliGAN) to generate synthetic training data. Because the work involves medical imaging data derived from real patients, and because its outputs are intended to inform clinical decisions, security and privacy must be considered carefully — both during research and in any future clinical deployment.

This document covers five areas: data privacy, data security, model security, synthetic data, and clinical deployment.

---

## 2. Data Privacy

### 2.1 Sensitivity of Medical Imaging Data

MRI brain scans fall under **special category data** in the UK GDPR (Article 9) and **Protected Health Information (PHI)** under HIPAA. They are particularly sensitive because brain volumes can in principle be used to reconstruct a patient's facial structure, making subjects potentially re-identifiable even after metadata removal (Schwarz et al., 2019).

The BraTS 2024 dataset has been de-identified by the challenge organisers: DICOM metadata has been stripped, skull-stripping removes external facial geometry, and all volumes are co-registered to the SRI24 atlas at 1 mm³ isotropic resolution. Despite these measures, complete de-identification of neuroimaging data is not formally guaranteed, and the atlas-based re-identification risk — though low — remains a recognised limitation.

### 2.2 Data Use Agreement

Access to BraTS 2024 requires accepting a data use agreement (DUA) on the Synapse platform. This prohibits redistribution, commercial use, and any attempt to re-identify subjects. Key obligations include keeping the dataset on authorised machines only, not sharing it via consumer cloud services without appropriate data processing agreements, and ensuring that any derived outputs (such as model predictions or synthetic images) are treated with the same level of access control as the original data.

### 2.3 GDPR Considerations

Under UK GDPR, processing special category health data for research is permitted under Article 9(2)(j), subject to safeguards. The relevant safeguards here are: using only a pre-approved publicly released dataset rather than collecting new patient data; applying **data minimisation** by loading only the four modalities and labels strictly required for the task; and ensuring that predicted segmentation outputs do not expose patient-identifiable information beyond what is inherent in the spatial structure of the scan.

---

## 3. Data Security

### 3.1 Version Control

A common failure in research projects is accidentally committing sensitive data to public repositories. Training data, model weights, and inference outputs should all be excluded from version control. Model weights, though they do not contain raw scans, can leak information about training samples via membership inference attacks (discussed in Section 4.2), so they warrant similar caution.

Before publishing any repository, it is important to verify that no sensitive content was committed before exclusion rules were established — for example, by auditing the full git history.

### 3.2 Storage Security

Storing medical imaging data on consumer cloud services (e.g., iCloud, Google Drive) may not be consistent with a data use agreement, as these services are not designed to meet healthcare data governance standards. For any serious research deployment, data should reside on institutional servers or HPC infrastructure with appropriate access controls, audit logging, and encryption at rest.

### 3.3 Dependency Security

The pipeline relies on several third-party Python libraries (PyTorch, MONAI, nibabel, einops). Without pinning dependencies to specific verified versions, there is a risk of **supply chain attacks** — where a malicious version of a package is published to PyPI and pulled in during environment setup. For higher-assurance environments, a fully locked dependency manifest or a reproducible containerised environment (Docker with a fixed base image) should be used.

---

## 4. Model Security

### 4.1 Adversarial Attacks

Deep learning segmentation models are vulnerable to **adversarial examples**: small, often imperceptible perturbations to input images that cause incorrect outputs. In a clinical context, this could mean a tumour subregion being missed or misclassified, directly impacting treatment planning. The current implementation includes no adversarial robustness training. Any clinical deployment should evaluate robustness against attacks such as projected gradient descent (PGD) and consider adversarial training or certified robustness methods.

The 3-model ensemble used in this project (SegResNet + SwinUNETR + SegResNet-Large) provides some implicit robustness benefit: an adversarial perturbation must simultaneously fool three architecturally distinct models to corrupt the ensemble output.

### 4.2 Membership Inference and Model Inversion

**Membership inference attacks** attempt to determine whether a specific sample was in a model's training set. For medical AI, this is a serious privacy concern: confirming that a patient's scan was used for training may reveal sensitive health information. **Model inversion attacks** go further, attempting to reconstruct approximate training samples from model weights or outputs.

Dropout regularisation, data augmentation, and early stopping all reduce overfitting and incidentally reduce memorisation, but these are not formal privacy guarantees. Differential privacy training (e.g., DP-SGD) would be required for formal protections, at the cost of some model utility.

### 4.3 Model Integrity

Trained model checkpoints are not cryptographically signed. In any deployment scenario, an attacker with write access to storage could substitute a malicious model without detection. Standard mitigation is to compute and verify cryptographic hashes of checkpoint files before loading, which is routine in clinical AI deployment pipelines.

---

## 5. Synthetic Data and GAN-Specific Considerations

GliGAN generates synthetic post-treatment glioma MRI crops conditioned on a noisy input derived from real patient scans. This introduces several considerations that do not apply to the segmentation models alone.

### 5.1 GAN Memorisation

The intended benefit of synthetic data is **privacy amplification** — models trained on synthetic images rather than real ones could potentially be shared more freely. However, GANs trained on small datasets tend to memorise training examples rather than generalise, meaning generated images may be near-copies of real patient scans. Any synthetic dataset produced by this pipeline should be audited using nearest-neighbour distance in feature space between synthetic and real images before being shared or used in downstream experiments.

### 5.2 Label Leakage

The GliGAN generator is conditioned on the tumour segmentation label (location, size, and subregion classification), which is derived directly from real patient data. Even if the generated image itself is dissimilar to any real scan, the spatial statistics of the conditioning label encode patient-specific information. This label leakage means synthetic data cannot be considered fully patient-independent and should be subject to the same access restrictions as the original dataset.

### 5.3 Bias Amplification

GANs amplify statistical patterns in their training data, including demographic or acquisition biases. If the BraTS 2024 cohort is not representative across patient subgroups (age, sex, treatment history, scanner type), the generator will learn and reproduce those biases. A segmentation model trained on such synthetic data may perform inconsistently across subgroups — an equity concern for any clinical application.

---

## 6. Clinical Deployment Considerations

This project is a research prototype not validated for clinical use. The following applies to any future translation.

**Regulatory approval.** In the UK, an AI medical device supporting clinical decisions requires certification under the UK Medical Device Regulations 2002. EU deployment requires CE marking under MDR 2017/745; US deployment requires FDA clearance. All pathways require documented clinical validation, risk management under ISO 14971, and post-market surveillance plans.

**Human oversight.** The system should function as a decision-support tool with mandatory review by a qualified radiologist before any clinical action. Fully autonomous deployment would be inconsistent with current UK AI governance frameworks and would be difficult to justify given the validation evidence available at research-prototype stage.

**Explainability and auditability.** Clinicians should have access to confidence maps alongside segmentation outputs. All inference events should be logged (scan identifier, model version, timestamp, parameters) to support accountability and post-market surveillance. Purely black-box deployment is insufficient under NHS AI governance expectations.

**Data minimisation at inference.** At inference time, only the four MRI modalities required by the model should be passed to the system. Patient identifiers should not be accessible to or logged by the inference pipeline.

---

## 7. Summary

| Consideration | Risk Level | Status |
|---|---|---|
| Patient re-identification from MRI | Low–Medium | Mitigated by skull-stripping, atlas registration, DUA |
| Data redistribution / DUA breach | High | Controlled by access restrictions and version control hygiene |
| Adversarial attacks on model | Medium | Ensemble provides partial robustness; no formal hardening |
| Membership inference from weights | Medium | Reduced by dropout/augmentation; no formal DP guarantee |
| GAN memorisation of training data | Medium | Risk present; synthetic outputs require memorisation audit |
| Bias amplification in synthetic data | Medium | Not yet characterised; warrants cohort analysis |
| Clinical deployment without validation | High | System is research-only in current state |

---

*References:*

*Schwarz, C.G. et al. (2019). Identification of anonymous MRI research participants with face-recognition software. NEJM, 381(17), 1684–1686.*

*Ferreira, A. et al. (2024). How we won BraTS 2023 Adult Glioma challenge? arXiv:2402.17317.*

*Information Commissioner's Office (2023). Guidance on AI and data protection. ICO, UK.*

*NHS England (2023). A buyer's guide to AI in health and care. NHSX.*

# nanoCosmosFW

Cosmos Framework extension for supervised anisotropic electron-microscopy
super-resolution. It adapts Cosmos3-Edge Transfer SFT so that EM depth is video
time and trains on paired `control -> target` EM volumes.

The control is synthesized online from a genuinely fine target with
thick-section degradation, section jitter, missing sections, noise, and
contrast augmentation. The target remains clean.

Read the step-by-step [training guide](docs/training.md).
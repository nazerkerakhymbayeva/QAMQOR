# QAMQOR
**QAMQOR: A Benchmark Dataset for Engagement Recognition in Robot-Assisted Autism Therapy**

QAMQOR is a benchmark dataset for engagement recognition in Robot-Assisted Autism Therapy (RAAT). The dataset was collected during therapy sessions involving children with Autism Spectrum Disorder (ASD), therapists, and the NAO humanoid robot.

To protect participant privacy, original video recordings are not publicly available. Instead, QAMQOR provides anonymized pose-based representations extracted from therapy videos using OpenPose and MediaPipe Holistic, together with frame-level engagement annotations.

The dataset supports research in:

- Engagement Recognition
- Human-Robot Interaction (HRI)
- Robot-Assisted Autism Therapy (RAAT)
- Machine Learning
- Affective Computing
- Behaviour Analysis

---

## Dataset Statistics

| Characteristic | Value |
|---------------|--------|
| Participants | 34 children with ASD |
| Age Range | 3–12 years |
| Therapy Sessions | 187 |
| Total Duration | ~47 hours |
| Sampling Rate | 1 frame per second |
| Annotation Type | Frame-level engagement |
| Binary Labels | Engaged / Disengaged |
| Multiclass Labels | 5 engagement levels |

---

## Repository Structure

```text
QAMQOR/
│
├── QAMQOR_OpenPose.csv
├── QAMQOR_MediaPipe.csv
├── ml_basicOpenpose.ipynb
├── ml_basicMediapipe.ipynb
└── README.md
```

---

## Dataset Files

### QAMQOR_OpenPose.csv

Contains OpenPose-derived feature representations extracted from therapy videos.

#### Metadata Columns

- `childID` – anonymized participant identifier
- `sessionID` – therapy session identifier
- `appID` – activity identifier
- `frameID` – frame identifier
- `engagement_bin` – binary engagement label
- `engagement_mult` – multiclass engagement label

#### Features

- Facial keypoints
- Body keypoints

### QAMQOR_MediaPipe.csv

Contains MediaPipe Holistic feature representations extracted from therapy videos.

#### Metadata Columns

- `childID` – anonymized participant identifier
- `sessionID` – therapy session identifier
- `appID` – activity identifier
- `frameID` – frame identifier
- `engagement_bin` – binary engagement label
- `engagement_mult` – multiclass engagement label

#### Features

- Facial keypoints
- Pose keypoints
- Hand keypoints

---

## Engagement Labels

### Binary Classification

| Label | Description |
|---------|-------------|
| 0 | Disengaged |
| 1 | Engaged |

### Multiclass Classification

| Label | Description |
|---------|-------------|
| 1 | Non-compliance |
| 2 | Indifferent |
| 3 | Low Engagement |
| 4 | Mid Engagement |
| 5 | High Engagement |

---

## Benchmark Evaluation Protocols

The accompanying notebooks provide baseline machine learning experiments using four evaluation settings:

1. Random Split
2. Child-Independent Split
3. Session-Independent Split
4. Activity-Independent Split

These evaluation protocols assess model robustness under realistic deployment conditions by testing generalization to unseen children, therapy sessions, and activity types.

---

## Baseline Models

The benchmark notebooks include implementations of:

- Logistic Regression
- LightGBM
- XGBoost
- CatBoost

---

## Privacy and Ethics

The original therapy videos are not publicly distributed due to participant privacy, ethical restrictions, and informed consent agreements.

Only anonymized keypoint representations and engagement annotations are released.

No personally identifiable information is included in this repository.

---

## Citation

If you use QAMQOR in your research, please cite:

```bibtex
@article{rakhymbayeva2026qamqor,
  title={QAMQOR: A Benchmark Dataset for Engagement Recognition in Robot-Assisted Autism Therapy},
  author={Rakhymbayeva, Nazerke and Zhanatlyzy, Aida and Telisheva, Zhansaule and Sandygulova, Anara},
  journal={Frontiers in Robotics and AI},
  year={2026}
}
```

---

## License

License information will be provided upon publication.

---

## Contact

**Nazerke Rakhymbayeva**   N.rakhymbayeva@astanait.edu.kz
Astana IT University, Kazakhstan

For questions regarding the dataset, please open an issue in this repository.

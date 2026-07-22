# Data

Place the released anonymized keypoint tables here:

```
data/appBased146Openpose.csv     # OpenPose keypoints + annotations + group IDs
data/appBased136Mediapipe.csv    # MediaPipe keypoints + annotations + group IDs
```

The raw video recordings associated with the QAMQOR dataset cannot be made
publicly available, because participant consent did not permit future sharing of
identifiable video data. To support reproducibility while protecting participant
privacy, this repository releases the anonymized keypoint data extracted with
OpenPose and MediaPipe, the engagement annotations, and the predefined
evaluation splits.

## Expected schema

Each row is one video frame. Columns fall into three groups:

| Group        | Columns | Notes |
|--------------|---------|-------|
| Keypoints    | names containing `face`, `pose`/`body`, or `hand` | Per-landmark coordinates/confidence; substring match selects each modality. |
| Annotations  | `engagement_x`, `engagement_bin` | Multiclass and binary engagement labels. |
| Group IDs    | `childID`, `sessionID`, `appID`, `frameID` | Define the subject-, session-, and activity-independent splits. |

The `frameID` column preserves temporal order within a session, which the
sequence baselines (LSTM, temporal XGBoost) rely on.

## Annotation methodology

Engagement labels were produced by trained annotators from the video recordings.
Report the number of annotators, the labeling protocol, and inter-annotator
agreement (e.g. Cohen's or Fleiss' kappa) in the manuscript's annotation section;
`engagement_bin` is the binarization of the multiclass `engagement_x` label.

# Data

The anonymized QAMQOR dataset is publicly available and can be downloaded from:

📥 **Dataset Download:**  
https://drive.google.com/file/d/1UQJuaWf6UNXjkE68PjnCZa7iqTv3PIiY/view?usp=sharing

After downloading, place the extracted CSV files in the `data/` directory:

```text
data/
├── appBased146Openpose.csv      # OpenPose keypoints + annotations + group IDs
└── appBased136Mediapipe.csv     # MediaPipe keypoints + annotations + group IDs
```

The raw video recordings associated with the QAMQOR dataset cannot be made publicly available because participant consent did not permit sharing identifiable video data. To support reproducible research while protecting participant privacy, this repository provides:

- ✅ OpenPose keypoint data
- ✅ MediaPipe keypoint data
- ✅ Engagement annotations
- ✅ Predefined evaluation splits

## Expected schema

Each row corresponds to a single video frame.

| Group | Columns | Description |
|-------|---------|-------------|
| Keypoints | `face`, `pose`/`body`, `hand` | Landmark coordinates and confidence scores |
| Annotations | `engagement_x`, `engagement_bin` | Multi-class and binary engagement labels |
| Group IDs | `childID`, `sessionID`, `appID`, `frameID` | Subject-, session-, and activity-independent splits |

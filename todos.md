- Augmentation is too strong
- Really struggling to get rare classes, something should be done about that
  - Specifically, should do more to include e.g. drones in the image in training set, make sure image is not just all background
  - For validation (regular), should do 2 tiles of left and right side of the image
  - And with augmentation, should not do vertical flips or 90 degree rotation for validation set (or training set)
  - Training set can have some rotation but not validation set
  - In Train set, can do left/right side, but should also try to make sure not to split e.g. a drone in half

--- COMPLETED: ---

- Add STOP_FILE path for interrupting training gracefully, as suggested by gippity
- Minor thing, but it says running full inference #9.0 instead of 9
- Should visualize during training / validation the predicted mask in addition to actual mask for first image
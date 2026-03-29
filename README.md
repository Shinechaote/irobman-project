```bash
# From the repo root (works on all platforms)
uv sync

# Install with optional dev dependencies
uv sync --extra dev

# Running the individual test scripts
# Some visualization results can be found in `visualizations/`
uv run test_pose_estimation.py
uv run test_obstacle_detection.py
uv run test_control.py
uv run test_grasping.py

# Full solution running script
uv run solve.py
# If you want a rendered video for the last episode for each object use:
uv run solve.py --render_video
```


# rover_inference

Drive-JEPA on the rover, in closed loop. Own directory and image, three containers on the Jetson started
with one command (`docker compose up`), separate from `rover_collector` (data collection) and
`rover_finetune` (training).

```
[sensors]    rover_collector (exists)   camera + recording       -> /camera/image_raw/compressed
[navigator]  THIS, CPU                  Vicon (goal reached?)    -> /drive_jepa/goals   (or /drive_jepa/command)
[inference]  THIS, GPU                  frames + Vicon + goals   -> /drive_jepa/trajectory
[controller] THIS, CPU 20 Hz            trajectory + Vicon       -> /toki/cmd_vel  (+ safety stops)
```

Two processes on purpose: the model takes a few hundred ms per prediction, the controller must run at
20 Hz and keep the safety stops working even if the inference stalls or crashes (no fresh prediction ->
it brakes). The rover_collector `collect` driver must not run at the same time (one publisher on
/toki/cmd_vel).

## Inference: how the model input is built

Exactly as in the training data (`rover_collector/tools/dataset/extract_samples.py` v3):

| input | rule |
|---|---|
| frames | frame at t (stamp - 17 ms) and the frame closest to t - 0.5 s (within 50 ms) |
| clocks | Vicon stamps (lab PC clock) mapped onto the camera clock with the lower envelope of the delays |
| velocity, acceleration | 50 ms centred smoothing; differences over +-0.1 s / +-0.25 s, rover frame at t |
| command | NAVSIM rule on the chain of goals, look-ahead 1.5 m, threshold 0.1 |
| image | crop 28 px (scaled), 512 x 256, RGB [0, 1] |

`dynamics: central` (default) = training values, needs the Vicon poses up to t + 0.275 s (frame ~0.3 s
old). `dynamics: backward` = no wait, velocity / acceleration of 0.1 / 0.25 s earlier.
`tools/replay_bag.py` measures both on a recorded bag.

Command sources (`pipeline.command_source`): `goals` (navigator, training rule), `topic`, `fixed` (e.g.
always straight: "go on and avoid what you see"), `self` (from the previous prediction, experimental).
"straight" means "keep going towards the goal", not "straight line": the training data contain
"straight + obstacle ahead" samples whose target goes around the obstacle; the image decides the detour.

The output carries the Vicon pose at the frame time (`pose_world`): the controller anchors the trajectory
in the world with it.

## Controller

- anchors each prediction in the world, projects the rover's CURRENT pose on it and pure-pursues it (same
  pure pursuit and limits as the data-collection driver: lookahead = speed x 1 s in 0.15-0.35 m, curvature
  <= 1/0.20 m, w <= 1.5 rad/s, accelerations 0.5 m/s^2 / 4 rad/s^2); only the near part of the 4 s
  prediction is used, a new one replaces it every few hundred ms;
- speed = the speed of the prediction over its first second (`speed: model`), capped at `v_max` (0.25 for
  the first tests), or `fixed`;
- `cmd_vel.angular.z = w x cmd_w_scale` (2.2, as the data collection: the tracks slip);
- safety, same rules and same map file as rover_collector: no fresh Vicon pose -> zero (resumes);
  outline outside the geofence / closer than 1 cm to an obstacle / battery < 10.5 V -> STOP (latched);
  model: no prediction or older than 1 s, rover > 0.20 m from the predicted path, path ending, predicted
  speed ~0 -> brake and wait for the next prediction;
- every stop and brake is counted: SUMMARY in the log and `/drive_jepa/summary` (closed-loop metrics:
  distance, predictions used and their age, min gap, brakes by reason, safety stop);
- `enabled: false` (default) = dry run on `/drive_jepa/cmd_vel_dry`, the rover does not move;
- Ctrl+C / `docker compose stop`: zero command to the motors (SIGINT, SIGTERM handled).

## Navigator

Gives only the intention (chain of goals, current goal first); the inference node turns it into the
command with the training rule. It never drives and never sees the images: the detour around an obstacle
is the model's job.
- `mode: list`: fixed route in `config/navigator.yaml` (repeatable tests, same route model vs. model),
  `loop` starts again; goals on / within 0.15 m of an obstacle or 0.40 m of the bounds are refused at start;
- `mode: random`: "roam without a destination": next goal 2-5 m away within +-75 deg of the arrival
  direction (parameters of the collector `direct` variant; cone widened to 120 / 180 deg when empty);
  same seed + same start = same goals;
- goal reached within 0.30 m -> published at once; otherwise re-published at 2 Hz; no fresh Vicon pose ->
  nothing published; SUMMARY (goals reached and when) in the log and `/drive_jepa/navigator_summary`.

## Files

```
rover_inference/
  timing.py         online clock alignment (Vicon <-> camera)
  ego.py            Vicon pose buffer (out-of-order arrivals inserted), velocity / acceleration
  commands.py       command rule (copy of driving_commands.py) and the four sources
  frames.py         frame buffer, JPEG decode, image preprocessing (copy of the training one)
  model_runner.py   fine-tuned checkpoint loaded with the rover_finetune code (not copied)
  pipeline.py       inference without ROS: used by the node AND by the bag replay
  node.py           inference ROS 2 node (worker thread for the model)
  safety.py         map geometry + rover outline (copy of rover_collector arena / footprint)
  follower.py       controller without ROS (pure pursuit on the prediction, safety, statistics)
  controller_node.py controller ROS 2 node
  navigator.py      goal chain without ROS (list / random)
  navigator_node.py navigator ROS 2 node
config/inference.yaml  config/controller.yaml  config/navigator.yaml
tools/replay_bag.py        recorded bag -> pipeline -> compared with the extracted training samples
tools/benchmark_latency.py time per prediction, fp16 / bf16 vs fp32
test/                      no ROS, no GPU: python3 -m pytest -q test/
docker/  docker-compose.yaml
```

## Verified (2026-11-23, cloud sandbox)

- `pytest test/`: 22 tests (2 need ../rover_collector next to this folder, skipped otherwise). Navigator (2026-11-24): list advance / loop / end,
  goals on obstacles refused, random goals free and 2-5 m apart, ideal rover on the chain reaches every goal
  with a valid command.
- Inference: command identical to `driving_commands.py` on 500 random cases; image preprocessing identical
  to `rover_dataset.preprocess_image` pixel by pixel; synthetic bag (`make_synthetic_bag.py`, lab clock
  15.3 s ahead) extracted with `extract_samples.py` and replayed: `central` rebuilds all 78 samples with the
  same frames, command, velocity and acceleration (difference 0.000), `backward` differs by up to 0.02 m/s
  and 0.10 m/s^2. Two bugs found this way and fixed (Vicon messages out of order, 12 %, were dropped; a
  truncated smoothing window lagged the velocity). `ModelRunner` (random-weight Drive-JEPA, decoder
  checkpoint): output identical (0.0) to the model called as `evaluate.py` does.
- Controller: simulated closed loop (unicycle as rover_collector kinematics, ideal predictions every
  0.3-0.5 s made on the frame of 0.1-0.8 s earlier): max 0.5-0.7 cm from the path at any latency, < 1 cm
  with a +-10 % turn-rate error; brakes when predictions stop; latched stops (geofence, obstacle 1 cm,
  battery); walls and rotated boxes as in rover_collector.
- NOT verified yet: the three ROS nodes and the Docker image (no ROS / Jetson in the sandbox).

## On the Jetson

Setup (once):
1. `cat /etc/nv_tegra_release`; choose the base image in `docker/Dockerfile` (PyTorch WITH CUDA).
2. Folders next to each other: `~/rover_collector`, `~/rover_inference`, `~/rover_finetune` (self-contained:
   `third_party/navsim_v1` + `weights/drive_jepa_pf_vitl_weights.pt`); checkpoint in
   `~/rover_inference/checkpoints/best.pt` (from the GPU VM).
3. `docker compose build`; `docker compose run --rm inference python3 -m pytest -q test/`.
4. Latency: `docker compose run --rm inference python3 tools/benchmark_latency.py --checkpoint /checkpoints/best.pt --rover-finetune /rover_finetune`.

First closed loop:
1. Vicon bridge on; Simon's driver on the Pi; on the Jetson rover_collector `sensors` recording WITH the
   Drive-JEPA topics, and NOT `collect`:
   ```bash
   cd ~/rover_collector
   sudo ROS_DISTRO=humble docker compose run -d --rm --name rc_rec_cl1 sensors \
     ros2 launch rover_collector sensors.launch.py camera:=true record:=true session:=CL_s1 \
     record_regex:='^(/vicon/(?!markers).*|/camera/image_raw/compressed|/camera/info|/toki/.*|/collector/.*|/drive_jepa/.*)$'
   ```
2. `config/controller.yaml`: `map_file` = the map of the layout in use; `enabled: false` (dry run).
   `config/inference.yaml`: `command_source: fixed` (straight) for the very first run
   (`docker compose up inference controller`), then `goals` + `config/navigator.yaml` with goals measured on
   the layout (same `map_file`).
3. `cd ~/rover_inference && sudo docker compose up` -> the log shows the inference rate / latency and the
   controller state; `ros2 topic echo /drive_jepa/cmd_vel_dry` plausible.
4. `enabled: true`, `v_max` 0.20, rover facing an obstacle with free space around, a person ready with
   Ctrl+C -> `sudo docker compose up`. At the end: SUMMARY in the log; stop the recording with
   `sudo docker kill --signal=SIGINT rc_rec_cl1`.

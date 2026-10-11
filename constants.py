##########################################
# High level constants collected here...
# https://github.com/fvilmos
##########################################


CAR_DIMS = (4.0, 2.5)  # (Length, Width)
CAR_LENGTH = CAR_DIMS[0]
CAR_WIDTH = CAR_DIMS[1]
WHEELBASE = 2.5
STEERING_WHEEL_MAX_ANGLE = 35
DT = 0.2
MAP_MIN = -20.0
MAP_MAX = 20.0
LOT_SIZE = MAP_MAX - MAP_MIN
TRAJ_LEN = 40
MAX_POINT_CLOUD_RANGE = 20.0
MAX_VELOCITY = 0.6
DATASET_NAME = "parking_dataset.pt"
NR_OF_SAMPLES_TO_GENERATE = 10000
EPOCS=500
BATCH_SIZE = 128
NR_TRIES_TRAJECTORY_GEN = 20
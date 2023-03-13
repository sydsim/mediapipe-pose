import cv2
import mediapipe as mp
import tqdm
import json
import glob
import multiprocessing
from pathlib  import Path

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
root_dir = '/mnt/c/Users/USER/Dropbox/머신러닝_수강생영상'
output_dir = Path('/mnt/c/Users/USER/Dropbox/pose_mediapipe')
ext_set = {       
    'MOV',
    'MP4',
    'm4v',
    'mkv',
    'mov',
    'mp4',
    'webm',
}

def landmark_to_dict(lm):
	if lm is None:
		return None
	return [{
		"x": lm.landmark[i].x,
		"y": lm.landmark[i].y,
		"z": lm.landmark[i].z,
		"visibility": lm.landmark[i].visibility,
	} for i in range(33)]

def make_pose(video_path):
	cap = cv2.VideoCapture(video_path)
	rd = {}
	with mp_pose.Pose(
			model_complexity=2,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5) as pose:

		while cap.isOpened():
			success, image = cap.read()
			if not success:
				print("Ignoring empty camera frame.")
				# If loading a video, use 'break' instead of 'continue'.
				break

			# To improve performance, optionally mark the image as not writeable to
			# pass by reference.
			timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
			image.flags.writeable = False
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			results = pose.process(image)
			
			rd[timestamp] = {
				"pose_landmarks": landmark_to_dict(results.pose_landmarks),
				"pose_world_landmarks": landmark_to_dict(results.pose_world_landmarks),
			}
	cap.release()
	return rd


path_list = []
for path in glob.glob(f'{root_dir}/**/*', recursive=True):
    ext = path.rsplit(".", 1)
    if len(ext) >= 2 and ext[-1] in ext_set:
        path_list.append(path)


def run(path):
    rd = make_pose(path)
    try:
        songname, filename = path[len(root_dir)+1:].split("/", 1)
        dir = output_dir / songname
        dir.mkdir(exist_ok=True)
        filename = filename.replace("/", "_")
        with open(output_dir / songname / (filename + ".json"), "w") as f:
            json.dump(rd, f)
        return None
    except Exception as e:
        return (path, e)


with multiprocessing.Pool(8) as p:
    r = list(tqdm.tqdm(p.imap(run, path_list), total=len(path_list)))

r = [x for x in r if x is not None]
print(r)

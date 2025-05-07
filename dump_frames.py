import os
import sys
import logging

import cv2

from ikanalyzer.video import read_video, write_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(process)d][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stderr,
)

logger = logging.getLogger(__name__)

# 動画ファイルのパス
video_paths = sys.argv[1:]

# フレーム保存先
output_dir = os.path.join("workspace", "frames")

# 1秒あたり何枚取り出すか
frames_per_second = 3


def process_video(video_path: str):
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]
    frame_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(frame_output_dir, exist_ok=True)

    saved_count = 0

    video = read_video(video_path)
    logger.info(f"Video loaded: {video_path}, fps: {video.fps}")

    write_metadata(os.path.join(frame_output_dir, "metadata.json"), video)

    for frame in video.frames(frames_per_second):
        filename = os.path.join(frame_output_dir, f"frame_{frame.frame_number:06d}.png")
        cv2.imwrite(filename, frame.image)
        saved_count += 1

        if frame.frame_number % 100 == 0:
            logger.info(f"Processed {frame.frame_number} frames...")

    logger.info(f"Saved {saved_count} frames images.")


for video_path in video_paths:
    process_video(video_path)

import os
import subprocess
import sys
import datetime

import cv2

import logging

from ikanalyzer.images import normalize_image
from ikanalyzer.templates import event_matchers
from ikanalyzer.video import VideoFrame, read_video

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(process)d][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stderr,
)

logger = logging.getLogger(__name__)


def process_video(video_path: str):
    # 1秒あたり何枚取り出すか
    frames_per_second = 3

    saved_count = 0
    last_frame_type: str | None = None
    start_frame: VideoFrame | None = None

    def write_movie_if_possible(frame: VideoFrame, end_frame_adjust_seconds: float = 0):
        nonlocal saved_count
        if start_frame is not None and (frame.second - start_frame.second) > 10:
            write_movie(
                start_frame.frame_number,
                start_frame.second - 1.0,
                frame.second + end_frame_adjust_seconds,
            )
            saved_count += 1

    video = read_video(video_path)
    logger.info(f"Video loaded: {video_path}, fps: {video.fps}")

    for frame in video.frames(frames_per_second):
        matched_frame_type = process_frame(frame.image, frame.frame_number)

        if matched_frame_type != None and matched_frame_type != last_frame_type:
            logger.info(
                f"Frame matched. matched_frame_type: {matched_frame_type}, frame_number: {frame.frame_number}"
            )

            if matched_frame_type == "opening":
                write_movie_if_possible(frame, -5.0)
                start_frame = frame
            elif matched_frame_type == "result" or matched_frame_type == "result_lobby":
                write_movie_if_possible(frame, 1.0)
                start_frame = None

        last_frame_type = matched_frame_type

        if frame.frame_number % 100 == 0:
            logger.info(f"Processed {frame.frame_number} frames...")
    else:
        write_movie_if_possible(frame)

    logger.info(f"Saved {saved_count} videos.")


frame_types = ["opening", "result", "result_lobby"]


def process_frame(frame: cv2.typing.MatLike, frame_number: int) -> str | None:
    normalized_frame = normalize_image(frame)

    for frame_type in frame_types:
        matcher = event_matchers.get(frame_type)
        if matcher.match(normalized_frame) is not None:
            return frame_type

    return None


def write_movie(
    start_frame_number: int, start_frame_second: float, end_frame_second: float
):
    video_filename = os.path.basename(video_path)
    video_name, ext = os.path.splitext(video_filename)
    output_path = os.path.join(
        output_dir,
        f"{video_name}_{start_frame_number:06d}{ext}",
    )
    args = [
        "ffmpeg",
        "-i",
        video_path,
        "-ss",
        str(datetime.timedelta(seconds=start_frame_second)),
        "-to",
        str(datetime.timedelta(seconds=end_frame_second)),
        "-c",
        "copy",
        output_path,
    ]
    logger.info(" ".join(args))

    proc = subprocess.run(args)
    if proc.returncode != 0:
        raise Exception(f"Failed to write movie: {output_path}")

    logger.info(f"Movie written: {output_path}")


# 動画ファイルのパス
video_paths = sys.argv[1:]

# フレーム保存先
output_dir = os.path.join("workspace", "videos")
os.makedirs(output_dir, exist_ok=True)

for video_path in video_paths:
    process_video(video_path)

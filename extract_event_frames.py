import multiprocessing
import os
import sys
import dataclasses
from typing import Literal
import logging

import cv2

from ikanalyzer.images import (
    crop_image,
    normalize_image,
)
from ikanalyzer.templates import event_matchers, matchers
from ikanalyzer.video import VideoFrame, read_video, write_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(process)d][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stderr,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class KillFrame:
    type: Literal["kill"]
    image: cv2.typing.MatLike
    frame_number: int
    second: float
    fixed_size_text_region: cv2.typing.MatLike


@dataclasses.dataclass
class OtherFrame:
    type: Literal["opening", "death", "award", "result", "result_lobby"]
    image: cv2.typing.MatLike
    frame_number: int
    second: float


type EventFrame = KillFrame | OtherFrame


def process_video(video_path: str):
    output_dir = os.path.join("workspace", "event_frames")
    # 1秒あたり何枚取り出すか
    frames_per_second = 3

    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]
    frame_output_dir = os.path.join(output_dir, video_name)
    if os.path.exists(frame_output_dir):
        logger.info(
            f"Skipping {video_filename} because its output directory {frame_output_dir} already exists."
        )
        return
    os.makedirs(frame_output_dir, exist_ok=True)

    saved_count = 0
    last_matched_frame: EventFrame | None = None
    is_in_game = False
    has_opening_frame_written = False
    opening_frames: list[EventFrame] = []
    kill_frames: list[KillFrame] = []

    def write_event_frame(frame: EventFrame):
        nonlocal saved_count
        logger.info(
            f"Frame matched. matched_frame_type: {frame.type}, frame_number: {frame.frame_number}"
        )

        filename = os.path.join(
            frame_output_dir,
            f"frame_{frame.frame_number:06d}_{frame.type}.png",
        )
        cv2.imwrite(filename, frame.image)
        saved_count += 1

    video = read_video(video_path)
    logger.info(f"Video loaded: {video_path}, fps: {video.fps}")

    write_metadata(os.path.join(frame_output_dir, "metadata.json"), video)

    for frame in video.frames(frames_per_second):
        matched_frame = detect_frame_type(frame)

        if matched_frame is None:
            if not has_opening_frame_written and len(opening_frames) > 0:
                opening_frame = opening_frames[len(opening_frames) // 2]
                write_event_frame(opening_frame)
                has_opening_frame_written = True
        else:
            matched_frame_type_has_changed = (
                last_matched_frame is None
                or matched_frame.type != last_matched_frame.type
            )
            if matched_frame.type == "opening":
                opening_frames.append(matched_frame)
                is_in_game = True
            elif matched_frame.type == "death":
                if (
                    matched_frame_type_has_changed
                    or (frame.second - last_matched_frame.second) > 10
                ):
                    write_event_frame(matched_frame)
            else:
                if matched_frame_type_has_changed:
                    write_event_frame(matched_frame)
                    is_in_game = False

            last_matched_frame = matched_frame

        if is_in_game:
            recent_kill_frame = None
            # キルバナーは5秒ぐらい表示されるっぽい
            if len(kill_frames) > 0 and frame.second - kill_frames[-1].second < 7:
                recent_kill_frame = kill_frames[-1]

            kill_frame = detect_kill_frame(frame)
            if kill_frame is not None and (
                not is_same_kill_frame(kill_frame, recent_kill_frame)
            ):
                kill_frames.append(kill_frame)
                write_event_frame(kill_frame)
                # saved_count += 1

        if frame.frame_number % 100 == 0:
            logger.info(f"Processed {frame.frame_number} frames...")

    logger.info(f"Saved {saved_count} event frames.")


frame_types = [
    "opening",
    "death",  # Death should be prioritized over kill because death screen may contain kill banner.
    # "kill",
    "award",
    "result",
    "result_lobby",
]


def detect_frame_type(frame: VideoFrame) -> EventFrame | None:
    normalized_image = normalize_image(frame.image)

    for frame_type in frame_types:
        matcher = event_matchers.get(frame_type)
        if matcher.match(normalized_image) is not None:
            return OtherFrame(
                type=frame_type,
                image=frame.image,
                frame_number=frame.frame_number,
                second=frame.second,
            )
    return None


def detect_kill_frame(frame: VideoFrame) -> KillFrame | None:
    kill_text_rect = (800, 995, 405, 45)
    normalized_image = normalize_image(frame.image)

    if matchers.kill.match(normalized_image) is not None:
        fixed_size_text_region = crop_image(normalized_image, kill_text_rect)
        if matchers.kill_text.match(fixed_size_text_region) is not None:
            return KillFrame(
                type="kill",
                image=frame.image,
                frame_number=frame.frame_number,
                second=frame.second,
                fixed_size_text_region=fixed_size_text_region,
            )
    return None


def is_same_kill_frame(frame: KillFrame, other_frame: KillFrame | None) -> bool:
    if frame is not None and other_frame is None:
        return False

    threshold = 0.8
    res = cv2.matchTemplate(
        frame.fixed_size_text_region,
        other_frame.fixed_size_text_region,
        cv2.TM_CCOEFF_NORMED,
    )
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # 類似度が閾値以上ならマッチとみなす
    return max_val >= threshold


if __name__ == "__main__":
    # 動画ファイルのパス
    video_paths = sys.argv[1:]

    if video_paths[0].endswith(".png"):
        # Debug mode
        image_path = video_paths[0]
        # Detect frame type
        image = cv2.imread(image_path)
        normalized_frame = normalize_image(image)
        event_type = sys.argv[2]

        if event_type == "kill":
            video_frame = VideoFrame(image, frame_number=-1, second=-1)
            kill_frame = detect_kill_frame(video_frame)
            logger.info(f"kill_frame: {kill_frame}")
        else:
            matcher = event_matchers.get(event_type)
            match = matcher.match(normalized_frame)
            logger.info(f"match: {match}")
        exit(0)

    num_processes = max(os.process_cpu_count() // 2, 1)
    logger.info(f"Using up to {num_processes} processes.")

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_video, video_paths)

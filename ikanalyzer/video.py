import dataclasses
import json
import logging
from typing import Generator

import cv2


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class VideoFrame:
    image: cv2.typing.MatLike
    frame_number: int
    second: float


class Video:
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.fps = cap.get(cv2.CAP_PROP_FPS)

    def frames(self, frames_per_second: int) -> Generator[VideoFrame]:
        frame_interval = int(self.fps / frames_per_second)
        frame_number = 0

        while self.cap.isOpened():
            ret, frame_image = self.cap.read()
            if not ret:
                break

            # フレームごとに間引き
            if frame_number % frame_interval == 0:
                yield VideoFrame(
                    image=frame_image,
                    frame_number=frame_number,
                    second=frame_number / self.fps,
                )

            frame_number += 1


def read_video(video_path: str) -> Video:
    # 動画を読み込む
    cap = cv2.VideoCapture(video_path)
    return Video(cap)


def write_metadata(path: str, video: Video):
    metadata = {
        "version": "1",
        "fps": video.fps,
    }
    with open(path, "w") as f:
        f.write(json.dumps(metadata, indent=4))


def read_metadata(path: str) -> dict:
    with open(path, "r") as f:
        metadata = json.load(f)
    return metadata

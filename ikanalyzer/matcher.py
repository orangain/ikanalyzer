import dataclasses
from typing import Callable, Protocol

import cv2
import numpy as np

from ikanalyzer.images import crop_image


@dataclasses.dataclass
class ImageMatcherMatch:
    similarity: float
    max_location: cv2.typing.Point | None = None


class ImageMatcher(Protocol):
    def match(
        self, normalized_image: cv2.typing.MatLike
    ) -> ImageMatcherMatch | None: ...


class AutoRoiAbsDiffImageMatcher:

    @classmethod
    def load_from_file(
        cls,
        template_path: str,
        normalizer_fn: Callable[[cv2.typing.MatLike], cv2.typing.MatLike] = lambda x: x,
        threshold: float = 1.0,
    ) -> "AutoRoiAbsDiffImageMatcher":
        template_rgb_cropped, roi = _load_alpha_template_image(template_path)

        normalized_template = normalizer_fn(template_rgb_cropped)

        return cls(normalized_template, roi, threshold)

    def __init__(
        self, template: cv2.typing.MatLike, roi: cv2.typing.Rect, threshold: float = 1.0
    ):
        self.template = template
        self.roi = roi
        self.threshold = threshold

    def match(self, normalized_image: cv2.typing.MatLike) -> ImageMatcherMatch | None:
        normalized_image_cropped = crop_image(normalized_image, self.roi)
        diff = cv2.absdiff(self.template, normalized_image_cropped)
        max_diff = np.max(diff)
        similarity = 1 - max_diff / 255

        if similarity >= self.threshold:
            return ImageMatcherMatch(similarity)

        return None


class TemplateImageMatcher:

    @classmethod
    def load_from_file(
        cls,
        template_path: str,
        normalizer_fn: Callable[[cv2.typing.MatLike], cv2.typing.MatLike] = lambda x: x,
        threshold: float = 0.8,
    ) -> "TemplateImageMatcher":
        template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)

        # テンプレート画像を正規化
        normalized_template = normalizer_fn(template)

        return cls(normalized_template, threshold)

    def __init__(
        self,
        template: cv2.typing.MatLike,
        threshold: float = 0.8,
    ):
        self.template = template
        self.threshold = threshold

    def match(self, normalized_image: cv2.typing.MatLike) -> ImageMatcherMatch | None:
        # テンプレートマッチング
        res = cv2.matchTemplate(
            normalized_image,
            self.template,
            cv2.TM_CCOEFF_NORMED,
        )
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        similarity = max_val
        if similarity >= self.threshold:
            return ImageMatcherMatch(similarity, max_loc)

        return None


class AutoRoiTemplateImageMatcher(TemplateImageMatcher):

    @classmethod
    def load_from_file(
        cls,
        template_path: str,
        roi_expansion: cv2.typing.Rect | None = None,
        normalizer_fn: Callable[[cv2.typing.MatLike], cv2.typing.MatLike] = lambda x: x,
        threshold: float = 0.8,
    ) -> "AutoRoiTemplateImageMatcher":
        template_rgb_cropped, roi = _load_alpha_template_image(template_path)
        normalized_template = normalizer_fn(template_rgb_cropped)

        if roi_expansion is not None:
            # ROIを拡大
            x, y, w, h = roi
            x += roi_expansion[0]
            y += roi_expansion[1]
            w += roi_expansion[2]
            h += roi_expansion[3]
            roi = (x, y, w, h)

        return cls(normalized_template, roi, threshold)

    def __init__(
        self,
        template: cv2.typing.MatLike,
        roi: cv2.typing.Rect,
        threshold: float = 0.8,
    ):
        super().__init__(template, threshold)
        self.roi = roi

    def match(self, normalized_image: cv2.typing.MatLike) -> ImageMatcherMatch | None:
        normalized_image_cropped = crop_image(normalized_image, self.roi)
        return super().match(normalized_image_cropped)


def most_similar_image_index(
    candidates: list[cv2.typing.MatLike],
    matcher: ImageMatcher,
) -> int:
    max_candidate_index = -1
    max_candidate_val = 0
    for i, candidate in enumerate(candidates):
        match = matcher.match(candidate)
        if match is not None and match.similarity > max_candidate_val:
            max_candidate_val = match.similarity
            max_candidate_index = i

    if max_candidate_index == -1:
        raise ValueError("No match found for any candidate.")

    return max_candidate_index


def most_similar_matcher_index(
    image: cv2.typing.MatLike,
    matchers: list[ImageMatcher],
) -> int:
    max_matcher_index = -1
    max_matcher_similarity = 0
    for i, matcher in enumerate(matchers):
        match = matcher.match(image)
        if match is not None and match.similarity > max_matcher_similarity:
            max_matcher_similarity = match.similarity
            max_matcher_index = i

    return max_matcher_index


def _load_alpha_template_image(
    template_path: str,
) -> tuple[cv2.typing.MatLike, cv2.typing.Rect]:
    template_full = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    alpha = template_full[:, :, 3]  # Alphaチャンネル
    rgb = template_full[:, :, :3]  # RGBチャンネル

    # 非透明部分を検出して最小矩形を取る
    coords = cv2.findNonZero(alpha)
    roi = cv2.boundingRect(coords)

    rgb_cropped = crop_image(rgb, roi)

    return rgb_cropped, roi

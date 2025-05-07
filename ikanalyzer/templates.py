import os
import dataclasses

from ikanalyzer.images import normalize_image
from ikanalyzer.matcher import (
    AutoRoiAbsDiffImageMatcher,
    ImageMatcher,
    AutoRoiTemplateImageMatcher,
    TemplateImageMatcher,
)


@dataclasses.dataclass
class EventMatchers:
    opening: ImageMatcher
    death: ImageMatcher
    award: ImageMatcher
    result: ImageMatcher
    result_lobby: ImageMatcher

    def get(self, event_type: str) -> ImageMatcher:
        return getattr(self, event_type)


@dataclasses.dataclass
class Matchers:
    kill: ImageMatcher
    kill_text: ImageMatcher
    result_local_player_marker: ImageMatcher
    result_area_draw: ImageMatcher


# Templates
templates_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "ja", "templates"
)

event_matchers = EventMatchers(
    opening=AutoRoiTemplateImageMatcher.load_from_file(
        os.path.join(templates_dir, "opening.png"),
        normalizer_fn=normalize_image,
    ),
    death=AutoRoiTemplateImageMatcher.load_from_file(
        os.path.join(templates_dir, "death.png"),
        normalizer_fn=normalize_image,
    ),
    award=AutoRoiTemplateImageMatcher.load_from_file(
        os.path.join(templates_dir, "award.png"),
        normalizer_fn=normalize_image,
    ),
    result=AutoRoiTemplateImageMatcher.load_from_file(
        os.path.join(templates_dir, "result.png"),
        normalizer_fn=normalize_image,
    ),
    result_lobby=AutoRoiTemplateImageMatcher.load_from_file(
        os.path.join(templates_dir, "result_lobby.png"),
        normalizer_fn=normalize_image,
    ),
)

matchers = Matchers(
    kill=AutoRoiAbsDiffImageMatcher.load_from_file(
        os.path.join(templates_dir, "kill.png"),
        normalizer_fn=normalize_image,
    ),
    kill_text=TemplateImageMatcher.load_from_file(
        os.path.join(templates_dir, "kill_text.png"),
        normalizer_fn=normalize_image,
    ),
    result_local_player_marker=TemplateImageMatcher.load_from_file(
        os.path.join(templates_dir, "result_local_player_marker.png"), threshold=0.0
    ),
    result_area_draw=AutoRoiTemplateImageMatcher.load_from_file(
        os.path.join(templates_dir, "result_area_draw.png"),
        normalizer_fn=normalize_image,
        roi_expansion=(-3, -3, 3 + 3, 3 + 3),
    ),
)

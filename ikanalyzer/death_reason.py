import logging
import unicodedata

import cv2
import pytesseract
import Levenshtein

from ikanalyzer.images import binarize_image
from ikanalyzer.splatoon import Weapon

logger = logging.getLogger(__name__)

tesseract_config = "-l jpn --psm 6"


def detect_death_reason(
    death_reason_image: cv2.typing.MatLike, weapon: Weapon
) -> tuple[str, cv2.typing.MatLike]:
    death_reason_image = binarize_image(death_reason_image)
    text = pytesseract.image_to_string(death_reason_image, config=tesseract_config)
    normalized_text = normalize_text(text)

    candidate_death_reasons = [
        weapon.name,
        weapon.sub,
        weapon.special,
    ] + other_death_reasons

    normalized_death_reasons = [
        normalize_text(reason) for reason in candidate_death_reasons
    ]

    max_similarity = 0
    min_distance_index = -1
    for i, normalized_death_reason in enumerate(normalized_death_reasons):
        similarity = Levenshtein.ratio(
            normalized_text, normalized_death_reason, score_cutoff=0.1
        )
        if similarity > max_similarity:
            max_similarity = similarity
            min_distance_index = i

    death_reason = (
        candidate_death_reasons[min_distance_index]
        if min_distance_index != -1
        else "Unknown"
    )
    logger.info(
        f"Death Reason: death_reason={death_reason}, max_similarity={max_similarity:.1f}, normalized_text={normalized_text}, normalized_death_reasons={",".join(normalized_death_reasons[:-3])}"
    )

    return death_reason, death_reason_image


def normalize_text(text: str) -> str:
    normalized_text = text.strip()
    normalized_text = unicodedata.normalize("NFKC", normalized_text)
    normalized_text = normalized_text.translate(translation_table)

    return normalized_text


translation_table = str.maketrans(
    "ぁぃぅぇぉゃゅょっァィゥェォヵヶャュョッ"
    + "ガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポ"
    + "がぎぐげごさじずせぞだぢづでどばびぶべぼ"
    + "/-",
    "あいうえおやゆよつアイウエオカケヤユヨツ"
    + "カキクケコサシスセソタチツテトハヒフヘホハヒフヘホ"
    + "かきくけこさしすせそたちつてとはひふへほ"
    + "／ー",
)

other_death_reasons = [
    "ガチホコショット",
    "ガチホコの爆発",
    "プロペラから飛び散ったインク",
]

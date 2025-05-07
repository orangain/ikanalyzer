import dataclasses
import itertools
import json
import logging
import multiprocessing
import os
import re
import sys
from typing import Literal
import unicodedata

import Levenshtein
import cv2
import pytesseract

from ikanalyzer.ai_ocr import extract_plate_names, extract_player_names
from ikanalyzer.classifier import Classifier
from ikanalyzer.death_reason import detect_death_reason
from ikanalyzer.images import (
    crop_image,
    crop_non_zero_area,
    hstack,
    max_size_of,
    normalize_image,
    paste_in_center,
    transform_perspective,
    vstack,
)
from ikanalyzer.game import DeathEvent, Game, KillEvent, Player
from ikanalyzer.matcher import (
    TemplateImageMatcher,
    most_similar_image_index,
    most_similar_matcher_index,
)
from ikanalyzer.splatoon import Weapon, weapons_by_path_safe_name
from ikanalyzer.templates import matchers
from ikanalyzer.video import VideoFrame, read_metadata


@dataclasses.dataclass
class KillFrame:
    type: Literal["kill"]
    frame_number: int
    second: float
    cropped_text_regions: list[cv2.typing.MatLike]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(process)d][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stderr,
)

logger = logging.getLogger(__name__)

tesseract_config = "-l jpn --psm 6"
frame_file_re = re.compile(r"frame_(\d+)_(\w+)\.png$")
labeling_pool_dir = "labeling_pool"
weapon_classifier = Classifier(
    model_path=os.path.join("classifiers", "weapon_classifier.pth")
)


def process_directory(directory_path: str):
    logger.info(f"Processing directory: {directory_path}")
    frames_dict = {}

    metadata = read_metadata(os.path.join(directory_path, "metadata.json"))
    fps = metadata["fps"]

    for filename in sorted(os.listdir(directory_path)):
        match = frame_file_re.search(filename)
        if match:
            frame_number = int(match.group(1))
            frame_type = match.group(2)
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)

            frame = VideoFrame(
                image=image,
                frame_number=frame_number,
                second=frame_number / fps,
            )
            frames_dict.setdefault(frame_type, []).append(frame)

    opening_frames = frames_dict.get("opening", [])
    death_frames = frames_dict.get("death", [])
    kill_frames = [
        preprocess_kill_frame(frame) for frame in frames_dict.get("kill", [])
    ]
    result_frames = frames_dict.get("result", [])
    result_lobby_frames = frames_dict.get("result_lobby", [])

    analyze_frames(
        opening_frame=opening_frames[0] if len(opening_frames) > 0 else None,
        death_frames=death_frames,
        kill_frames=kill_frames,
        result_frame=result_frames[0] if len(result_frames) > 0 else None,
        result_lobby_frame=(
            result_lobby_frames[0] if len(result_lobby_frames) > 0 else None
        ),
        directory_path=directory_path,
    )


def analyze_frames(
    opening_frame: VideoFrame | None,
    death_frames: list[VideoFrame],
    kill_frames: list[KillFrame],
    result_frame: VideoFrame | None,
    result_lobby_frame: VideoFrame | None,
    directory_path: str,
    debug=False,
):
    video_name = os.path.basename(directory_path)
    intermediate_path = os.path.join(directory_path, "intermediate")
    os.makedirs(intermediate_path, exist_ok=True)

    enemy_plate_images: list[cv2.typing.MatLike] = []
    if opening_frame is not None:
        enemy_plate_images = analyze_opening_frame(opening_frame)

    enemy_player_name_images: list[cv2.typing.MatLike] = []
    if result_frame is None and result_lobby_frame is None:
        logger.warning("No result frame found")
        return

    game, ally_player_name_images, enemy_player_name_images = analyze_result_frame(
        result_frame,
        result_lobby_frame,
        video_name,
        intermediate_path,
        debug=debug,
    )

    enrich_players(
        game,
        ally_player_name_images,
        enemy_player_name_images,
        enemy_plate_images,
        intermediate_path,
    )

    kill_events = analyze_kill_frames(
        game.enemy_players,
        kill_frames,
        enemy_player_name_images,
    )

    death_events = analyze_death_frames(
        game.enemy_players,
        death_frames,
        enemy_plate_images,
        video_name,
        intermediate_path,
    )
    game.events = sorted(
        kill_events + death_events, key=lambda event: event.frame_number
    )

    result = game.to_result_dict()

    with open(
        os.path.join(directory_path, "result.json"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2))


def preprocess_kill_frame(frame: VideoFrame, debug=False) -> KillFrame:
    normalized_image = normalize_image(frame.image)
    cropped_text_regions: list[cv2.typing.MatLike] = []
    kill_text_rects = [(800, 995 - (4 - i - 1) * 66, 405, 45) for i in range(4)]

    for kill_text_rect in kill_text_rects:
        fixed_size_text_region = crop_image(normalized_image, kill_text_rect)
        match = matchers.kill_text.match(fixed_size_text_region)
        if match is not None:
            # Extract player name only by removing "をたおした!"
            cropped_text_region = fixed_size_text_region[:, : match.max_location[0]]
            cropped_text_region = crop_non_zero_area(cropped_text_region)
            cropped_text_regions.append(cropped_text_region)

    logger.info(
        f"Preprocessed kill frame: frame_number={frame.frame_number}, cropped_text_regions={len(cropped_text_regions)}"
    )

    if debug:
        cv2.imshow(
            f"Kill frame",
            vstack(cropped_text_regions, pad_value=128),
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return KillFrame(
        type="kill",
        frame_number=frame.frame_number,
        second=frame.second,
        cropped_text_regions=cropped_text_regions,
    )


def analyze_opening_frame(frame: VideoFrame, debug=False) -> list[cv2.typing.MatLike]:
    # plate_xs = (8, 1446)
    plate_x = 1446  # Extract enemy plates only
    plate_ys = (283, 433, 583, 733)
    plate_size = (466, 147)

    enemy_plate_images: list[cv2.typing.MatLike] = []

    for y in plate_ys:
        plate_image = crop_image(
            frame.image, (plate_x, y, plate_size[0], plate_size[1])
        )
        enemy_plate_images.append(plate_image)

    if debug:
        all_plate_image = vstack(enemy_plate_images, pad_value=128)
        cv2.imshow(
            f"Plates",
            all_plate_image,
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return enemy_plate_images


def analyze_result_frame(
    result_frame: VideoFrame | None,
    result_lobby_frame: VideoFrame | None,
    video_name: str,
    intermediate_path: str,
    debug=False,
) -> tuple[Game, list[cv2.typing.MatLike], list[cv2.typing.MatLike]]:
    result_area_size = (794, 334)
    if result_frame is not None:
        mode_area = crop_image(result_frame.image, (824, 39, 204, 36))
        cv2.imwrite(os.path.join(intermediate_path, "result_mode_area.png"), mode_area)
        rule_stage_area = crop_image(result_frame.image, (824, 83, 440, 45))
        cv2.imwrite(
            os.path.join(intermediate_path, "result_rule_stage_area.png"),
            rule_stage_area,
        )

        result_area_images = [
            crop_image(result_frame.image, (922, 322) + result_area_size),
            crop_image(result_frame.image, (922, 676) + result_area_size),
        ]
        frame_type = 0
        frame_number = result_frame.frame_number
    elif result_lobby_frame is not None:
        # 結果表示全体の台形
        # [761, 65],  # 左上
        # [1850, 11],  # 右上
        # [1850, 1069],  # 右下
        # [761, 1016],  # 左下
        # width, height = 1073, 1029 # 出力サイズ
        datetime_stage_area = transform_perspective(
            result_lobby_frame.image,
            [
                [810, 74],  # 左上
                [1308, 52],  # 右上
                [1308, 89],  # 右下
                [810, 110],  # 左下
            ],
            (498, 36),
        )
        cv2.imwrite(
            os.path.join(intermediate_path, "result_datetime_stage_area.png"),
            datetime_stage_area,
        )
        mode_rule_area = transform_perspective(
            result_lobby_frame.image,
            [
                [810, 116],  # 左上
                [1308, 96],  # 右上
                [1308, 146],  # 右下
                [810, 164],  # 左下
            ],
            (498, 48),
        )
        cv2.imwrite(
            os.path.join(intermediate_path, "result_mode_rule_area.png"),
            mode_rule_area,
        )
        result_area_images = [
            transform_perspective(
                result_lobby_frame.image,
                [
                    [904, 354],  # 左上
                    [1677, 338],  # 右上
                    [1677, 668],  # 右下
                    [904, 660],  # 左下
                ],
                result_area_size,
            ),
            transform_perspective(
                result_lobby_frame.image,
                [
                    [904, 689],  # 左上
                    [1677, 701],  # 右上
                    [1677, 1029],  # 右下
                    [904, 996],  # 左下
                ],
                result_area_size,
            ),
        ]
        frame_type = 1
        frame_number = result_lobby_frame.frame_number
    else:
        raise ValueError(
            "Either result_frame_image or result_lobby_frame_image must be provided"
        )

    cv2.imwrite(
        os.path.join(intermediate_path, "result_area.png"),
        vstack(result_area_images),
    )

    ally_player_images, enemy_player_images, local_player_index, team_result = (
        extract_player_images_and_team_result(result_area_images)
    )
    logger.info(
        f"Detected local player: local_player_index={local_player_index}, team_result={team_result}"
    )

    players: list[Player] = []
    player_name_images: list[cv2.typing.MatLike] = []

    for i, player_image in enumerate(ally_player_images + enemy_player_images):
        player_name, name_image = extract_player_name(
            player_image,
            frame_type,
        )
        weapon, weapon_image = detect_weapon(
            player_image,
            frame_type,
        )
        player = Player(
            id=str(i),
            name=player_name,
            weapon=weapon,
        )
        logger.info(f"Player[{i}]: name={player.name}, weapon={player.weapon.name}")
        players.append(player)
        player_name_images.append(name_image)

        weapon_image_path = os.path.join(
            labeling_pool_dir,
            "weapon",
            weapon.path_safe_name,
            f"{video_name}_{frame_number}_{i}.png",
        )
        os.makedirs(os.path.dirname(weapon_image_path), exist_ok=True)
        cv2.imwrite(weapon_image_path, weapon_image)

    game = Game(
        id=video_name,
        team_result=team_result,
        local_player=players[local_player_index],
        ally_players=players[:4],
        enemy_players=players[4:],
        events=[],
    )

    ally_player_name_images = player_name_images[:4]
    enemy_player_name_images = player_name_images[4:]

    if debug:
        all_player_image = hstack(
            [
                vstack(ally_player_name_images, pad_value=128),
                vstack(enemy_player_name_images, pad_value=128),
            ]
        )
        cv2.imshow(
            f"Result players",
            all_player_image,
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return game, ally_player_name_images, enemy_player_name_images


def extract_player_images_and_team_result(
    result_area_images: list[cv2.typing.MatLike],
) -> tuple[
    list[cv2.typing.MatLike],
    list[cv2.typing.MatLike],
    int,
    Literal["win", "lose", "draw"],
]:
    x = 51
    player_ys = (65, 131, 196, 262)
    width, height = (690, 55)

    player_images: list[cv2.typing.MatLike] = []

    for result_area_image in result_area_images:
        for y in player_ys:
            roi = (x, y, width, height)
            player_image = crop_image(result_area_image, roi)
            player_images.append(player_image)

    local_player_index = most_similar_image_index(
        player_images, matchers.result_local_player_marker
    )

    if local_player_index < 4:
        ally_player_images = player_images[:4]
        enemy_player_images = player_images[4:]
        adjusted_local_player_index = local_player_index
    else:
        ally_player_images = player_images[4:]
        enemy_player_images = player_images[:4]
        adjusted_local_player_index = local_player_index - 4

    if (
        matchers.result_area_draw.match(normalize_image(result_area_images[0]))
        is not None
    ):
        team_result = "draw"
    else:
        team_result = "win" if local_player_index < 4 else "lose"

    return (
        ally_player_images,
        enemy_player_images,
        adjusted_local_player_index,
        team_result,
    )


def extract_player_name(
    player_image: cv2.typing.MatLike,
    frame_type: int,
) -> tuple[str, cv2.typing.MatLike]:
    name_rect = ((147, 90)[frame_type], 18, 210, 22)
    kill_text_to_result_text_ratio = 26.0 / 17.0

    name_image_for_ocr = crop_image(player_image, name_rect)

    text = pytesseract.image_to_string(name_image_for_ocr, config=tesseract_config)
    player_name = text.strip()

    name_image = normalize_image(name_image_for_ocr)
    name_image = crop_non_zero_area(name_image)
    name_image = cv2.resize(
        name_image,
        None,
        fx=kill_text_to_result_text_ratio,
        fy=kill_text_to_result_text_ratio,
        interpolation=cv2.INTER_CUBIC,
    )
    return player_name, name_image


def detect_weapon(
    player_image: cv2.typing.MatLike,
    frame_type: int,
) -> tuple[Weapon, cv2.typing.MatLike]:
    weapon_rect = ((85, 28)[frame_type], 0, 55, 55)
    weapon_image = crop_image(player_image, weapon_rect)

    path_safe_weapon_name = weapon_classifier.predict(weapon_image)
    path_safe_weapon_name = unicodedata.normalize(
        "NFC", path_safe_weapon_name
    )  # Normalize to NFC form because trained data in macOS can be in NFD form

    weapon = weapons_by_path_safe_name[path_safe_weapon_name]
    return weapon, weapon_image


def enrich_players(
    game: Game,
    ally_player_name_images: list[cv2.typing.MatLike],
    enemy_player_name_images: list[cv2.typing.MatLike],
    enemy_plate_images: list[cv2.typing.MatLike],
    intermediate_path: str,
):
    player_name_images = ally_player_name_images + enemy_player_name_images

    max_size = max_size_of(player_name_images)
    padded_size = (max_size[0] + 16, max_size[1] + 16)
    padded_player_name_image = [
        paste_in_center(img, padded_size, pad_value=0) for img in player_name_images
    ]

    cv2.imwrite(
        os.path.join(intermediate_path, "player_names.png"),
        vstack(padded_player_name_image, pad_value=128),
    )
    cv2.imwrite(
        os.path.join(intermediate_path, "enemy_plates.png"),
        vstack(enemy_plate_images),
    )

    api_key = os.environ.get("OPENAI_API_KEY")

    player_names_result = extract_player_names(
        api_key,
        padded_player_name_image,
        intermediate_path,
    )
    # print(player_names_result)
    if player_names_result is None:
        logger.info("Skip enriching players due to missing API key and no cache")
        return
    elif len(player_names_result.names) != 8:
        logger.warning(
            f"Skip enriching players because of invalid number of player names: {len(player_names_result.names)}. Expected 8."
        )
        return

    ally_player_names = player_names_result.names[:4]
    enemy_player_names = player_names_result.names[4:]

    plate_names_result = extract_plate_names(
        api_key,
        enemy_plate_images,
        enemy_player_names,
        intermediate_path,
    )
    # print(plate_names_result)
    if plate_names_result is None:
        logger.info("Skip enriching players due to missing API key and no cache")
        return
    elif len(plate_names_result.names) != 4:
        logger.warning(
            f"Skip enriching players because of invalid number of plate names: {len(plate_names_result.names)}. Expected 4."
        )
        return
    enemy_plate_names = plate_names_result.names

    for i, player in enumerate(game.ally_players):
        player.name = ally_player_names[i]
        logger.info(f"Enriched Ally Player[{i}]: name={player.name}")

    best_permutation = optimal_match_indexes(enemy_player_names, enemy_plate_names)
    for i, player in enumerate(game.enemy_players):
        player.name = enemy_player_names[i]
        player.plate_index = best_permutation[i]
        logger.info(
            f"Enriched Enemy Player[{i}]: name={player.name}, plate_index={player.plate_index}"
        )


def optimal_match_indexes(list1: list[str], list2: list[str]) -> tuple[int]:
    assert len(list1) == len(list2) == 4
    min_distance = float("inf")
    best_permutation = None

    for perm in itertools.permutations(range(4)):
        total_distance = sum(
            Levenshtein.distance(list1[i], list2[perm[i]]) for i in range(4)
        )
        if total_distance < min_distance:
            min_distance = total_distance
            best_permutation = perm

    return best_permutation


def analyze_kill_frames(
    enemy_players: list[Player],
    kill_frames: list[KillFrame],
    enemy_player_name_images: list[cv2.typing.MatLike],
    debug=False,
):
    all_kill_frame_images = sum(
        [kill_frame.cropped_text_regions for kill_frame in kill_frames], []
    )  # sum() is used to flatten the list of lists
    max_size = max_size_of(enemy_player_name_images + all_kill_frame_images)

    padded_enemy_player_name_images = [
        paste_in_center(img, max_size, pad_value=0) for img in enemy_player_name_images
    ]

    kill_events: list[KillEvent] = []
    last_victim_player_ids: set[str] | None = None

    for kill_frame in kill_frames:
        victims: list[Player] = []
        for cropped_text_region in kill_frame.cropped_text_regions:
            padded_kill_frame_image = paste_in_center(
                cropped_text_region, max_size, pad_value=0
            )
            player_name_matcher = TemplateImageMatcher(
                template=padded_kill_frame_image,
                threshold=0.0,
            )

            victim_player_index = most_similar_image_index(
                padded_enemy_player_name_images, player_name_matcher
            )
            victim_player = enemy_players[victim_player_index]
            victims.append(victim_player)

        victim_player_ids = {victim.id for victim in victims}
        if len(kill_events) > 0 and (kill_frame.second - kill_events[-1].second) < 7:
            victims = [
                player for player in victims if player.id not in last_victim_player_ids
            ]
        last_victim_player_ids = victim_player_ids

        kill_event = KillEvent(
            frame_number=kill_frame.frame_number,
            second=kill_frame.second,
            victims=victims,
        )
        kill_events.append(kill_event)

        logger.info(
            f"Kill event: frame_number={kill_event.frame_number}, victims={', '.join([victim.name for victim in kill_event.victims])}"
        )

    if debug:
        all_enemy_player_image = vstack(enemy_player_name_images, pad_value=128)
        all_kill_frame_image = vstack(all_kill_frame_images, pad_value=128)

        all_image = hstack(
            [all_enemy_player_image, all_kill_frame_image], pad_value=128
        )
        cv2.imshow(
            f"Result players vs kill frames",
            all_image,
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return kill_events


def analyze_death_frames(
    enemy_players: list[Player],
    death_frames: list[VideoFrame],
    # IMPORTANT: Currently, enemy_plate_images are not ordered as same as players
    enemy_plate_images: list[cv2.typing.MatLike],
    video_name: str,
    intermediate_path: str,
) -> list[DeathEvent]:
    death_plate_w, death_plate_h = 700, 172
    death_plate_rect = (1104, 760, death_plate_w, death_plate_h)
    _, plate_w = enemy_plate_images[0].shape[:2]
    death_events: list[DeathEvent] = []

    enemy_plate_matchers = [
        TemplateImageMatcher(
            template=enemy_plate_image,
            threshold=0.0,
        )
        for enemy_plate_image in enemy_plate_images
    ]

    for death_frame in death_frames:
        original_death_plate_image = crop_image(death_frame.image, death_plate_rect)
        dsize = (
            plate_w,
            int(death_plate_h * plate_w / death_plate_w),
        )
        death_plage_image = cv2.resize(
            original_death_plate_image, dsize=dsize, interpolation=cv2.INTER_CUBIC
        )

        cv2.imwrite(
            os.path.join(
                intermediate_path, f"death_plage_{death_frame.frame_number}.png"
            ),
            death_plage_image,
        )
        killer_plate_index = most_similar_matcher_index(
            death_plage_image, enemy_plate_matchers
        )
        killer_candidates = [
            player
            for player in enemy_players
            if player.plate_index == killer_plate_index
        ]
        killer = killer_candidates[0] if len(killer_candidates) > 0 else None

        death_event = DeathEvent(
            frame_number=death_frame.frame_number,
            killer_plate_index=killer_plate_index,
            killer=killer,
        )
        death_events.append(death_event)

        killer_info = (
            f"killer.name={killer.name}, killer.weapon={killer.weapon.name}"
            if killer
            else "killer=None"
        )
        logger.info(
            f"Death event: frame_number={death_event.frame_number}, killer_plate_index={killer_plate_index}, {killer_info}"
        )

        if killer is not None:
            death_reason_rect = (723, 353, 474, 58)
            death_reason_image = crop_image(death_frame.image, death_reason_rect)
            death_reason, _ = detect_death_reason(death_reason_image, killer.weapon)
            death_reason_image_path = os.path.join(
                labeling_pool_dir,
                "death_reason",
                death_reason,
                f"{video_name}_{death_frame.frame_number}.png",
            )
            os.makedirs(os.path.dirname(death_reason_image_path), exist_ok=True)
            cv2.imwrite(death_reason_image_path, death_reason_image)

    return death_events


if __name__ == "__main__":
    directory_paths = sys.argv[1:]

    num_processes = os.process_cpu_count()
    logger.info(f"Using up to {num_processes} processes.")

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_directory, directory_paths)

import argparse
import csv
import dataclasses
import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(process)d][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stderr,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GameRow:
    game_id: str
    team_result: str
    local_player_id: str
    local_player_name: str
    local_player_weapon_name: str
    local_player_canonical_weapon_name: str


@dataclasses.dataclass
class PlayerRow:
    game_id: str
    team: str
    player_id: str
    player_name: str
    player_weapon_name: str
    player_canonical_weapon_name: str


@dataclasses.dataclass
class EventRow:
    game_id: str
    event_type: str
    event_frame_number: str
    kill_event_victim_index: str = ""
    death_event_killer_plate_index: str = ""
    enemy_player_id: str = ""
    enemy_player_name: str = ""
    enemy_player_weapon_name: str = ""
    enemy_player_canonical_weapon_name: str = ""


@dataclasses.dataclass
class EnemyWeaponStatRow:
    canonical_weapon_name: str
    match: int
    kill: int
    death: int
    ratio: float


def export_results(directory_paths: list[str], output_dir: str):
    results: list[dict] = []
    for directory_path in directory_paths:
        if os.path.isdir(directory_path):
            result_path = os.path.join(directory_path, "result.json")
        else:
            result_path = directory_path

        if not os.path.exists(result_path):
            logger.warning(f"File does not exist: {result_path}")
            continue

        with open(result_path, "r") as f:
            result = json.load(f)

        logger.info(f"File loaded: {result_path}")
        results.append(result)

    results.sort(key=lambda r: r["id"])

    game_rows: list[GameRow] = []
    player_rows: list[PlayerRow] = []
    event_rows: list[EventRow] = []

    for result in results:
        game_id = result["id"]
        local_player = result["local_player"]
        game_row = GameRow(
            game_id=game_id,
            team_result=result["team_result"],
            local_player_id=local_player["id"],
            local_player_name=local_player["name"],
            local_player_weapon_name=local_player["weapon"]["name"],
            local_player_canonical_weapon_name=local_player["weapon"]["canonical_name"],
        )
        game_rows.append(game_row)

        for player in result["ally_players"]:
            row = PlayerRow(
                game_id=game_id,
                team="ally",
                player_id=player["id"],
                player_name=player["name"],
                player_weapon_name=player["weapon"]["name"],
                player_canonical_weapon_name=player["weapon"]["canonical_name"],
            )
            player_rows.append(row)
        for player in result["enemy_players"]:
            row = PlayerRow(
                game_id=game_id,
                team="enemy",
                player_id=player["id"],
                player_name=player["name"],
                player_weapon_name=player["weapon"]["name"],
                player_canonical_weapon_name=player["weapon"]["canonical_name"],
            )
            player_rows.append(row)

        for event in result["events"]:
            event_kwargs = {
                "game_id": game_id,
                "event_type": event["type"],
                "event_frame_number": str(event["frame_number"]),
            }
            if event["type"] == "kill":
                for i, victim in enumerate(event["victims"]):
                    row = EventRow(
                        kill_event_victim_index=str(i),
                        enemy_player_id=victim["id"],
                        enemy_player_name=victim["name"],
                        enemy_player_weapon_name=victim["weapon"]["name"],
                        enemy_player_canonical_weapon_name=victim["weapon"][
                            "canonical_name"
                        ],
                        **event_kwargs,
                    )
                    event_rows.append(row)
            elif event["type"] == "death":
                killer = event.get("killer") or {}
                row = EventRow(
                    death_event_killer_plate_index=str(event["killer_plate_index"]),
                    enemy_player_id=killer.get("id", ""),
                    enemy_player_name=killer.get("name", ""),
                    enemy_player_weapon_name=killer.get("weapon", {}).get("name", ""),
                    enemy_player_canonical_weapon_name=killer.get("weapon", {}).get(
                        "canonical_name", ""
                    ),
                    **event_kwargs,
                )
                event_rows.append(row)

    os.makedirs(output_dir, exist_ok=True)
    _write_game_rows(game_rows, os.path.join(output_dir, "games.tsv"))
    _write_player_rows(player_rows, os.path.join(output_dir, "players.tsv"))
    _write_event_rows(event_rows, os.path.join(output_dir, "events.tsv"))
    _write_enemy_weapon_stats(
        _build_enemy_weapon_stats(player_rows, event_rows),
        os.path.join(output_dir, "enemy_weapon_stats.tsv"),
    )


def _write_game_rows(rows: list[GameRow], output_path: str):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=[
                "game_id",
                "team_result",
                "local_player_id",
                "local_player_name",
                "local_player_weapon_name",
                "local_player_canonical_weapon_name",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(dataclasses.asdict(row))

    logger.info(f"Successfully exported {len(rows)} game rows to {output_path}.")


def _write_player_rows(rows: list[PlayerRow], output_path: str):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=[
                "game_id",
                "team",
                "player_id",
                "player_name",
                "player_weapon_name",
                "player_canonical_weapon_name",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(dataclasses.asdict(row))

    logger.info(f"Successfully exported {len(rows)} player rows to {output_path}.")


def _write_event_rows(rows: list[EventRow], output_path: str):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=[
                "game_id",
                "event_type",
                "event_frame_number",
                "kill_event_victim_index",
                "death_event_killer_plate_index",
                "enemy_player_id",
                "enemy_player_name",
                "enemy_player_weapon_name",
                "enemy_player_canonical_weapon_name",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(dataclasses.asdict(row))

    logger.info(f"Successfully exported {len(rows)} event rows to {output_path}.")


def _build_enemy_weapon_stats(
    player_rows: list[PlayerRow],
    event_rows: list[EventRow],
) -> list[EnemyWeaponStatRow]:
    enemy_weapon_stat_rows: list[EnemyWeaponStatRow] = []
    enemy_weapon_matches: dict[str, int] = {}
    enemy_weapon_kills: dict[str, int] = {}
    enemy_weapon_deaths: dict[str, int] = {}

    for row in player_rows:
        if row.team == "enemy":
            enemy_weapon_matches.setdefault(row.player_canonical_weapon_name, 0)
            enemy_weapon_matches[row.player_canonical_weapon_name] += 1

    for row in event_rows:
        if row.event_type == "kill":
            enemy_weapon_kills.setdefault(row.enemy_player_canonical_weapon_name, 0)
            enemy_weapon_kills[row.enemy_player_canonical_weapon_name] += 1
        elif row.event_type == "death":
            enemy_weapon_deaths.setdefault(row.enemy_player_canonical_weapon_name, 0)
            enemy_weapon_deaths[row.enemy_player_canonical_weapon_name] += 1

    for weapon_name, match in enemy_weapon_matches.items():
        kill = enemy_weapon_kills.get(weapon_name, 0)
        death = enemy_weapon_deaths.get(weapon_name, 0)
        ratio = (kill - death) / match
        row = EnemyWeaponStatRow(
            canonical_weapon_name=weapon_name,
            match=match,
            kill=kill,
            death=death,
            ratio=ratio,
        )
        enemy_weapon_stat_rows.append(row)

    enemy_weapon_stat_rows.sort(key=lambda r: r.ratio)
    return enemy_weapon_stat_rows


def _write_enemy_weapon_stats(rows: list[EnemyWeaponStatRow], output_path: str):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=[
                "canonical_weapon_name",
                "match",
                "kill",
                "death",
                "(kill - death) / match",
            ],
        )
        writer.writeheader()
        for row in rows:
            d = dataclasses.asdict(row)
            ratio = d.pop("ratio")
            d["(kill - death) / match"] = f"{ratio:.2f}"
            writer.writerow(d)

    logger.info(
        f"Successfully exported {len(rows)} enemy weapon stat rows to {output_path}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("PATH", nargs="+", help="Path to the directory or file")
    parser.add_argument("-o", "--output-dir", required=True)
    args = parser.parse_args()

    export_results(args.PATH, args.output_dir)

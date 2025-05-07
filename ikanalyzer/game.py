import dataclasses
from typing import Literal

from ikanalyzer.splatoon import Weapon


@dataclasses.dataclass
class Player:
    id: str
    name: str
    weapon: Weapon
    plate_index: int = -1

    def to_result_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "weapon": self.weapon.to_result_dict(),
            "plate_index": self.plate_index,
        }


@dataclasses.dataclass
class KillEvent:
    frame_number: int
    second: float
    victims: list[Player]

    def to_result_dict(self) -> dict:
        return {
            "type": "kill",
            "frame_number": self.frame_number,
            "victims": [
                {
                    "id": victim.id,
                    "name": victim.name,
                    "weapon": victim.weapon.to_result_dict(),
                }
                for victim in self.victims
            ],
        }


@dataclasses.dataclass
class DeathEvent:
    frame_number: int
    killer_plate_index: int
    killer: Player | None

    def to_result_dict(self) -> dict:
        return {
            "type": "death",
            "frame_number": self.frame_number,
            "killer_plate_index": self.killer_plate_index,
            "killer": (
                {
                    "id": self.killer.id,
                    "name": self.killer.name,
                    "weapon": self.killer.weapon.to_result_dict(),
                }
                if self.killer
                else None
            ),
        }


@dataclasses.dataclass
class Game:
    id: str
    team_result: Literal["win", "lose", "draw"]
    local_player: Player
    ally_players: list[Player]
    enemy_players: list[Player]
    events: list[KillEvent | DeathEvent]

    def to_result_dict(self) -> dict:
        return {
            "version": "1",
            "id": self.id,
            "team_result": self.team_result,
            "local_player": self.local_player.to_result_dict(),
            "ally_players": [player.to_result_dict() for player in self.ally_players],
            "enemy_players": [player.to_result_dict() for player in self.enemy_players],
            "events": [event.to_result_dict() for event in self.events],
        }

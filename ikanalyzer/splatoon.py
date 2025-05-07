import csv
import dataclasses
import os

weapons_tsv_path = os.path.join(os.path.dirname(__file__), "data", "ja", "weapons.tsv")


@dataclasses.dataclass
class Weapon:
    name: str
    sub: str
    special: str
    path_safe_name: str  # Name that is safe to use in paths
    canonical_name: str  # Canonical name for the weapon. e.g. "スプラシューター" for "オーダーシューター レプリカ"

    def to_result_dict(self) -> dict:
        return {
            "name": self.name,
            "canonical_name": self.canonical_name,
        }


weapons: list[Weapon] = []

with open(weapons_tsv_path) as fh:
    csv_reader = csv.DictReader(fh, delimiter="\t")
    for row in csv_reader:
        name = row["name"].strip()
        sub = row["sub"].strip()
        special = row["special"].strip()
        path_safe_name = row["path_safe_name"].strip()
        canonical_name = row["canonical_name"].strip()
        weapons.append(
            Weapon(
                name=name,
                sub=sub,
                special=special,
                path_safe_name=path_safe_name,
                canonical_name=canonical_name if canonical_name != "" else name,
            )
        )

weapons_by_path_safe_name = {weapon.path_safe_name: weapon for weapon in weapons}

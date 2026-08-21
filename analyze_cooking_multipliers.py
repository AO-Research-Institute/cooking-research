#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable


DATA_PATH = Path(__file__).parent / "data" / "ingredients.json"
ANALYSIS_DIR = Path(__file__).parent / "analysis"


# Preferred constants that are plausible game-design values. ONLY used after a shared compatible interval has been found.
KNOWN_SIMPLE_MULTIPLIERS = [
    Fraction(1, 4),    # 0.25
    Fraction(1, 2),    # 0.5
    Fraction(3, 4),    # 0.75
    Fraction(1, 1),    # 1.0
    Fraction(6, 5),    # 1.2
    Fraction(5, 4),    # 1.25
    Fraction(4, 3),    # 1.333...
    Fraction(3, 2),    # 1.5
    Fraction(8, 5),    # 1.6
    Fraction(13, 8),   # 1.625
    Fraction(5, 3),    # 1.666...
    Fraction(7, 4),    # 1.75
    Fraction(9, 5),    # 1.8
    Fraction(37, 20),  # 1.85
    Fraction(15, 8),   # 1.875
    Fraction(2, 1),    # 2.0
    Fraction(9, 4),    # 2.25
    Fraction(23, 10),  # 2.3
    Fraction(5, 2),    # 2.5
    Fraction(3, 1),    # 3.0
    Fraction(15, 4),   # 3.75
    Fraction(4, 1),    # 4.0
    Fraction(5, 1),    # 5.0
    Fraction(21, 4),   # 5.25
]

MAX_FRACTION_DENOMINATOR = 40

# Minimum number of ingredients for a discovered interval to be treated as a shared rule. Single-item intervals are reported separately.
MIN_SHARED_GROUP_SIZE = 2


@dataclass(frozen=True)
class Bounds:
    lower: Fraction
    upper: Fraction

    @property
    def width(self) -> Fraction:
        return self.upper - self.lower

    def contains_fraction(self, value: Fraction) -> bool:
        return self.lower <= value < self.upper

    def intersects(self, other: "Bounds") -> bool:
        return max(self.lower, other.lower) < min(self.upper, other.upper)

    def intersection(self, other: "Bounds") -> "Bounds | None":
        lower = max(self.lower, other.lower)
        upper = min(self.upper, other.upper)

        if lower >= upper:
            return None

        return Bounds(lower, upper)


@dataclass
class Row:
    item: dict
    bounds: Bounds
    observed_ratio: Fraction
    group_id: int | None = None
    inferred: Fraction | None = None
    confidence: str = "unresolved"


@dataclass
class RuleGroup:
    group_id: int
    members: list[Row]
    bounds: Bounds
    representative: Fraction | None
    confidence: str


def fmt_float(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def fmt_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)

    decimal = float(value)

    # Show clean common decimals where useful.
    if value in KNOWN_SIMPLE_MULTIPLIERS:
        return f"{fmt_float(decimal)}"

    return f"{value.numerator}/{value.denominator} (~{fmt_float(decimal)})"


def fmt_bounds(bounds: Bounds) -> str:
    return (
        f"[{fmt_float(float(bounds.lower))}, "
        f"{fmt_float(float(bounds.upper))})"
    )


def interval(energy: int, cooked: int) -> Bounds:
    """Return exact [lower, upper) floor-compatible multiplier interval."""
    return Bounds(
        lower=Fraction(cooked, energy),
        upper=Fraction(cooked + 1, energy),
    )


def fraction_matches(energy: int, cooked: int, multiplier: Fraction) -> bool:
    """Check floor(E*m) == C exactly, without floating point."""
    scaled_num = energy * multiplier.numerator
    denominator = multiplier.denominator

    return (
        cooked * denominator
        <= scaled_num
        < (cooked + 1) * denominator
    )


def load_items() -> list[dict]:
    with DATA_PATH.open(encoding="utf-8") as handle:
        items = json.load(handle)

    valid: list[dict] = []

    for item in items:
        name = item.get("name", "<unnamed>")
        energy = item.get("energy")
        cooked = item.get("cooked_energy")

        if not isinstance(energy, int) or not isinstance(cooked, int):
            print(f"Skipping {name}: energy/cooked_energy must be integers")
            continue

        if energy <= 0:
            print(f"Skipping {name}: non-positive raw energy")
            continue

        if cooked < 0:
            print(f"Skipping {name}: negative cooked energy")
            continue

        valid.append(item)

    return valid


def build_rows(items: list[dict]) -> list[Row]:
    rows: list[Row] = []

    for item in items:
        energy = item["energy"]
        cooked = item["cooked_energy"]

        rows.append(
            Row(
                item=item,
                bounds=interval(energy, cooked),
                observed_ratio=Fraction(cooked, energy),
            )
        )

    return rows


def intersect_rows(rows: Iterable[Row]) -> Bounds | None:
    rows = list(rows)

    if not rows:
        return None

    lower = max(row.bounds.lower for row in rows)
    upper = min(row.bounds.upper for row in rows)

    if lower >= upper:
        return None

    return Bounds(lower, upper)


def simple_fraction_in_interval(
    bounds: Bounds,
    max_denominator: int = MAX_FRACTION_DENOMINATOR,
) -> Fraction | None:

    for candidate in KNOWN_SIMPLE_MULTIPLIERS:
        if bounds.contains_fraction(candidate):
            return candidate

    midpoint = (bounds.lower + bounds.upper) / 2
    possibilities: list[Fraction] = []

    for denominator in range(1, max_denominator + 1):
        first_num = math.ceil(
            bounds.lower.numerator
            * denominator
            / bounds.lower.denominator
        )

        # Search a small range; interval widths are tiny in practice.
        for numerator in range(first_num, first_num + 4):
            candidate = Fraction(numerator, denominator)

            if bounds.contains_fraction(candidate):
                possibilities.append(candidate)

    if not possibilities:
        return None

    return min(
        possibilities,
        key=lambda value: (
            value.denominator,
            abs(value - midpoint),
            value.numerator,
        ),
    )


def confidence_for_group(
    member_count: int,
    bounds: Bounds,
    representative: Fraction | None,
) -> str:
    """Assign confidence based on group support and interval tightness."""

    width = float(bounds.width)

    if representative is None:
        return "unresolved"

    if member_count >= 5 and width <= 0.01:
        return "very_strong"

    if member_count >= 3 and width <= 0.02:
        return "very_strong"

    if member_count >= 2 and width <= 0.05:
        return "strong"

    if member_count >= 2:
        return "moderate"

    if width <= 0.01:
        return "strong_single"

    return "single"


def max_overlap_group(rows: list[Row]) -> tuple[list[Row], Bounds] | None:
    """Find a largest subset of rows sharing at least one multiplier.

    For intervals on a line, a maximum-overlap point can be found by examining interval endpoints. We test all lower endpoints as candidate points and collect intervals that contain that point.

    Once members are identified, exact common intersection is returned.
    """

    if not rows:
        return None

    candidate_points = sorted({row.bounds.lower for row in rows})

    best_members: list[Row] = []
    best_bounds: Bounds | None = None

    for point in candidate_points:
        members = [
            row
            for row in rows
            if row.bounds.lower <= point < row.bounds.upper
        ]

        if not members:
            continue

        common = intersect_rows(members)

        if common is None:
            continue

        if len(members) > len(best_members):
            best_members = members
            best_bounds = common
            continue

        if len(members) == len(best_members) and best_bounds is not None:
            # Prefer the tighter shared interval when group sizes tie.
            if common.width < best_bounds.width:
                best_members = members
                best_bounds = common

    if not best_members or best_bounds is None:
        return None

    return best_members, best_bounds


def discover_groups(rows: list[Row]) -> list[RuleGroup]:
    """Greedily discover shared multiplier groups from interval overlap."""

    remaining = list(rows)
    groups: list[RuleGroup] = []
    next_group_id = 1

    while remaining:
        result = max_overlap_group(remaining)

        if result is None:
            break

        members, bounds = result

        # Do not consume a singleton as a "shared" group yet.
        if len(members) < MIN_SHARED_GROUP_SIZE:
            break

        representative = simple_fraction_in_interval(bounds)
        confidence = confidence_for_group(
            member_count=len(members),
            bounds=bounds,
            representative=representative,
        )

        group = RuleGroup(
            group_id=next_group_id,
            members=members,
            bounds=bounds,
            representative=representative,
            confidence=confidence,
        )

        for row in members:
            row.group_id = group.group_id
            row.inferred = representative
            row.confidence = confidence

        groups.append(group)

        member_ids = {id(row) for row in members}
        remaining = [
            row
            for row in remaining
            if id(row) not in member_ids
        ]

        next_group_id += 1

    # Remaining ingredients are treated individually.
    for row in remaining:
        representative = simple_fraction_in_interval(row.bounds)

        row.group_id = next_group_id
        row.inferred = representative
        row.confidence = confidence_for_group(
            member_count=1,
            bounds=row.bounds,
            representative=representative,
        )

        groups.append(
            RuleGroup(
                group_id=next_group_id,
                members=[row],
                bounds=row.bounds,
                representative=representative,
                confidence=row.confidence,
            )
        )

        next_group_id += 1

    return groups


def print_table(rows: list[Row]) -> None:
    rows = sorted(
        rows,
        key=lambda row: (
            float("inf")
            if row.inferred is None
            else float(row.inferred),
            row.item["name"].lower(),
        ),
    )

    header = (
        "name | category | raw | cooked | ratio | compatible interval | "
        "inferred rule | confidence"
    )

    print("\n" + header)
    print("-" * len(header))

    for row in rows:
        item = row.item
        inferred = (
            "—"
            if row.inferred is None
            else fmt_fraction(row.inferred)
        )

        print(
            f"{item['name']} | "
            f"{item.get('category', 'unknown')} | "
            f"{item['energy']} | "
            f"{item['cooked_energy']} | "
            f"{float(row.observed_ratio):.3f} | "
            f"{fmt_bounds(row.bounds)} | "
            f"{inferred} | "
            f"{row.confidence}"
        )


def print_group_summary(groups: list[RuleGroup]) -> None:
    print("\nShared multiplier groups")

    ordered = sorted(
        groups,
        key=lambda group: (
            float("inf")
            if group.representative is None
            else float(group.representative),
            -len(group.members),
        ),
    )

    for group in ordered:
        representative = (
            "unresolved"
            if group.representative is None
            else fmt_fraction(group.representative)
        )

        names = ", ".join(
            row.item["name"]
            for row in sorted(
                group.members,
                key=lambda row: row.item["name"].lower(),
            )
        )

        print(
            f"\nGroup {group.group_id}: {representative}x"
            f"\n  members: {len(group.members)}"
            f"\n  shared interval: {fmt_bounds(group.bounds)}"
            f"\n  interval width: {fmt_float(float(group.bounds.width), 8)}"
            f"\n  confidence: {group.confidence}"
            f"\n  ingredients: {names}"
        )


def print_family_notes(rows: list[Row]) -> None:
    print("\nFamily observations")

    for category in ("fruit", "mushroom", "meat", "wheat", "limited"):
        family = [
            row
            for row in rows
            if row.item.get("category") == category
        ]

        if not family:
            continue

        counter = Counter(
            fmt_fraction(row.inferred)
            if row.inferred is not None
            else "unresolved"
            for row in family
        )

        summary = ", ".join(
            f"{rule}x ({count})"
            for rule, count in sorted(counter.items())
        )

        print(f"{category}: {summary}")

    pumpkins = [
        row
        for row in rows
        if "pumpkin" in row.item["name"].lower()
    ]

    watermelons = [
        row
        for row in rows
        if "watermelon" in row.item["name"].lower()
    ]

    if pumpkins:
        print(
            "pumpkins: "
            + ", ".join(
                f"{row.item['name']}="
                f"{float(row.observed_ratio):.3f}x "
                f"{fmt_bounds(row.bounds)}"
                for row in pumpkins
            )
        )

    if watermelons:
        print(
            "watermelons: "
            + ", ".join(
                f"{row.item['name']}="
                f"{float(row.observed_ratio):.3f}x "
                f"{fmt_bounds(row.bounds)}"
                for row in watermelons
            )
        )


def print_special_behavior(rows: list[Row]) -> None:
    special = [
        row
        for row in rows
        if row.item.get("category") == "wheat"
        or "pumpkin" in row.item["name"].lower()
        or row.item["name"] == "Festive Cookie Dough"
    ]

    print("\nLikely special meal-type behavior")

    if not special:
        print("None")
        return

    for row in special:
        inferred = (
            "—"
            if row.inferred is None
            else fmt_fraction(row.inferred)
        )

        print(
            f"{row.item['name']}: "
            f"{row.item['energy']} -> {row.item['cooked_energy']}, "
            f"observed {float(row.observed_ratio):.3f}x, "
            f"compatible {fmt_bounds(row.bounds)}, "
            f"inferred {inferred}x"
        )


def validate_floor_model(rows: list[Row]) -> list[str]:
    failures: list[str] = []

    for row in rows:
        if row.inferred is None:
            failures.append(row.item["name"])
            continue

        if not fraction_matches(
            row.item["energy"],
            row.item["cooked_energy"],
            row.inferred,
        ):
            failures.append(row.item["name"])

    return failures


def family_key(name: str) -> str | None:
    """Very lightweight heuristic for transcription sanity checks."""

    lower = name.lower()

    cap_tokens = (
        "greencap",
        "redcap",
        "purplecap",
        "seacap",
        "sourcap",
        "swiftcap",
        "cloudcap",
    )

    for token in cap_tokens:
        if token in lower:
            return token

    if "pumpkin" in lower:
        return "pumpkin"

    if "watermelon" in lower:
        return "watermelon"

    if "apple" in lower:
        return "apple"

    if "pear" in lower:
        return "pear"

    return None


def print_transcription_warnings(rows: list[Row]) -> None:
    """Flag suspicious family inconsistencies for manual review."""

    families: defaultdict[str, list[Row]] = defaultdict(list)

    for row in rows:
        key = family_key(row.item["name"])

        if key is not None:
            families[key].append(row)

    warnings: list[str] = []

    for family_name, members in families.items():
        inferred_values = [
            float(row.inferred)
            for row in members
            if row.inferred is not None
        ]

        if len(inferred_values) < 2:
            continue

        minimum = min(inferred_values)
        maximum = max(inferred_values)

        if maximum - minimum >= 0.5:
            warnings.append(
                f"{family_name}: unusually wide inferred-rule spread "
                f"({minimum:.3f}x to {maximum:.3f}x) across "
                + ", ".join(row.item["name"] for row in members)
            )

    print("\nTranscription / family sanity warnings")

    if not warnings:
        print("None")
        return

    for warning in warnings:
        print(f"- {warning}")


def export_analysis(
    rows: list[Row],
    groups: list[RuleGroup],
) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    ingredient_output = []

    for row in rows:
        ingredient_output.append(
            {
                "name": row.item["name"],
                "category": row.item.get("category"),
                "energy": row.item["energy"],
                "cooked_energy": row.item["cooked_energy"],
                "observed_ratio": float(row.observed_ratio),
                "compatible_interval": {
                    "lower": float(row.bounds.lower),
                    "upper": float(row.bounds.upper),
                },
                "group_id": row.group_id,
                "inferred_multiplier": (
                    None
                    if row.inferred is None
                    else float(row.inferred)
                ),
                "inferred_multiplier_fraction": (
                    None
                    if row.inferred is None
                    else f"{row.inferred.numerator}/{row.inferred.denominator}"
                ),
                "confidence": row.confidence,
            }
        )

    group_output = []

    for group in groups:
        group_output.append(
            {
                "group_id": group.group_id,
                "member_count": len(group.members),
                "members": [
                    row.item["name"]
                    for row in group.members
                ],
                "compatible_interval": {
                    "lower": float(group.bounds.lower),
                    "upper": float(group.bounds.upper),
                    "width": float(group.bounds.width),
                },
                "representative_multiplier": (
                    None
                    if group.representative is None
                    else float(group.representative)
                ),
                "representative_fraction": (
                    None
                    if group.representative is None
                    else (
                        f"{group.representative.numerator}/"
                        f"{group.representative.denominator}"
                    )
                ),
                "confidence": group.confidence,
            }
        )

    ingredient_path = (
        ANALYSIS_DIR / "ingredient_multiplier_analysis.json"
    )

    group_path = ANALYSIS_DIR / "multiplier_groups.json"

    with ingredient_path.open("w", encoding="utf-8") as handle:
        json.dump(
            ingredient_output,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    with group_path.open("w", encoding="utf-8") as handle:
        json.dump(
            group_output,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("\nAnalysis exports")
    print(f"- {ingredient_path}")
    print(f"- {group_path}")


def main() -> None:
    items = load_items()
    rows = build_rows(items)
    groups = discover_groups(rows)

    print(f"Loaded {len(items)} ingredients from {DATA_PATH}")

    print_table(rows)
    print_group_summary(groups)
    print_family_notes(rows)
    print_special_behavior(rows)
    print_transcription_warnings(rows)

    failures = validate_floor_model(rows)

    print("\nFloor-model flags")
    print("None" if not failures else ", ".join(failures))

    export_analysis(rows, groups)


if __name__ == "__main__":
    main()
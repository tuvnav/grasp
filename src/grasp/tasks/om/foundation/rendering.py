from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Sequence, assert_never
from universal_ml_utils.table import generate_table

from grasp.utils import format_section


type DisplayContent = str | Section | Item | Table
type SectionContent = str | Item | Table


@dataclass
class Section:
    title: str
    content: Sequence[SectionContent] = ()
    # this division makes sure subsections
    # come after other content of the section
    subsections: Sequence[Section] = ()


@dataclass(frozen=True)
class Item:
    text: str
    # sublist
    children: Sequence[Item] = ()


@dataclass
class Table:
    columns: Sequence[str]
    rows: Sequence[Sequence[str]]

    def __post_init__(self):
        for row in self.rows:
            if len(row) != len(self.columns):
                raise ValueError(f"Table row {list(row)} has {len(row)} cells, but there are "
                                 f"{len(self.columns)} columns {list(self.columns)}")



LIST_BULLET = "- "
LIST_INDENT = len(LIST_BULLET)
BLOCK_SEPARATOR = "\n\n"


def format_display_content(
        display_content: DisplayContent,
        level: int = 2,
        depth: int = 0
        ) -> str:
    if level < 1:
        raise ValueError(
            "Level count begins with 1, not 0."
        )
    match display_content:
        case str():
            return display_content.strip()

        case Section(title, content, subsections):
            body = ""
            previous = None
            for element in [*content, *subsections]:
                text = format_display_content(element, level + 1)
                if not text:
                    continue
                if body:
                    both_items = isinstance(previous, Item) and isinstance(element, Item)
                    body += "\n" if both_items else BLOCK_SEPARATOR
                body += text
                previous = element

            return format_section(title, body, level).strip()

        case Item(text, children):
            indent = " " * (LIST_INDENT * depth)
            text = text.strip().replace("\n", "\n" + indent + " " * LIST_INDENT)
            item_group = [f"{indent}{LIST_BULLET}{text}"]
            item_group += [
                format_display_content(child, level, depth + 1)
                for child in children
                ]
            return "\n".join(item_group)

        case Table(columns, rows):
            return generate_table(
                [list(row) for row in rows],
                [list(columns)],
                max_column_width=sys.maxsize
                )

        case _:
            assert_never(display_content)

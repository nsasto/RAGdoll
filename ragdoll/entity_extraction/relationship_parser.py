from __future__ import annotations

import re
from typing import List, Sequence, Type

from pydantic import BaseModel

from ragdoll.utils import json_parse

from .models import Relationship, RelationshipList


class RelationshipOutputParser:
    """Parses LLM output describing relationships into structured models."""

    def __init__(
        self,
        preferred_format: str = "auto",
        *,
        schema_model: Type[BaseModel] = RelationshipList,
    ) -> None:
        self.preferred_format = (preferred_format or "auto").lower()
        self.schema_model = schema_model

    def parse(self, response: str) -> RelationshipList:
        if not response:
            return RelationshipList(relationships=[])

        strategies = self._strategies()

        for parser in strategies:
            relationships = parser(response)
            if relationships:
                return RelationshipList(relationships=relationships)

        return RelationshipList(relationships=[])

    def _strategies(self):
        ordered: List = []
        if self.preferred_format in {"json", "auto"}:
            ordered.append(self._parse_json)
        if self.preferred_format in {"table", "markdown", "auto"}:
            ordered.append(self._parse_markdown_table)
        ordered.append(self._parse_delimited_lines)
        ordered.append(self._parse_arrow_notation)
        return ordered

    def _parse_json(self, text: str) -> List[Relationship]:
        parsed = json_parse(text, self.schema_model)
        if not parsed:
            return []

        relationships = getattr(parsed, "relationships", None)
        if not relationships:
            return []

        normalized: List[Relationship] = []
        for rel in relationships:
            if isinstance(rel, Relationship):
                normalized.append(rel)
            elif isinstance(rel, BaseModel):
                normalized.append(Relationship(**rel.model_dump()))
            elif isinstance(rel, dict):
                try:
                    normalized.append(Relationship(**rel))
                except TypeError:
                    continue
        if normalized:
            return normalized
        return []

    @staticmethod
    def _parse_markdown_table(text: str) -> List[Relationship]:
        rows: List[List[str]] = []
        table_started = False
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("|"):
                if table_started:
                    break
                continue
            table_started = True
            row = [cell.strip() for cell in stripped.strip("|").split("|")]
            rows.append(row)

        if len(rows) < 2:
            return []

        header = [col.lower() for col in rows[0]]
        if not {"subject", "relationship", "object"} <= set(header):
            return []

        data_rows = rows[1:]
        if data_rows and all(
            cell.replace(":", "").replace("-", "").strip() == "" for cell in data_rows[0]
        ):
            data_rows = data_rows[1:]

        rels: List[Relationship] = []
        for values in data_rows:
            if len(values) < len(header):
                continue
            data = dict(zip(header, values))
            subject = data.get("subject")
            relation = data.get("relationship") or data.get("relation")
            obj = data.get("object")
            if subject and relation and obj:
                rels.append(
                    Relationship(subject=subject, relationship=relation, object=obj)
                )
        return rels

    @staticmethod
    def _parse_delimited_lines(text: str) -> List[Relationship]:
        rels: List[Relationship] = []
        arrow_pattern = re.compile(
            r"^(?P<subject>.+?)\s*-+\s*(?P<relationship>.+?)\s*->\s*(?P<object>.+)$"
        )
        for raw_line in text.splitlines():
            line = raw_line.strip()
            line = line.lstrip("-*•").strip()
            if not line:
                continue
            if "|" in line:
                parts = [part.strip() for part in line.split("|")]
            elif ";" in line:
                parts = [part.strip() for part in line.split(";")]
            else:
                match = arrow_pattern.match(line)
                if match:
                    rels.append(
                        Relationship(
                            subject=match.group("subject").strip(),
                            relationship=match.group("relationship").strip(),
                            object=match.group("object").strip(),
                        )
                    )
                continue
            if len(parts) < 3:
                continue
            subject, relation, obj = parts[0], parts[1], parts[2]
            if subject and relation and obj:
                rels.append(
                    Relationship(subject=subject, relationship=relation, object=obj)
                )
        return rels

    @staticmethod
    def _parse_arrow_notation(text: str) -> List[Relationship]:
        pattern = re.compile(
            r"^(?P<subject>.+?)\s*-\s*(?P<relationship>.+?)\s*->\s*(?P<object>.+)$"
        )
        rels: List[Relationship] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            line = line.lstrip("-*•").strip()
            if not line or "->" not in line:
                continue
            match = pattern.match(line)
            if not match:
                continue
            rels.append(
                Relationship(
                    subject=match.group("subject").strip(),
                    relationship=match.group("relationship").strip(),
                    object=match.group("object").strip(),
                )
            )
        return rels

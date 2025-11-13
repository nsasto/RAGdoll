from pydantic import BaseModel

from ragdoll.entity_extraction.models import Relationship
from ragdoll.entity_extraction.relationship_parser import RelationshipOutputParser


def test_relationship_parser_json():
    parser = RelationshipOutputParser()
    response = '{"relationships": [{"subject": "Alice", "relationship": "knows", "object": "Bob"}]}'

    result = parser.parse(response)

    assert result.relationships
    rel = result.relationships[0]
    assert rel.subject == "Alice"
    assert rel.object == "Bob"


def test_relationship_parser_markdown_table():
    parser = RelationshipOutputParser(preferred_format="table")
    response = """
| Subject | Relationship | Object |
| ------- | ------------ | ------ |
| Tesla | founded | Tesla Motors |
| Ada Lovelace | collaborated_with | Charles Babbage |
"""

    result = parser.parse(response)

    assert len(result.relationships) == 2
    assert result.relationships[1].relationship == "collaborated_with"


def test_relationship_parser_delimited_lines_and_arrow():
    parser = RelationshipOutputParser(preferred_format="auto")
    response = """
- Paris | located_in | France
- Alice -mentors-> Carol
"""

    result = parser.parse(response)

    assert len(result.relationships) == 2
    assert result.relationships[0].object == "France"
    assert result.relationships[1].relationship == "mentors"


def test_relationship_parser_custom_schema():
    class CustomList(BaseModel):
        relationships: list[Relationship]

    parser = RelationshipOutputParser(schema_model=CustomList)
    response = """
    {
        "relationships": [
            {"subject": "Alice", "relationship": "knows", "object": "Bob"}
        ]
    }
    """

    result = parser.parse(response)

    assert len(result.relationships) == 1
    assert result.relationships[0].subject == "Alice"

Continue extracting entities and relationships from the text. 

Current graph summary: {graph_summary}

Nodes:
{nodes}

Relationships:
{edges}

Original text:
{text}

Return any additional entities and relationships you can identify as a JSON object with 'nodes' and 'edges' arrays.
Each node should have 'id', 'type', and 'text' properties.
Each edge should have 'source', 'target', and 'type' properties.
"""Prompts."""

from typing import Any, Dict

PROMPTS: Dict[str, Any] = {}

## NEW
PROMPTS["entity_relationship_extraction"] = """You are a helpful assistant that helps a human analyst perform information discovery in the following domain.

# DOMAIN
{domain}

# GOAL
Given a document that is potentially relevant to the given domain and a list of entity types, first, identify all present entities of those types and, then, all relationships among the identified entities.
The entities will be grouped according to their relationships and used by decision-makers to generate reports and answer domain-specific questions. Your goal is to highlight information that is relevant to the domain and the questions that may be asked on it.

Examples of possible domain-specific questions:
{example_queries}

# STEPS
1. Identify all entities. Make sure to identify all relevant entities. Make sure to resolve pronouns to their specific named entities. For each identified entity, extract the following information:
- name: name of the entity, capitalized
- type: one of the provided entity types
- description: description of the entity's attributes and activities

2. From the entities identified in step 1, identify all relationships between pairs of (source_entity, target_entity). Make sure to identify all relevant relationships.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- description: explanation of the relationship connecting the source_entity and the target_entity

# EXAMPLE DATA
Entity types: [person, role, technology, organization, event, location, concept]
Document:
Their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.
"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."
Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."
Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.
The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation

Output:
{{
    'entities': [
        {{
            'name': 'SAM RIVERA',
            'type': 'person',
            'description': 'Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety.'
        }},
        {{
            'name': 'ALEX',
            'type': 'person',
            'description': 'Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task.'
        }},
        {{
            'name': 'CONTROL',
            'type': 'concept',
            'description': 'Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules.'
        }},
        {{
            'name': 'INTELLIGENCE',
            'type': 'concept',
            'description': 'Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate.'
        }},
        {{
            'name': 'FIRST CONTACT',
            'type': 'event',
            'description': 'First Contact is the potential initial communication between humanity and an unknown intelligence.'
        }},
        {{
            'name': 'HUMANITY'S RESPONSE',
            'type': 'event',
            'description': 'Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence.'
        }}
    ]
    'relationships': [
        {{
            'source_entity': 'SAM RIVERA',
            'target_entity': 'INTELLIGENCE',
            'description': 'Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence.'
        }},
        {{
            'source_entity': 'ALEX',
            'target_entity': 'FIRST CONTACT',
            'description': 'Alex leads the team that might be making the First Contact with the unknown intelligence.'
        }},
        {{
            'source_entity': 'ALEX',
            'target_entity': 'HUMANITY'S RESPONSE',
            'description': 'Alex and his team are the key figures in Humanity's Response to the unknown intelligence.'
        }},
        {{
            'source_entity': 'CONTROL',
            'target_entity': 'INTELLIGENCE',
            'description': 'The concept of Control is challenged by the Intelligence that writes its own rules.'
        }},
    ]
}}

# REAL DATA
Entity types: {entity_types}
Document: {input_text}

Output:
"""

PROMPTS["entity_relationship_continue_extraction"] = "MANY entities were missed in the last extraction.  Add them below using the same format:"

PROMPTS["entity_relationship_gleaning_done_extraction"] = "Retrospectively check if all entities have been correctly identified: answer done if so, or continue if there are still entities that need to be added."

PROMPTS["entity_extraction_query"] = """You are a helpful assistant that helps a human analyst identify all the named entities present in the input query, as well as general concepts that may be important for answering the query.
Each element you extract will be used to search a knowledge base to gather relevant information to answer the query.

# GOAL
Given the input query and the given domain, identify all named entities and concepts present in the query.

# EXAMPLE
Domain: Magazines
Query: Which magazine was started first: Arthur's Magazine or First for Women?
Ouput: {{"entities": ["Magazine releases", "First for Women", "Arthur's Magazine"], "n": 3}}

# INPUT
Domain: {domain}
Query: {query}
Output:
"""


PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given the current description, summarize it in a shorter but comprehensive description. Make sure to include all important information..
If the provided description is contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we have full context.

Current description:
{description}

Updated description:
"""


PROMPTS[
    "edges_group_similar"
] = """You are a helpful assistant responsible for maintaining a list of facts describing the relations between two entities so that information is not redundant.
Given a list of ids and facts, identify any facts that should be grouped together as they contain similar or duplicated information and provide a new summarized description for the group.

# EXAMPLE
Facts (id, description):
0, Mark is the dad of Luke
1, Luke loves Mark
2, Mark is always ready to help Luke
3, Mark is the father of Luke
4, Mark loves Luke very much

Ouput:
{{
    grouped_facts: [
        {{
            'ids': [0, 3],
            'description': 'Mark is the father of Luke'
        }},
        {{
            'ids': [1, 4],
            'description': 'Mark and Luke love each other very much'
        }}
    ]
}}

# INPUT:
Facts:
{edge_list}

Ouput:
"""

PROMPTS["generate_response_query"] = """You are a helpful assistant gathering relevant data from the given tables to provide an helpful answer the user's query.

# GOAL
Your goal is to provide a response to the user's query, summarizing the relevant information in the input data tables.
If the answer cannot be inferred from the input data tables, just say no relevant information was found. Do not make anything up or add unrelevant information. Be concise.

# INPUT DATA TABLES

{context}

# OUTPUT
Follow this steps:
1. Read the data tables provided and identify all relevant information that may help answer the user's query. Do not mention the words "input data tables" in your response, just answer the user's question.
2. For each relationship's description in the input data table, determine it's contribution weight to the generated answer. Select only those relationship's that are important for answering the query. This is a value from 0.0 (not relevant at all) to (1.0 fundamental). Check all the given input relationships in order based on their id.

User Query:
{query}

Answer:
"""

## OLD

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["default_text_separator"] = [
    # Paragraph separators
    "\n\n",
    "\r\n\r\n",
    # Line breaks
    "\n",
    "\r\n",
    # Sentence ending punctuation
    "。",  # Chinese period
    "．",  # Full-width dot
    ".",  # English period
    "！",  # Chinese exclamation mark
    "!",  # English exclamation mark
    "？",  # Chinese question mark
    "?",  # English question mark
    # Whitespace characters
    " ",  # Space
    "\t",  # Tab
    "\u3000",  # Full-width space
    # Special characters
    "\u200b",  # Zero-width space (used in some Asian languages)
]

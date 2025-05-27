from dataclasses import dataclass, field
from typing import List, Optional, TypedDict

# Using TypedDict for the 'content' part of a message example,
# as 'action' is optional and dataclasses handle optional fields
# with defaults in a way that might conflict if not carefully managed
# for simple structures like this. TypedDict is more straightforward here.
class MessageContent(TypedDict):
    text: str
    action: Optional[str]

@dataclass
class MessageTurn:
    user: str
    content: MessageContent # Nested TypedDict

@dataclass
class KnowledgeItem:
    id: str
    path: str
    content: str

@dataclass
class Style:
    all: List[str]
    chat: List[str]
    post: List[str]

@dataclass
class ElizaOsCharacter:
    name: str
    bio: List[str]
    lore: List[str]
    messageExamples: List[List[MessageTurn]]
    postExamples: List[str]
    adjectives: List[str]
    topics: List[str]
    style: Style
    knowledge: Optional[List[KnowledgeItem]] = field(default_factory=list) # Not in top-level required, defaults to empty list

    # It's good practice to ensure lists that are required and have minItems > 0
    # are initialized, though actual enforcement of minItems > 0 happens at data validation/parsing time.
    # For fields like 'bio', 'lore', etc., they are required, so they must be provided during instantiation.
    # Dataclasses handle this by default for non-optional fields.
    # The default_factory for knowledge is because it's optional at the top level.
    # If it were required, we wouldn't use default_factory here.
    # The schema implies minItems for required fields, so an empty list would technically be invalid
    # for those upon validation, but the type hint is List[str], not NonEmptyList[str] (which doesn't exist directly).

    def __post_init__(self):
        # Basic validation for minItems can be added here if desired,
        # though full validation against the schema is a larger task.
        # For example, ensuring 'bio' is not empty:
        if not self.bio:
            raise ValueError("bio list cannot be empty")
        if not self.lore:
            raise ValueError("lore list cannot be empty")
        if not self.messageExamples or not any(self.messageExamples):
            raise ValueError("messageExamples list cannot be empty and must contain non-empty conversations")
        for conversation in self.messageExamples:
            if not conversation:
                raise ValueError("Each conversation in messageExamples cannot be empty")
        if not self.postExamples:
            raise ValueError("postExamples list cannot be empty")
        if not self.adjectives:
            raise ValueError("adjectives list cannot be empty")
        if not self.topics:
            raise ValueError("topics list cannot be empty")
        if not self.style.all:
            raise ValueError("style.all list cannot be empty")
        if not self.style.chat:
            raise ValueError("style.chat list cannot be empty")
        if not self.style.post:
            raise ValueError("style.post list cannot be empty")

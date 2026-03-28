from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Annotated, Literal, Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langgraph.types import Send
from langchain.messages import HumanMessage, AIMessage, SystemMessage
import operator
from pathlib import Path



# --------------------------------------------llm--------------------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


#----------------------------------------------structured Outputs----------------------------
class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description="3–6 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(..., description="Target word count for this section (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False



class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]



class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  
    snippet: Optional[str] = None
    source: Optional[str] = None



class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)



class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)




# --------------------------------------------state of graph----------------------------------
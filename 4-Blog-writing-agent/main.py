from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Annotated, Literal, Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.types import Send
from langchain.messages import HumanMessage, AIMessage, SystemMessage
import operator
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()



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
class BlogState(TypedDict):
   topic : str
   mode : str
   needs_research: bool
   queries: List[str]
   evidence : List[EvidenceItem]
   plan : Plan

   sections : Annotated[List[tuple[str, int]], operator.add]  #[ (1, "Introduction"),(2, "How AI Works"),]
   final : str


# ----------------------------------------------functions--------------------------------------

ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 2–5 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.
"""

def router_node(state : BlogState) -> dict:
    
    topic = state["topic"]
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Topic: {topic}"),   
        ])
    
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }


   
def _tavily_search(query: str, max_results : int) :
    results = TavilySearch(max_results = max_results).invoke({"query": query})
    items = results.get("results", []) if isinstance(results, dict) else results

    output = []
    for r in items:
        output.append({
        "title": r.get("title", ""),
        "url": r.get("url", ""),
        "published_at": r.get("published_at", ""),
        "snippet": r.get("content", "")[:1200],
        "source": r.get("source", ""),
        })

    return output



RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state : BlogState):
    
    queries = state.get('queries') or []
    max_results = 2
    # queries = state.get('queries') or []
    # max_results = 5
    raw_results: List[dict] = []
    for query in queries:
        raw_results.extend(_tavily_search(query, max_results))

    if not raw_results:
        return {"evidence": []}

    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=f"Raw results:\n{raw_results}"),
        ]
    )

     # Deduplicate by URL
    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e

    return {"evidence": list(dedup.values())}



ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create 5–9 sections (tasks) suitable for the topic and audience.
- Each task must include:
  1) goal (1 sentence)
  2) 3–6 bullets that are concrete, specific, and non-overlapping
  3) target word count (120–550)

Quality bar:
- Assume the reader is a developer; use correct terminology.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes at least 2 of these somewhere:
  * minimal code sketch / MWE (set requires_code=True for that section)
  * edge cases / failure modes
  * performance/cost considerations
  * security/privacy considerations (if relevant)
  * debugging/observability tips

Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid:
  - Use evidence for up-to-date examples (models/tools/releases) in bullets.
  - Mark sections using fresh info as requires_research=True and requires_citations=True.
- Mode open_book:
  - Set blog_kind = "news_roundup".
  - Every section is about summarizing events + implications.
  - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
  - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources"
    and includes only what can be supported.

Output must strictly match the Plan schema.
"""


def orchestrator_node(state : BlogState):
    
    topic = state.get('topic')
    evidence = state.get('evidence' or [])
    mode = state.get("mode", "closed_book")

    plan = llm.with_structured_output(Plan).invoke([
        SystemMessage( ORCH_SYSTEM),
        HumanMessage(
            content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n\n"
                    f"Evidence (ONLY use for fresh claims; may be empty):\n"
                    f"{[e.model_dump() for e in evidence][:16]}"
                )
            )
    ])

    return {"plan" : plan}



def fanout(state: BlogState):
    return [
        Send("worker", {
            "task": task,
            "topic": state["topic"],
            "plan": state["plan"],
            "mode": state["mode"],
            "evidence": [e.model_dump() for e in state.get("evidence", [])]
        })
        for task in state["plan"].tasks
    ]



WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (±15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
  - For each event claim, attach a source as a Markdown link: ([Source](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
  - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
"""


def worker_node(payload: dict) -> dict:
    
    task = payload["task"] if isinstance(payload["task"], Task) else Task(**payload["task"])
    plan = payload["plan"] if isinstance(payload["plan"], Plan) else Plan(**payload["plan"])
    evidence = [e if isinstance(e, EvidenceItem) else EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:20]
        )

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}



def reducer_node(state: BlogState) -> dict:

    plan = state["plan"]

    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    final_md = f"# {plan.blog_title}\n\n{body}\n"

    filename = f"{plan.blog_title}.md"
    Path(filename).write_text(final_md, encoding="utf-8")

    return {"final": final_md}



def next_route(state: BlogState) -> str:
    return "research" if state["needs_research"] else "orchestrator"
 







#-----------------------------------------------graph -----------------------------------------

graph = StateGraph(BlogState)

# -----------------------------nodes--------------------------
graph.add_node('router', router_node)
graph.add_node('research', research_node)
graph.add_node('orchestrator', orchestrator_node)
graph.add_node('worker', worker_node)
graph.add_node('reducer', reducer_node)


# -----------------------------edges--------------------------
graph.add_edge(START, 'router')
graph.add_conditional_edges('router', next_route,  {"research": "research", "orchestrator": "orchestrator"})
graph.add_edge('research', 'orchestrator')
graph.add_conditional_edges('orchestrator', fanout, ['worker'])
graph.add_edge('worker', 'reducer')
graph.add_edge('reducer', END)

workflow = graph.compile()
 

def run(topic: str):
    out = workflow.invoke(
        {
            "topic": topic,
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "sections": [],
            "final": "",
        }
    )

    return out

run("State of Multimodal LLMs in 2026")


print("working...")
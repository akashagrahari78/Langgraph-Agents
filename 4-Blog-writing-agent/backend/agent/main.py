from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Annotated, Literal, Optional
from math import floor
from pydantic import BaseModel, Field
from langchain_tavily import TavilySearch
from langgraph.types import Send
from langchain.messages import HumanMessage, AIMessage, SystemMessage
import operator,os
from pathlib import Path
from datetime import date, timedelta
from langgraph.types import interrupt
from dotenv import load_dotenv
import sys
load_dotenv()



# --------------------------------------------llm--------------------------------------------
DEFAULT_PROVIDER = "groq"
DEFAULT_MODEL_BY_PROVIDER = {
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4.1-mini",
    "claude": "claude-3-5-sonnet-latest",
}

llm = None
active_provider = DEFAULT_PROVIDER
active_model = DEFAULT_MODEL_BY_PROVIDER[DEFAULT_PROVIDER]


def configure_llm(provider: Optional[str] = None, model: Optional[str] = None):
    global llm, active_provider, active_model

    selected_provider = (provider or DEFAULT_PROVIDER).strip().lower()
    selected_model = (model or DEFAULT_MODEL_BY_PROVIDER.get(selected_provider) or "").strip()

    if selected_provider not in DEFAULT_MODEL_BY_PROVIDER:
        raise ValueError(f"Unsupported LLM provider: {selected_provider}")

    if not selected_model:
        selected_model = DEFAULT_MODEL_BY_PROVIDER[selected_provider]

    if selected_provider == "groq":
        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set.")
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=selected_model,
            temperature=0
        )
    elif selected_provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=selected_model,
            temperature=0
        )
    elif selected_provider == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError("Claude support requires langchain-anthropic to be installed.") from exc

        llm = ChatAnthropic(
            model=selected_model,
            temperature=0
        )

    active_provider = selected_provider
    active_model = selected_model
    return llm


configure_llm()


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


class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)


def desired_section_range(total_words: int) -> tuple[int, int]:
    if total_words <= 700:
        return (3, 4)
    if total_words <= 1200:
        return (4, 5)
    if total_words <= 2200:
        return (5, 6)
    if total_words <= 3200:
        return (6, 7)
    return (7, 8)


def rebalance_plan_to_budget(plan: Plan, total_words: int) -> Plan:
    if not plan.tasks:
        return plan

    min_sections, max_sections = desired_section_range(total_words)
    trimmed_tasks = plan.tasks[:max_sections]

    if len(trimmed_tasks) < min_sections:
        min_sections = len(trimmed_tasks)

    n = max(1, len(trimmed_tasks))

    # Keep planned section targets slightly below the user budget so the final
    # draft can still vary a bit and remain within the requested upper bound.
    planned_total = max(total_words - 80, floor(total_words * 0.88))
    per_section = max(90, floor(planned_total / n))
    max_per_section = max(140, floor((total_words + 120) / n))

    rebalanced_tasks = []
    remaining = planned_total

    for index, task in enumerate(trimmed_tasks):
        sections_left = n - index
        current_target = min(per_section, max_per_section)

        if sections_left == 1:
            current_target = remaining
        else:
            min_remaining_after = max(70 * (sections_left - 1), 0)
            current_target = min(current_target, remaining - min_remaining_after)

        current_target = max(70, current_target)
        remaining -= current_target

        rebalanced_tasks.append(
            task.model_copy(
                update={
                    "target_words": current_target,
                }
            )
        )

    return plan.model_copy(update={"tasks": rebalanced_tasks})


# --------------------------------------------state of graph----------------------------------

class BlogState(TypedDict):
    topic: str
    llm_provider: str
    llm_model: str
    audience: str
    tone: str
    target_word_count: int
    include_code: bool
    include_citations: bool
    include_images: bool

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    plan_approved: bool

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str

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
- Create a section count that fits the requested total length.
  - For short blogs (~500-700 words), prefer 3-4 sections.
  - For medium blogs (~800-1500 words), prefer 4-5 sections.
  - For long blogs, you may use more sections.
- Each task must include:
  1) goal (1 sentence)
  2) 3–6 bullets that are concrete, specific, and non-overlapping
  3) target word count that fits the total budget

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



def orchestrator_node(state : BlogState) -> dict:
    
    topic = state.get('topic')
    evidence = state.get('evidence' or [])
    mode = state.get("mode", "closed_book")
    audience = state.get("audience", "developers")
    tone = state.get("tone", "professional")
    target_word_count = state.get("target_word_count", 2000)
    include_code = state.get("include_code", True)
    include_citations = state.get("include_citations", True)
    include_images = state.get("include_images", False)

    plan = llm.with_structured_output(Plan).invoke([
        SystemMessage( ORCH_SYSTEM),
        HumanMessage(
            content=(
                    f"Topic: {state['topic']}\n"
                    f"Preferred audience: {audience}\n"
                    f"Preferred tone: {tone}\n"
                    f"Approximate total length budget: about {target_word_count} words\n"
                    f"Include code snippets: {include_code}\n"
                    f"Include citations: {include_citations}\n"
                    f"Include images/diagrams: {include_images}\n"
                    f"Mode: {mode}\n\n"
                    "Honor the requested audience and tone in the plan fields.\n"
                    "Use the length budget only as a soft guide for assigning section target words.\n"
                    "If code snippets are disabled, set requires_code=False for every section.\n"
                    "If citations are disabled, only require citations when absolutely necessary for accuracy.\n\n"
                    f"Evidence (ONLY use for fresh claims; may be empty):\n"
                    f"{[e.model_dump() for e in evidence][:16]}"
                )
            )
    ])

    plan = rebalance_plan_to_budget(plan, target_word_count)

    return {"plan" : plan}


def review_plan_node(state: BlogState) -> dict:
    plan = state.get("plan")
    if not plan:
        raise ValueError("Cannot review a plan before it exists.")

    review_payload = {
        "kind": "plan_review",
        "message": "Review the proposed blog plan before drafting begins.",
        "topic": state.get("topic", ""),
        "mode": state.get("mode", "closed_book"),
        "plan": plan.model_dump() if isinstance(plan, Plan) else plan,
    }

    decision = interrupt(review_payload)

    approved = False
    if isinstance(decision, dict):
        approved = bool(decision.get("approved"))
    else:
        approved = bool(decision)

    return {"plan_approved": approved}


def route_after_plan_review(state: BlogState):
    if state.get("plan_approved"):
        return "dispatch_workers"
    return "router"


def dispatch_workers_node(state: BlogState) -> dict:
    return {}



def fanout(state: BlogState):
    return [
        Send("worker", {
            "task": task,
            "topic": state["topic"],
            "plan": state["plan"],
            "mode": state["mode"],
            "evidence": [e.model_dump() for e in state.get("evidence", [])],
            "include_code": state.get("include_code", True),
            "include_citations": state.get("include_citations", True),
            "target_word_count": state.get("target_word_count", 2000),
        })
        for task in state["plan"].tasks
    ]



WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words.
- Do not exceed the section target by more than 10%.
- If the section can be completed clearly in fewer words, prefer fewer words over padding.
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
                    f"Global include_code preference: {payload.get('include_code', True)}\n"
                    f"Global include_citations preference: {payload.get('include_citations', True)}\n\n"
                    f"Requested total blog length: about {payload.get('target_word_count', 2000)} words\n"
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




# subgraph of reducer 

def merge_content(state: BlogState) -> dict:

    plan = state["plan"]

    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: md_with_placeholders must equal input and images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
Return strictly GlobalImagePlan.
"""


def decide_images(state: BlogState) -> dict:
    
    if not state.get("include_images", False):
        return {
            "md_with_placeholders": state["merged_md"],
            "image_specs": [],
        }

    planner = llm.with_structured_output(GlobalImagePlan)
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    image_plan = planner.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Insert placeholders + propose image prompts.\n\n"
                    f"{merged_md}"
                )
            ),
        ]
    )

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


def enforce_word_budget(markdown_text: str, target_word_count: int) -> str:
    if not markdown_text:
        return markdown_text

    current_words = len(markdown_text.split())
    max_allowed = target_word_count + 300

    if current_words <= max_allowed:
        return markdown_text

    compressed = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are editing a Markdown technical blog to fit a strict length budget.\n"
                    "Preserve the title, headings, factual accuracy, citations, and overall structure.\n"
                    "Cut repetition, shorten examples, and tighten prose.\n"
                    "Return only the revised Markdown."
                )
            ),
            HumanMessage(
                content=(
                    f"Target length: about {target_word_count} words\n"
                    f"Hard maximum: {max_allowed} words\n"
                    f"Current length: {current_words} words\n\n"
                    f"{markdown_text}"
                )
            ),
        ]
    ).content.strip()

    return compressed or markdown_text



def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Returns raw image bytes generated by Gemini.
    Requires: pip install google-genai
    Env var: GOOGLE_API_KEY
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )


    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")


def generate_and_place_images(state: BlogState) -> dict:

    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []
    
    import re
    safe_title = re.sub(r'[<>:"/\\|?*]', '', plan.blog_title)

    # If no images requested, just write merged markdown
    if not image_specs:
        md = enforce_word_budget(md, state.get("target_word_count", 2000))
        filename = f"{safe_title}.md"
        Path(filename).write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        # generate only if needed
        if not out_path.exists():
            try:
                img_bytes = _gemini_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                # graceful fallback: keep doc usable
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    md = enforce_word_budget(md, state.get("target_word_count", 2000))
    filename = f"{safe_title}.md"
    Path(filename).write_text(md, encoding="utf-8")
    return {"final": md}




def next_route(state: BlogState) -> str:
    return "research" if state["needs_research"] else "orchestrator"
 









# -----------------------------------------------build reducer subgraph----------------------------
reducer_graph = StateGraph(BlogState)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()




#-----------------------------------------------graph -----------------------------------------
def build_workflow(checkpointer=None):
    graph = StateGraph(BlogState)
    graph.add_node('router', router_node)
    graph.add_node('research', research_node)
    graph.add_node('orchestrator', orchestrator_node)
    graph.add_node('review_plan', review_plan_node)
    graph.add_node('dispatch_workers', dispatch_workers_node)
    graph.add_node('worker', worker_node)
    graph.add_node('reducer', reducer_subgraph)

    graph.add_edge(START, 'router')
    graph.add_conditional_edges('router', next_route,  {"research": "research", "orchestrator": "orchestrator"})
    graph.add_edge('research', 'orchestrator')
    graph.add_edge('orchestrator', 'review_plan')
    graph.add_conditional_edges('review_plan', route_after_plan_review, {"dispatch_workers": "dispatch_workers", "router": "router"})
    graph.add_conditional_edges('dispatch_workers', fanout, ['worker'])
    graph.add_edge('worker', 'reducer')
    graph.add_edge('reducer', END)

    return graph.compile(checkpointer=checkpointer)


workflow = build_workflow()


def run(topic: str, as_of: Optional[str] = None, llm_provider: Optional[str] = None, llm_model: Optional[str] = None):
    if as_of is None:
        as_of = date.today().isoformat()

    configure_llm(llm_provider, llm_model)

    out = workflow.invoke(
        {
            "topic": topic,
            "llm_provider": active_provider,
            "llm_model": active_model,
            "audience": "developers",
            "tone": "professional",
            "target_word_count": 2000,
            "include_code": True,
            "include_citations": True,
            "include_images": False,
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "plan_approved": False,
            "as_of": as_of,
            "recency_days": 7,
            "sections": [],
            "merged_md": "",
            "md_with_placeholders": "",
            "image_specs": [],
            "final": "",
        }
    )

    return out


if __name__ == "__main__":
    topic = "impact of mobile on brain health"
    llm_provider = DEFAULT_PROVIDER
    llm_model = DEFAULT_MODEL_BY_PROVIDER[DEFAULT_PROVIDER]
    if len(sys.argv) > 1 and sys.argv[1].strip():
        topic = sys.argv[1].strip()
    if len(sys.argv) > 2 and sys.argv[2].strip():
        llm_provider = sys.argv[2].strip()
    if len(sys.argv) > 3 and sys.argv[3].strip():
        llm_model = sys.argv[3].strip()

    out = run(topic, llm_provider=llm_provider, llm_model=llm_model)
    plan = out.get("plan")
    import re
    blog_title = getattr(plan, "blog_title", topic)
    safe_title = re.sub(r'[<>:"/\\|?*]', '', blog_title)
    output_file = f"{safe_title}.md"
    final_text = out.get("final", "")

    print(f"Saved markdown file: {output_file}")
    print(f"Final content length: {len(final_text)} chars")
    print("working...")

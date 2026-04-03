"""
Session-based wrapper for the LangGraph blog-writing agent.

It starts the graph, emits progress events, pauses on LangGraph interrupts for
human review, and can resume from stdin commands without losing graph state.
"""
import json
import os
import re
import sys
import uuid
from datetime import date

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BACKEND_DIR = os.path.join(PROJECT_DIR, "backend")
AGENT_DIR = os.path.join(BACKEND_DIR, "agent")

load_dotenv(os.path.join(PROJECT_DIR, ".env"))
load_dotenv(os.path.join(os.path.dirname(PROJECT_DIR), ".env"))

sys.path.insert(0, BACKEND_DIR)


def emit(prefix, payload):
    print(f"{prefix}:{json.dumps(payload)}", flush=True)


def emit_step(index, status):
    emit("STEP", {"index": index, "status": status})


def build_sections(plan, final_markdown):
    if not final_markdown:
        return []

    chunks = re.split(r"(?m)^##\s+", final_markdown)
    sections = []

    if chunks and chunks[0].startswith("# "):
        chunks = chunks[1:]

    for index, chunk in enumerate(chunks, start=1):
        chunk = chunk.strip()
        if not chunk:
            continue

        lines = chunk.splitlines()
        title = lines[0].strip() if lines else f"Section {index}"
        content = f"## {chunk}".strip()
        sections.append(
            {
                "id": index,
                "title": title,
                "content": content,
            }
        )

    if not sections and plan and getattr(plan, "tasks", None):
        return [{"id": task.id, "title": task.title, "content": ""} for task in plan.tasks]

    return sections


def parse_payload():
    raw_payload = sys.argv[1] if len(sys.argv) > 1 else ""
    if not raw_payload:
        return {}

    try:
        return json.loads(raw_payload)
    except json.JSONDecodeError:
        return {"topic": raw_payload}


def build_input_state(payload, default_model_by_provider):
    topic = (payload.get("topic") or "AI in 2026").strip()
    llm_provider = (payload.get("llmProvider") or "groq").strip()
    llm_model = (payload.get("llmModel") or default_model_by_provider.get(llm_provider, "")).strip()

    return {
        "topic": topic,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "audience": (payload.get("audience") or "developers").strip(),
        "tone": (payload.get("tone") or "professional").strip(),
        "target_word_count": int(payload.get("targetWordCount") or 2000),
        "include_code": bool(payload.get("includeCode", True)),
        "include_citations": bool(payload.get("includeCitations", True)),
        "include_images": bool(payload.get("includeImages", False)),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "plan_approved": False,
        "as_of": date.today().isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }


def serialize_final_result(out, payload, default_model_by_provider):
    values = out.value if hasattr(out, "value") else out
    plan = values.get("plan")
    final_markdown = values.get("final", "")
    llm_provider = (payload.get("llmProvider") or "groq").strip()
    llm_model = (payload.get("llmModel") or default_model_by_provider.get(llm_provider, "")).strip()

    return {
        "topic": payload.get("topic", ""),
        "mode": values.get("mode"),
        "llmProvider": llm_provider,
        "llmModel": llm_model,
        "audience": payload.get("audience", "developers"),
        "tone": payload.get("tone", "professional"),
        "targetWordCount": int(payload.get("targetWordCount") or 2000),
        "includeCode": bool(payload.get("includeCode", True)),
        "includeCitations": bool(payload.get("includeCitations", True)),
        "includeImages": bool(payload.get("includeImages", False)),
        "plan": plan.model_dump() if plan else None,
        "sections": build_sections(plan, final_markdown),
        "finalMarkdown": final_markdown,
        "imageSpecs": values.get("image_specs", []),
        "wordCount": len(final_markdown.split()),
    }


def build_workflow_with_progress():
    from agent.main import (
        BlogState,
        DEFAULT_MODEL_BY_PROVIDER,
        configure_llm,
        dispatch_workers_node,
        fanout,
        next_route,
        orchestrator_node,
        reducer_subgraph,
        research_node,
        review_plan_node,
        route_after_plan_review,
        router_node,
        worker_node,
    )

    payload = parse_payload()
    llm_provider = (payload.get("llmProvider") or "groq").strip()
    llm_model = (payload.get("llmModel") or DEFAULT_MODEL_BY_PROVIDER.get(llm_provider, "")).strip()
    configure_llm(llm_provider, llm_model)

    def router_with_progress(state):
        emit_step(0, "active")
        result = router_node(state)
        emit_step(0, "done")
        return result

    def research_with_progress(state):
        emit_step(1, "active")
        result = research_node(state)
        emit_step(1, "done")
        return result

    def orchestrator_with_progress(state):
        emit_step(2, "active")
        result = orchestrator_node(state)
        emit_step(2, "done")
        return result

    def review_with_progress(state):
        return review_plan_node(state)

    def dispatch_workers_with_progress(state):
        emit_step(3, "active")
        return dispatch_workers_node(state)

    def worker_with_progress(payload):
        return worker_node(payload)

    def reducer_with_progress(state):
        emit_step(3, "done")
        emit_step(4, "active")
        result = reducer_subgraph.invoke(state)
        emit_step(4, "done")
        return result

    graph = StateGraph(BlogState)
    graph.add_node("router", router_with_progress)
    graph.add_node("research", research_with_progress)
    graph.add_node("orchestrator", orchestrator_with_progress)
    graph.add_node("review_plan", review_with_progress)
    graph.add_node("dispatch_workers", dispatch_workers_with_progress)
    graph.add_node("worker", worker_with_progress)
    graph.add_node("reducer", reducer_with_progress)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", next_route, {"research": "research", "orchestrator": "orchestrator"})
    graph.add_edge("research", "orchestrator")
    graph.add_edge("orchestrator", "review_plan")
    graph.add_conditional_edges("review_plan", route_after_plan_review, {"dispatch_workers": "dispatch_workers", "router": "router"})
    graph.add_conditional_edges("dispatch_workers", fanout, ["worker"])
    graph.add_edge("worker", "reducer")
    graph.add_edge("reducer", END)

    workflow = graph.compile(checkpointer=InMemorySaver())
    return payload, DEFAULT_MODEL_BY_PROVIDER, workflow


def run_until_pause_or_complete(workflow, command_or_input, config, payload, default_model_by_provider):
    out = workflow.invoke(command_or_input, config=config)
    if hasattr(out, "interrupts"):
        interrupts = out.interrupts or ()
    elif isinstance(out, dict):
        interrupts = out.get("__interrupt__", ())
    else:
        interrupts = ()

    if interrupts:
        interrupt_value = interrupts[0].value
        emit(
            "INTERRUPT",
            {
                "threadId": config["configurable"]["thread_id"],
                "value": interrupt_value,
            },
        )
        return "interrupted"

    emit("RESULT", serialize_final_result(out, payload, default_model_by_provider))
    return "completed"


def main():
    payload, default_model_by_provider, workflow = build_workflow_with_progress()
    initial_state = build_input_state(payload, default_model_by_provider)
    thread_id = payload.get("threadId") or payload.get("sessionId") or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    status = run_until_pause_or_complete(
        workflow,
        initial_state,
        config,
        payload,
        default_model_by_provider,
    )

    if status == "completed":
        return

    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue

        try:
            command = json.loads(raw)
        except json.JSONDecodeError:
            emit("ERROR", {"message": "Malformed resume payload"})
            continue

        action = command.get("action")
        if action == "resume":
            status = run_until_pause_or_complete(
                workflow,
                Command(resume={"approved": bool(command.get("approved"))}),
                config,
                payload,
                default_model_by_provider,
            )
            if status == "completed":
                return
        elif action == "stop":
            return
        else:
            emit("ERROR", {"message": f"Unknown action: {action}"})


if __name__ == "__main__":
    main()

"""Pure helpers for custom flow graphs: validation, topological ordering, and
auto-layout. No LLM or I/O — the executor (workflow.py) uses these; the API and
UI use validate/auto_layout at authoring time."""
from __future__ import annotations

from .schemas import FlowGraph, GraphNode

MAX_GRAPH_NODES = 200


def validate_graph(graph: FlowGraph) -> list[str]:
    """Return a list of human-readable problems; empty means the graph is
    executable. Checks: at least one node, unique ids, dependencies (and fanout
    `over`) reference real nodes, a valid `output`, fanout has `over`, and no
    cycles."""
    errors: list[str] = []
    nodes = graph.nodes
    if not nodes:
        return ["Graph has no nodes."]
    # Bound the graph so a huge payload can't turn validation/layout into a DoS.
    if len(nodes) > MAX_GRAPH_NODES:
        return [f"Graph too large: {len(nodes)} nodes (max {MAX_GRAPH_NODES})."]

    ids = [n.id for n in nodes]
    seen: set[str] = set()
    for node in nodes:
        if not node.id:
            errors.append("A node is missing an id.")
        elif node.id in seen:
            errors.append(f"Duplicate node id: {node.id!r}.")
        seen.add(node.id)

    id_set = set(ids)
    for node in nodes:
        for dep in node.depends_on:
            if dep not in id_set:
                errors.append(f"Node {node.id!r} depends on unknown node {dep!r}.")
        if node.when_node and node.when_node not in id_set:
            errors.append(f"Node {node.id!r} routes on unknown node {node.when_node!r}.")
        if node.kind == "fanout":
            if not node.over:
                errors.append(f"Fanout node {node.id!r} needs an `over` node.")
            elif node.over not in id_set:
                errors.append(f"Fanout node {node.id!r} fans over unknown node {node.over!r}.")

    if not graph.output:
        errors.append("Graph has no `output` node set.")
    elif graph.output not in id_set:
        errors.append(f"`output` refers to unknown node {graph.output!r}.")

    # The data DAG (depends_on + fanout over) must be acyclic; a gate's loop_to is
    # a control back-edge, not a data dependency, so it is allowed and excluded.
    cyclic = _has_cycle(nodes)
    if cyclic:
        errors.append("Graph has a data cycle (dependencies must be acyclic; use a gate's loop_to for loops).")

    for node in nodes:
        if node.kind == "gate":
            if not node.checks:
                errors.append(f"Gate node {node.id!r} needs at least one check.")
            if not node.depends_on:
                errors.append(f"Gate node {node.id!r} needs a `depends_on` — the draft it scores.")
            if not node.loop_to:
                errors.append(f"Gate node {node.id!r} needs a `loop_to` node to revise.")
            elif node.loop_to not in id_set:
                errors.append(f"Gate node {node.id!r} loops to unknown node {node.loop_to!r}.")
            elif not cyclic and node.loop_to not in ancestors(nodes, node.id):
                errors.append(f"Gate node {node.id!r} must loop back to one of its ancestors.")
            elif not cyclic and any(n.kind == "gate" for n in loop_body(nodes, node)):
                errors.append(f"Gate node {node.id!r} has a nested gate in its loop body (unsupported).")
            if node.max_loops < 1:
                errors.append(f"Gate node {node.id!r} needs max_loops >= 1.")

    return errors


def _forward(nodes: list[GraphNode]) -> dict[str, set[str]]:
    """id -> set of node ids that depend on it (reverse of _incoming)."""
    fwd: dict[str, set[str]] = {n.id: set() for n in nodes}
    for nid, deps in _incoming(nodes).items():
        for dep in deps:
            if dep in fwd:
                fwd[dep].add(nid)
    return fwd


def _reach(adj: dict[str, set[str]], start: str) -> set[str]:
    seen: set[str] = set()
    stack = [start]
    while stack:
        cur = stack.pop()
        for nxt in adj.get(cur, set()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def ancestors(nodes: list[GraphNode], target: str) -> set[str]:
    """Strict ancestors of `target` (nodes it transitively depends on)."""
    return _reach(_incoming(nodes), target)


def descendants(nodes: list[GraphNode], start: str) -> set[str]:
    """Strict descendants of `start` (nodes that transitively depend on it)."""
    return _reach(_forward(nodes), start)


def loop_body(nodes: list[GraphNode], gate: GraphNode) -> list[GraphNode]:
    """The nodes re-run each time a gate loops: those downstream of `loop_to`
    (inclusive) and upstream of the gate — in topological order, gate excluded."""
    if not gate.loop_to:
        return []
    body_ids = (descendants(nodes, gate.loop_to) | {gate.loop_to}) & ancestors(nodes, gate.id)
    body_ids.discard(gate.id)
    return [n for n in topological_order(nodes) if n.id in body_ids]


def _incoming(nodes: list[GraphNode]) -> dict[str, set[str]]:
    """Full dependency set per node id, treating both depends_on and a fanout's
    `over` as edges."""
    deps: dict[str, set[str]] = {}
    for node in nodes:
        d = set(node.depends_on)
        if node.kind == "fanout" and node.over:
            d.add(node.over)
        if node.when_node:
            # A routing condition implies the router must run first — an ordering
            # edge, so it counts toward the DAG (unlike a gate's loop_to).
            d.add(node.when_node)
        deps[node.id] = d
    return deps


def _has_cycle(nodes: list[GraphNode]) -> bool:
    try:
        topological_order(nodes)
        return False
    except ValueError:
        return True


def _topo(nodes: list[GraphNode], deps: dict[str, set[str]]) -> list[GraphNode]:
    """Kahn's algorithm over a given dependency map. Raises ValueError on cycle."""
    by_id = {n.id: n for n in nodes}
    remaining = {nid: {d for d in ds if d in by_id} for nid, ds in deps.items()}
    ready = [nid for nid, ds in remaining.items() if not ds]
    ordered: list[str] = []
    while ready:
        nid = ready.pop(0)
        ordered.append(nid)
        for other, ds in remaining.items():
            if nid in ds:
                ds.discard(nid)
                if not ds and other not in ordered and other not in ready:
                    ready.append(other)
    if len(ordered) != len(nodes):
        raise ValueError("cycle")
    return [by_id[nid] for nid in ordered]


def topological_order(nodes: list[GraphNode]) -> list[GraphNode]:
    """Data-DAG order (depends_on / fanout over / when_node). Raises on cycle."""
    return _topo(nodes, _incoming(nodes))


def execution_order(nodes: list[GraphNode]) -> list[GraphNode]:
    """Topological order with gate barriers: a node downstream of a gate's loop
    body (but outside it) must run AFTER the gate, so it never reads a draft that
    a later loop iteration will refine. Adds control edges gate -> such nodes."""
    deps = {nid: set(ds) for nid, ds in _incoming(nodes).items()}
    for gate in nodes:
        if gate.kind != "gate" or not gate.loop_to:
            continue
        body = {n.id for n in loop_body(nodes, gate)}
        if not body:
            continue
        anc = ancestors(nodes, gate.id)
        downstream: set[str] = set()
        for member in body:
            downstream |= descendants(nodes, member)
        for nid in downstream:
            if nid in body or nid == gate.id or nid in anc:
                continue  # excluding ancestors keeps the added edge acyclic
            deps[nid].add(gate.id)
    return _topo(nodes, deps)


def auto_layout(graph: FlowGraph) -> FlowGraph:
    """Assign lane = dependency depth, order = index within the lane. A sane
    default the author can then nudge; deterministic so rendering is stable."""
    nodes = graph.nodes
    if not nodes or _has_cycle(nodes):
        return graph
    deps = _incoming(nodes)
    depth: dict[str, int] = {}
    for node in topological_order(nodes):
        node_deps = [d for d in deps[node.id] if d in depth]
        depth[node.id] = (max((depth[d] for d in node_deps), default=-1) + 1)
    lane_counts: dict[int, int] = {}
    laid_out: list[GraphNode] = []
    for node in nodes:
        lane = depth.get(node.id, 0)
        order = lane_counts.get(lane, 0)
        lane_counts[lane] = order + 1
        laid_out.append(node.model_copy(update={"lane": lane, "order": order}))
    return graph.model_copy(update={"nodes": laid_out})

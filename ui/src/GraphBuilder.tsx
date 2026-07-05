import { useRef, useState } from "react";
import { Plus, Star, Trash2, X } from "lucide-react";
import type { FlowGraph, GraphNode, LmModel } from "./types";
import { modelKey } from "./api";

const COL = 210;
const ROW = 120;
const NODE_W = 172;
const NODE_H = 92;

const KINDS: Array<GraphNode["kind"]> = ["llm", "fanout", "gate"];

function anchor(node: GraphNode) {
  const x = (node.lane ?? 0) * COL;
  const y = (node.order ?? 0) * ROW;
  return { x, y, cx: x + NODE_W / 2, cy: y + NODE_H / 2, rx: x + NODE_W, ry: y + NODE_H / 2, lx: x, ly: y + NODE_H / 2 };
}

/** Visual, drag-to-wire editor for a custom flow graph. Positions come from each
 * node's saved lane/order (the same layout the pop-out renders), so what you
 * arrange here is exactly what you watch run. */
export function GraphBuilder({
  graph,
  onChange,
  models
}: {
  graph: FlowGraph;
  onChange: (graph: FlowGraph) => void;
  models: LmModel[];
}) {
  const nodes = graph.nodes ?? [];
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [connectFrom, setConnectFrom] = useState<string | null>(null);
  const dragRef = useRef<{ id: string } | null>(null);
  const canvasRef = useRef<HTMLDivElement | null>(null);

  const selected = nodes.find((n) => n.id === selectedId) ?? null;
  const maxLane = Math.max(1, ...nodes.map((n) => (n.lane ?? 0) + 1));
  const maxOrder = Math.max(1, ...nodes.map((n) => (n.order ?? 0) + 1));

  function commit(next: GraphNode[], output = graph.output) {
    onChange({ ...graph, nodes: next, output });
  }

  function updateNode(id: string, patch: Partial<GraphNode>) {
    commit(nodes.map((n) => (n.id === id ? { ...n, ...patch } : n)));
  }

  function addNode() {
    const existing = new Set(nodes.map((n) => n.id));
    let i = nodes.length + 1;
    let id = `n${i}`;
    while (existing.has(id)) id = `n${++i}`;
    const lane = nodes.length === 0 ? 0 : maxLane;
    const node: GraphNode = { id, title: id, kind: "llm", prompt: "", depends_on: [], lane, order: 0 };
    commit([...nodes, node], graph.output || id);
    setSelectedId(id);
  }

  function deleteNode(id: string) {
    const next = nodes
      .filter((n) => n.id !== id)
      .map((n) => ({
        ...n,
        depends_on: (n.depends_on ?? []).filter((d) => d !== id),
        over: n.over === id ? "" : n.over,
        loop_to: n.loop_to === id ? "" : n.loop_to,
        when_node: n.when_node === id ? "" : n.when_node
      }));
    commit(next, graph.output === id ? "" : graph.output);
    if (selectedId === id) setSelectedId(null);
  }

  function addDependency(source: string, target: string) {
    if (source === target) return;
    const t = nodes.find((n) => n.id === target);
    if (!t) return;
    if ((t.depends_on ?? []).includes(source)) return;
    updateNode(target, { depends_on: [...(t.depends_on ?? []), source] });
  }

  function removeDependency(source: string, target: string) {
    const t = nodes.find((n) => n.id === target);
    if (!t) return;
    updateNode(target, { depends_on: (t.depends_on ?? []).filter((d) => d !== source) });
  }

  function onNodePointerDown(event: React.PointerEvent, id: string) {
    if (connectFrom) {
      if (connectFrom !== id) addDependency(connectFrom, id);
      setConnectFrom(null);
      return;
    }
    dragRef.current = { id };
    (event.target as HTMLElement).setPointerCapture?.(event.pointerId);
    setSelectedId(id);
  }

  function onCanvasPointerMove(event: React.PointerEvent) {
    if (!dragRef.current || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const lane = Math.max(0, Math.round((event.clientX - rect.left - NODE_W / 2) / COL));
    const order = Math.max(0, Math.round((event.clientY - rect.top - NODE_H / 2) / ROW));
    const node = nodes.find((n) => n.id === dragRef.current!.id);
    if (node && (node.lane !== lane || node.order !== order)) {
      updateNode(node.id, { lane, order });
    }
  }

  function endDrag() {
    dragRef.current = null;
  }

  const width = Math.max(maxLane + 1, 3) * COL;
  const height = Math.max(maxOrder + 1, 3) * ROW;

  return (
    <div className="graphBuilder">
      <div className="builderToolbar">
        <button className="secondaryButton" onClick={addNode}>
          <Plus size={15} /> Add node
        </button>
        {connectFrom ? (
          <span className="builderHint">Click a target node to add a dependency… <button className="linklike" onClick={() => setConnectFrom(null)}>cancel</button></span>
        ) : (
          <span className="builderHint">Drag to arrange · use a node's ◦ handle to wire a dependency · ★ marks the output.</span>
        )}
      </div>

      <div
        className="builderCanvas"
        ref={canvasRef}
        style={{ width, height }}
        onPointerMove={onCanvasPointerMove}
        onPointerUp={endDrag}
        onPointerLeave={endDrag}
        onClick={(e) => { if (e.target === canvasRef.current) { setSelectedId(null); setConnectFrom(null); } }}
      >
        <svg className="builderEdges" width={width} height={height}>
          {nodes.map((node) =>
            (node.depends_on ?? []).map((dep) => {
              const s = nodes.find((n) => n.id === dep);
              if (!s) return null;
              const a = anchor(s);
              const b = anchor(node);
              return (
                <line
                  key={`${dep}->${node.id}`}
                  className="edge dep"
                  x1={a.rx} y1={a.ry} x2={b.lx} y2={b.ly}
                  onClick={() => removeDependency(dep, node.id)}
                >
                  <title>click to remove dependency</title>
                </line>
              );
            })
          )}
          {nodes.map((node) => {
            const specials: Array<[string | undefined, string]> = [
              [node.over, "over"],
              [node.when_node, "when"],
              [node.loop_to, "loop"]
            ];
            return specials.map(([target, kind]) => {
              if (!target) return null;
              const s = kind === "loop" ? anchor(node) : anchor(nodes.find((n) => n.id === target) ?? node);
              const b = kind === "loop" ? anchor(nodes.find((n) => n.id === target) ?? node) : anchor(node);
              if (!nodes.find((n) => n.id === target)) return null;
              return <line key={`${node.id}-${kind}`} className={`edge ${kind}`} x1={s.cx} y1={s.cy} x2={b.cx} y2={b.cy} />;
            });
          })}
        </svg>

        {nodes.map((node) => {
          const a = anchor(node);
          return (
            <div
              key={node.id}
              className={`builderNode ${node.kind} ${selectedId === node.id ? "selected" : ""} ${connectFrom ? "targetable" : ""}`}
              style={{ left: a.x, top: a.y, width: NODE_W }}
              onPointerDown={(e) => onNodePointerDown(e, node.id)}
            >
              <div className="bnHead">
                <span className="bnKind">{node.kind}</span>
                {graph.output === node.id && <Star size={13} className="bnStar" />}
              </div>
              <strong className="bnTitle">{node.title || node.id}</strong>
              <code className="bnId">{node.id}</code>
              <button
                className="bnHandle"
                title="Wire a dependency to another node"
                onPointerDown={(e) => { e.stopPropagation(); setConnectFrom(node.id); setSelectedId(node.id); }}
              >
                ◦
              </button>
            </div>
          );
        })}
      </div>

      {selected && (
        <NodeEditor
          key={selected.id}
          node={selected}
          nodes={nodes}
          models={models}
          isOutput={graph.output === selected.id}
          onChange={(patch) => updateNode(selected.id, patch)}
          onSetOutput={() => commit(nodes, selected.id)}
          onDelete={() => deleteNode(selected.id)}
        />
      )}
    </div>
  );
}

function NodeEditor({
  node,
  nodes,
  models,
  isOutput,
  onChange,
  onSetOutput,
  onDelete
}: {
  node: GraphNode;
  nodes: GraphNode[];
  models: LmModel[];
  isOutput: boolean;
  onChange: (patch: Partial<GraphNode>) => void;
  onSetOutput: () => void;
  onDelete: () => void;
}) {
  const others = nodes.filter((n) => n.id !== node.id);
  const checks = node.checks ?? [];
  return (
    <div className="nodeEditor">
      <div className="sectionHeader">
        <h3>Node: {node.id}</h3>
        <div className="runActions">
          <button className={isOutput ? "primaryButton" : "secondaryButton"} onClick={onSetOutput} disabled={isOutput}>
            <Star size={14} /> {isOutput ? "Output" : "Set output"}
          </button>
          <button className="profileDelete" title="Delete node" onClick={onDelete}>
            <Trash2 size={15} />
          </button>
        </div>
      </div>
      <div className="formGrid">
        <label>Id<input value={node.id} onChange={(e) => onChange({ id: e.target.value.trim() })} /></label>
        <label>Title<input value={node.title ?? ""} onChange={(e) => onChange({ title: e.target.value })} /></label>
        <label>
          Kind
          <select value={node.kind} onChange={(e) => onChange({ kind: e.target.value as GraphNode["kind"] })}>
            {KINDS.map((k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
        <label>
          Model
          <input list="builder-models" value={node.model ?? ""} placeholder="(profile default)" onChange={(e) => onChange({ model: e.target.value })} />
          <datalist id="builder-models">
            {models.map((m, i) => <option key={i} value={modelKey(m)} />)}
          </datalist>
        </label>
        <label>
          Lane<input type="number" min={0} value={node.lane ?? 0} onChange={(e) => onChange({ lane: Math.max(0, Number(e.target.value) || 0) })} />
        </label>
        <label>
          Order<input type="number" min={0} value={node.order ?? 0} onChange={(e) => onChange({ order: Math.max(0, Number(e.target.value) || 0) })} />
        </label>
      </div>
      <label>Prompt<textarea className="compactText" value={node.prompt ?? ""} onChange={(e) => onChange({ prompt: e.target.value })} placeholder="Use {input}, {{node_id}}, {item}" /></label>

      {node.kind === "fanout" && (
        <label>
          Fan out over
          <select value={node.over ?? ""} onChange={(e) => onChange({ over: e.target.value })}>
            <option value="">— pick a node whose output is a list —</option>
            {others.map((n) => <option key={n.id} value={n.id}>{n.id}</option>)}
          </select>
        </label>
      )}

      {node.kind === "gate" && (
        <div className="gateFields">
          <div className="formGrid">
            <label>
              Loop back to
              <select value={node.loop_to ?? ""} onChange={(e) => onChange({ loop_to: e.target.value })}>
                <option value="">— refine node (an ancestor) —</option>
                {others.map((n) => <option key={n.id} value={n.id}>{n.id}</option>)}
              </select>
            </label>
            <label>Max loops<input type="number" min={1} value={node.max_loops ?? 3} onChange={(e) => onChange({ max_loops: Math.max(1, Number(e.target.value) || 1) })} /></label>
          </div>
          <div className="checksEditor">
            <span>Checks (term ≥ min):</span>
            {checks.map((c, i) => (
              <div className="checkRow" key={i}>
                <input value={c.term} placeholder="e.g. scope" onChange={(e) => onChange({ checks: checks.map((x, j) => (j === i ? { ...x, term: e.target.value } : x)) })} />
                <input type="number" step="0.05" min={0} max={1} value={c.min} onChange={(e) => onChange({ checks: checks.map((x, j) => (j === i ? { ...x, min: Number(e.target.value) } : x)) })} />
                <button className="profileDelete" onClick={() => onChange({ checks: checks.filter((_, j) => j !== i) })}><X size={13} /></button>
              </div>
            ))}
            <button className="secondaryButton" onClick={() => onChange({ checks: [...checks, { term: "", min: 0.7 }] })}><Plus size={13} /> Add check</button>
          </div>
        </div>
      )}

      <div className="formGrid">
        <label>
          Run only when (route)
          <select value={node.when_node ?? ""} onChange={(e) => onChange({ when_node: e.target.value })}>
            <option value="">— always —</option>
            {others.map((n) => <option key={n.id} value={n.id}>{n.id}</option>)}
          </select>
        </label>
        {node.when_node && (
          <label>output contains<input value={node.when_equals ?? ""} placeholder="e.g. code" onChange={(e) => onChange({ when_equals: e.target.value })} /></label>
        )}
      </div>
    </div>
  );
}

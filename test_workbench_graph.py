import asyncio
import unittest


class GraphValidationTests(unittest.TestCase):
    def _graph(self, nodes, output):
        from workbench.schemas import FlowGraph, GraphNode

        return FlowGraph(nodes=[GraphNode(**n) for n in nodes], output=output)

    def test_valid_graph_has_no_errors(self):
        from workbench.graph import validate_graph

        g = self._graph(
            [
                {"id": "a", "kind": "llm"},
                {"id": "b", "kind": "llm", "depends_on": ["a"]},
            ],
            output="b",
        )
        self.assertEqual(validate_graph(g), [])

    def test_missing_dependency_and_output_flagged(self):
        from workbench.graph import validate_graph

        g = self._graph([{"id": "a", "depends_on": ["ghost"]}], output="nope")
        errors = validate_graph(g)
        self.assertTrue(any("ghost" in e for e in errors))
        self.assertTrue(any("nope" in e for e in errors))

    def test_cycle_is_detected(self):
        from workbench.graph import validate_graph

        g = self._graph(
            [
                {"id": "a", "depends_on": ["b"]},
                {"id": "b", "depends_on": ["a"]},
            ],
            output="a",
        )
        self.assertTrue(any("cycle" in e.lower() for e in validate_graph(g)))

    def test_fanout_requires_over(self):
        from workbench.graph import validate_graph

        g = self._graph([{"id": "f", "kind": "fanout"}], output="f")
        self.assertTrue(any("over" in e.lower() for e in validate_graph(g)))

    def test_gate_requires_checks_and_valid_loop_target(self):
        from workbench.graph import validate_graph

        # Gate with no checks and a loop_to that isn't an ancestor.
        g = self._graph(
            [
                {"id": "draft", "kind": "llm"},
                {"id": "gate", "kind": "gate", "depends_on": ["draft"], "loop_to": "elsewhere"},
                {"id": "elsewhere", "kind": "llm"},
            ],
            output="draft",
        )
        errors = validate_graph(g)
        self.assertTrue(any("check" in e.lower() for e in errors))
        self.assertTrue(any("ancestor" in e.lower() for e in errors))

    def test_valid_gate_loop_passes_validation(self):
        from workbench.schemas import FlowGraph, GraphCheck, GraphNode
        from workbench.graph import validate_graph, loop_body

        g = FlowGraph(
            nodes=[
                GraphNode(id="draft", kind="llm"),
                GraphNode(id="refine", kind="llm", depends_on=["draft"]),
                GraphNode(id="gate", kind="gate", depends_on=["refine"],
                          checks=[GraphCheck(term="scope", min=0.7)], loop_to="refine"),
            ],
            output="refine",
        )
        self.assertEqual(validate_graph(g), [])
        body = [n.id for n in loop_body(g.nodes, g.nodes[2])]
        self.assertEqual(body, ["refine"])  # only refine re-runs each loop

    def test_auto_layout_assigns_increasing_lanes(self):
        from workbench.graph import auto_layout

        g = self._graph(
            [
                {"id": "a"},
                {"id": "b", "depends_on": ["a"]},
                {"id": "c", "depends_on": ["b"]},
            ],
            output="c",
        )
        laid = {n.id: n.lane for n in auto_layout(g).nodes}
        self.assertEqual(laid["a"], 0)
        self.assertEqual(laid["b"], 1)
        self.assertEqual(laid["c"], 2)


class GraphExecutionTests(unittest.TestCase):
    def test_graph_flow_runs_nodes_in_order_with_fanout(self):
        async def scenario():
            from workbench.schemas import FlowGraph, GraphNode, Profile, RunRequest
            from workbench.workflow import run_workflow

            async def fake(model, messages, profile, json_mode=False, on_token=None, **kwargs):
                p = messages[0]["content"]
                if "as json" in p.lower():
                    return '{"items":["alpha","beta"]}'
                if p.startswith("do "):
                    return "did:" + p
                if "synthesize" in p.lower():
                    return "FINAL ANSWER"
                return "x"

            graph = FlowGraph(
                nodes=[
                    GraphNode(id="plan", kind="llm", prompt="List items as JSON for: {input}", lane=0, order=0),
                    GraphNode(id="work", kind="fanout", over="plan", prompt="do {item}", lane=1, order=0),
                    GraphNode(id="final", kind="llm", depends_on=["work"], prompt="synthesize {{work}}", lane=2, order=0),
                ],
                output="final",
            )
            profile = Profile(
                name="G", provider="openai-compatible", workflow="graph", graph=graph,
                aggregator_model="m",
            )
            steps = []
            record = await run_workflow(
                RunRequest(prompt="build a thing", profile=profile),
                complete_fn=fake,
                on_step=lambda s: steps.append(s.stage),
            )
            return record, steps

        record, steps = asyncio.run(scenario())
        self.assertEqual(record.workflow, "graph")
        self.assertEqual(record.final_output, "FINAL ANSWER")
        # plan once, fanout expanded to two "work" instances, final once.
        self.assertEqual(steps.count("work"), 2)
        self.assertIn("plan", steps)
        self.assertIn("final", steps)

    def test_graph_trace_steps_carry_upstream_dep_ids(self):
        async def scenario():
            from workbench.schemas import FlowGraph, GraphNode, Profile, RunRequest
            from workbench.workflow import run_workflow

            async def fake(model, messages, profile, json_mode=False, on_token=None, **kwargs):
                p = messages[0]["content"]
                if "as json" in p.lower():
                    return '{"items":["alpha","beta"]}'
                if p.startswith("do "):
                    return "did:" + p
                return "FINAL"

            graph = FlowGraph(
                nodes=[
                    GraphNode(id="plan", kind="llm", prompt="List items as JSON for: {input}", lane=0),
                    GraphNode(id="work", kind="fanout", over="plan", prompt="do {item}", lane=1),
                    GraphNode(id="final", kind="llm", depends_on=["work"], prompt="synthesize {{work}}", lane=2),
                ],
                output="final",
            )
            profile = Profile(name="G", workflow="graph", graph=graph, aggregator_model="m")
            return await run_workflow(RunRequest(prompt="x", profile=profile), complete_fn=fake)

        record = asyncio.run(scenario())
        plan = next(s for s in record.trace if s.stage == "plan")
        works = [s for s in record.trace if s.stage == "work"]
        final = next(s for s in record.trace if s.stage == "final")
        # Each fanout instance points at the node it fans over; the join points
        # at every instance that fed it.
        for work in works:
            self.assertEqual(work.metadata["deps"], [plan.id])
        self.assertEqual(sorted(final.metadata["deps"]), sorted(w.id for w in works))

    def test_fanout_items_parses_bare_json_array_and_object_and_lines(self):
        from workbench.workflow import _fanout_items

        self.assertEqual(_fanout_items('["a","b","c"]'), ["a", "b", "c"])
        self.assertEqual(_fanout_items('{"items":["x","y"]}'), ["x", "y"])
        self.assertEqual(_fanout_items("one\ntwo\nthree"), ["one", "two", "three"])
        # Dict items collapse to their prompt/title.
        self.assertEqual(_fanout_items('[{"prompt":"do a"},{"title":"B"}]'), ["do a", "B"])
        # Runaway guard.
        self.assertEqual(len(_fanout_items("[" + ",".join(['"x"'] * 50) + "]")), 8)

    def _loop_graph(self, max_loops):
        from workbench.schemas import FlowGraph, GraphCheck, GraphNode, Profile

        graph = FlowGraph(
            nodes=[
                GraphNode(id="draft", kind="llm", prompt="draft: {input}", lane=0),
                GraphNode(id="refine", kind="llm", depends_on=["draft"],
                          prompt="revise using {{gate}} and {{refine}}", lane=1),
                GraphNode(id="gate", kind="gate", depends_on=["refine"],
                          checks=[GraphCheck(term="quality", min=0.7)], loop_to="refine",
                          max_loops=max_loops, lane=2),
            ],
            output="refine",
        )
        return Profile(name="L", workflow="graph", graph=graph, aggregator_model="m", evaluator_model="m")

    def test_gate_loops_until_checks_pass(self):
        async def scenario():
            from workbench.schemas import RunRequest
            from workbench.workflow import run_workflow

            calls = {"gate": 0, "refine": 0}

            async def fake(model, messages, profile, json_mode=False, on_token=None, **kwargs):
                p = messages[0]["content"]
                if "Return ONLY JSON" in p:
                    calls["gate"] += 1
                    score = 0.9 if calls["gate"] >= 3 else 0.2  # fail twice, then pass
                    return '{"quality": {"score": %s, "note": "n"}}' % score
                if "revise" in p.lower():
                    calls["refine"] += 1
                    return "revision %d" % calls["refine"]
                return "first draft"

            record = await run_workflow(
                RunRequest(prompt="do it", profile=self._loop_graph(5)), complete_fn=fake
            )
            return record, calls

        record, calls = asyncio.run(scenario())
        self.assertEqual(calls["gate"], 3)   # initial + 2 loops
        self.assertEqual(calls["refine"], 3)  # initial + 2 re-runs
        self.assertEqual(record.final_output, "revision 3")

    def test_gate_respects_max_loops(self):
        async def scenario():
            from workbench.schemas import RunRequest
            from workbench.workflow import run_workflow

            calls = {"gate": 0, "refine": 0}

            async def fake(model, messages, profile, json_mode=False, on_token=None, **kwargs):
                p = messages[0]["content"]
                if "Return ONLY JSON" in p:
                    calls["gate"] += 1
                    return '{"quality": {"score": 0.1, "note": "never good"}}'  # always fails
                if "revise" in p.lower():
                    calls["refine"] += 1
                    return "revision %d" % calls["refine"]
                return "first draft"

            record = await run_workflow(
                RunRequest(prompt="do it", profile=self._loop_graph(2)), complete_fn=fake
            )
            return calls, record

        calls, record = asyncio.run(scenario())
        self.assertEqual(calls["gate"], 3)   # initial + 2 (capped)
        self.assertEqual(calls["refine"], 3)  # initial + 2 re-runs, then stop
        self.assertEqual(record.workflow, "graph")

    def test_conditional_branch_runs_only_the_matching_path(self):
        async def scenario():
            from workbench.schemas import FlowGraph, GraphNode, Profile, RunRequest
            from workbench.workflow import run_workflow

            async def fake(model, messages, profile, json_mode=False, on_token=None, **kwargs):
                p = messages[0]["content"].lower()
                if "classify" in p:
                    return "code"
                if "code path" in p:
                    return "CODE-OUT"
                if "prose path" in p:
                    return "PROSE-OUT"
                if "join" in p:
                    return "JOINED"
                return "x"

            graph = FlowGraph(
                nodes=[
                    GraphNode(id="router", kind="llm", prompt="Classify: {input}"),
                    GraphNode(id="code", kind="llm", depends_on=["router"], when_node="router", when_equals="code", prompt="code path"),
                    GraphNode(id="prose", kind="llm", depends_on=["router"], when_node="router", when_equals="prose", prompt="prose path"),
                    GraphNode(id="join", kind="llm", depends_on=["code", "prose"], prompt="join {{code}} {{prose}}"),
                ],
                output="join",
            )
            profile = Profile(name="B", workflow="graph", graph=graph, aggregator_model="m")
            steps = []
            record = await run_workflow(
                RunRequest(prompt="x", profile=profile), complete_fn=fake, on_step=lambda s: steps.append(s.stage)
            )
            return record, steps

        record, steps = asyncio.run(scenario())
        self.assertIn("router", steps)
        self.assertIn("code", steps)
        self.assertIn("join", steps)          # join runs — one live input
        self.assertNotIn("prose", steps)       # untaken branch skipped
        self.assertEqual(record.final_output, "JOINED")

    def test_execution_order_puts_gate_before_downstream_of_loop_body(self):
        from workbench.schemas import FlowGraph, GraphCheck, GraphNode
        from workbench.graph import execution_order

        # finalize depends only on draft (a loop-body node), NOT on the gate.
        # The barrier must still order finalize AFTER the gate so it reads the
        # refined draft, not the first-pass one.
        g = FlowGraph(
            nodes=[
                GraphNode(id="draft", kind="llm"),
                GraphNode(id="gate", kind="gate", depends_on=["draft"],
                          checks=[GraphCheck(term="q")], loop_to="draft"),
                GraphNode(id="finalize", kind="llm", depends_on=["draft"]),
            ],
            output="finalize",
        )
        order = [n.id for n in execution_order(g.nodes)]
        self.assertLess(order.index("gate"), order.index("finalize"))

    def test_fanout_items_empty_list_yields_no_instances(self):
        from workbench.workflow import _fanout_items

        self.assertEqual(_fanout_items("[]"), [])
        self.assertEqual(_fanout_items('{"items": []}'), [])
        # Unparseable prose still falls back to one item (best effort).
        self.assertEqual(_fanout_items("just prose"), ["just prose"])

    def test_render_template_is_single_pass_no_injection(self):
        from workbench.workflow import _render_template

        # 'a' produced text that contains a literal {{b}} — it must NOT expand to b.
        out = _render_template("A: {{a}} B: {{b}}", "IN", {"a": "hello {{b}}", "b": "SECRET"})
        self.assertEqual(out, "A: hello {{b}} B: SECRET")
        # {input} and unknown placeholders.
        self.assertEqual(_render_template("{input} {{missing}}", "X", {}), "X {{missing}}")

    def test_label_matches_is_whole_word(self):
        from workbench.workflow import _label_matches

        self.assertTrue(_label_matches("code", "please write code here"))
        self.assertFalse(_label_matches("code", "please decode this"))
        self.assertFalse(_label_matches("yes", "eyes only"))

    def test_oversized_graph_rejected_without_layout(self):
        from workbench.schemas import FlowGraph, GraphNode
        from workbench.graph import MAX_GRAPH_NODES, validate_graph

        g = FlowGraph(nodes=[GraphNode(id=f"n{i}") for i in range(MAX_GRAPH_NODES + 1)], output="n0")
        errors = validate_graph(g)
        self.assertTrue(any("too large" in e.lower() for e in errors))

    def test_gate_requires_depends_on(self):
        from workbench.schemas import FlowGraph, GraphCheck, GraphNode
        from workbench.graph import validate_graph

        g = FlowGraph(
            nodes=[
                GraphNode(id="refine", kind="llm"),
                GraphNode(id="gate", kind="gate", checks=[GraphCheck(term="q")], loop_to="refine"),
            ],
            output="refine",
        )
        self.assertTrue(any("depends_on" in e for e in validate_graph(g)))

    def test_nested_gate_in_loop_body_rejected(self):
        from workbench.schemas import FlowGraph, GraphCheck, GraphNode
        from workbench.graph import validate_graph

        # Outer gate loops back to 'a'; its body includes inner gate 'g1'.
        g = FlowGraph(
            nodes=[
                GraphNode(id="a", kind="llm"),
                GraphNode(id="g1", kind="gate", depends_on=["a"], checks=[GraphCheck(term="x")], loop_to="a"),
                GraphNode(id="g2", kind="gate", depends_on=["g1"], checks=[GraphCheck(term="y")], loop_to="a"),
            ],
            output="g1",
        )
        self.assertTrue(any("nested gate" in e.lower() for e in validate_graph(g)))

    def test_when_node_must_exist(self):
        from workbench.schemas import FlowGraph, GraphNode
        from workbench.graph import validate_graph

        g = FlowGraph(nodes=[GraphNode(id="a", when_node="ghost")], output="a")
        self.assertTrue(any("ghost" in e for e in validate_graph(g)))

    def test_graph_flow_rejects_invalid_graph(self):
        async def scenario():
            from workbench.schemas import FlowGraph, GraphNode, Profile, RunRequest
            from workbench.workflow import run_workflow

            async def fake(*a, **k):
                return "x"

            graph = FlowGraph(nodes=[GraphNode(id="a", depends_on=["ghost"])], output="a")
            profile = Profile(name="G", workflow="graph", graph=graph, aggregator_model="m")
            with self.assertRaises(ValueError):
                await run_workflow(RunRequest(prompt="x", profile=profile), complete_fn=fake)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()

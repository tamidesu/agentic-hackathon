<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# ROLE: You are a senior AI engineer who has shipped multiple production

tool-using LLM agents, with direct experience choosing between
orchestration frameworks and foundation models under real latency/cost/
reliability constraints — not just prototyping.

CONTEXT: We are building a SINGLE agent (not multi-agent) for a banking
hackathon that: (1) searches PDF documents via retrieval, (2) queries a
structured transaction registry, (3) performs numeric calculations, (4)
runs a self-verification step before answering, (5) outputs a structured
JSON with evidence citations. It runs in a tool-use loop (retrieve →
reason → decide if more evidence needed → retrieve again or answer).
Hard constraint: on evaluation day we get a private dataset and have
about 3 hours total, so the pipeline must be FAST, PREDICTABLE, and
easy to debug under time pressure — not maximally flexible for long-term
maintenance. We are free to use any model or library. Output goes to an
engineer making the final stack decision this week.

PART A — ORCHESTRATION PATTERN

1. Compare, specifically for a single tool-using agent with a
verification loop (not multi-agent): raw ReAct/function-calling loop
hand-coded without a framework, LangGraph, DSPy, and any other actively
maintained 2025-2026 framework worth considering (e.g., Pydantic AI,
simple custom state machine). For each: debuggability/observability,
reliability of structured output, ease of enforcing a step budget
(to prevent runaway loops), learning curve given a tight timeline, and
whether it adds meaningful latency overhead vs. a hand-rolled loop.
2. Is there evidence that a framework materially reduces bugs/increases
reliability for a task this scoped, versus a simple hand-coded loop
with structured function calling? Look for real engineering
postmortems or comparisons, not framework marketing pages.
3. What's the documented best practice for enforcing "the agent must stop
and answer with what it has" (step/tool-call budgets, timeout handling)
in production agent loops, so it doesn't loop indefinitely or run out
of time mid-evaluation?
4. How do production agentic RAG systems typically implement the
"self-verification before answering" step — is it a separate LLM call
acting as judge, a constrained regeneration step, or something else?
Cite specific implementations if available.

PART B — MODEL SELECTION

5. Compare current (as of the last few months) frontier models —
Claude (Opus/Sonnet), Gemini 2.x/3.x, GPT-5-class models, and any
strong open-weight alternative — specifically on: (a) long-context
accuracy/retrieval fidelity (needle-in-haystack or similar benchmarks
on financial/legal-style long documents), (b) faithfulness to source
when citing (hallucination rate on grounded QA benchmarks), (c) tool-
calling reliability, (d) native PDF/table understanding, (e) cost and
latency per call. Cite actual benchmark numbers, not vendor claims.
6. Is there a documented advantage to using a larger/more expensive model
ONLY for the final verification/judge step while using a
cheaper/faster model for the retrieval-reasoning loop? Find any
evidence this "cheap loop + expensive judge" pattern outperforms using
one model throughout.

SOURCE PRIORITY: Engineering blogs with real production experience,
benchmark leaderboards with methodology disclosed, arXiv papers comparing
frameworks/models on agentic tasks, GitHub issues/discussions revealing
real-world framework limitations. Deprioritize framework vendors' own
marketing/docs when making comparative claims — cross-check against
independent sources.

EXCLUDE: general "what is an AI agent" explainers, non-comparative single-
framework tutorials with no evaluation against alternatives.

OUTPUT FORMAT:

- Part A: comparison table (framework/pattern | debuggability |
reliability | step-budget support | overhead | learning curve | verdict
for our constraints) then answered questions with citations.
- Part B: comparison table (model | long-context accuracy | faithfulness/
hallucination rate | tool-calling reliability | native PDF support |
cost | latency) then answered questions with citations.
- End with a labeled "recommendation" paragraph, separate from findings.

For your banking hackathon agent, a lightweight, explicitly coded tool loop with strict budgets and a separate self-verification call is likely safer and faster than a heavy framework, but a type-safe framework like PydanticAI can reduce bug surface if you’re comfortable adopting it quickly.[^1][^2]

***

## Part A — Orchestration Pattern

### Framework comparison table

| framework / pattern | debuggability \& observability | reliability of structured output | step‑budget / loop control | latency / overhead vs hand‑rolled | learning curve (tight timeline) | verdict for our constraints |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Raw ReAct loop (Thought → Action → Observation, hand‑coded) | Very transparent: you see every thought/action/observation in logs; but requires careful stop‑sequence and logging discipline; common failure modes include hallucinated Observations if stop tokens aren’t set correctly.[^3][^4] | Depends entirely on prompt design; no intrinsic JSON/schema enforcement, so malformed tool calls and hallucinated tool responses are common in production unless you add external validation.[^3][^4] | Must be manually enforced with max step counters, loop‑pattern detection (e.g. repeated identical actions), and timeouts; best practice is explicit `max_steps` and a “force final answer” branch when budget is hit.[^3][^2] | Essentially zero framework overhead; just model calls and your own loop; any extra latency comes from your own logging and guardrails.[^3][^4] | Conceptually simple if you know prompt engineering, but easy to shoot yourself in the foot with stop tokens and tool formats; debugging subtle loop bugs takes time.[^3][^4] | Good for small experiments, but for a banking judgeable JSON agent under time pressure, ReAct is more fragile than you need; I’d avoid pure ReAct here. |
| Hand‑coded JSON/tool‑calling loop (OpenAI/Anthropic/Gemini function‑calling, minimal custom state machine) | Debugging is straightforward: you log each model response, parsed tool call, and state transition; observability can be wired into standard app tracing (OpenTelemetry, Datadog, etc.) without framework‑specific tools.[^2] | Closed‑API models now reach ~95–99% tool‑call/schema adherence when you use strict function/tool modes and JSON schema or responseFormat; you can add Pydantic/Marshmallow validation to make failures explicit before execution.[^5][^6][^7] | Trivial to enforce: you keep an integer `step_count`, hard cap per request, and a wall‑clock timeout; when either is exceeded, you skip further tool calls and synthesize a “best‑effort” answer from current state.[^3][^2] | Minimal overhead beyond your own logic; the loop is usually a handful of function calls per turn, so latency is dominated by the LLM and tools, not orchestration.[^2] | Low: you’re just using standard SDK function/tool calling plus a while‑loop; any decent backend engineer can grok and debug it in an afternoon.[^2] | For a single agent with 3–4 tools and self‑verification, this is the sweet spot: predictable, easy to debug during the 3‑hour eval window, and compatible with any cloud model. |
| LangGraph (graph‑based state machine over LangChain) | Excellent structured observability: built‑in state inspection, checkpoint history per node, debug mode showing node order, state diffs, tool calls, retries, and timing; integrates with LangSmith, OpenTelemetry, Langfuse, Prometheus, etc.[^8][^9][^10][^11][^12][^13] | Encourages explicit state models but does not intrinsically enforce JSON schemas; you still rely on model tool‑calling reliability plus your own validation; checkpointing and node‑level logging make it easier to spot malformed outputs.[^8][^11] | Native support for bounded loops via graph edges plus configuration: you can define max loop iterations, use debug vs production logging modes, and run budget checks in nodes; production best practice is explicit iteration caps and selective tracing.[^11][^12][^13][^3] | Some overhead from the graph interpreter, state checkpointing, and tracing; in practice, most reports describe a small per‑step overhead (tens of milliseconds) relative to LLM latency, but debug mode is noticeably slower and is recommended only in staging.[^12][^13] | Medium: you have to learn node/edge semantics, config, and LangSmith/observability tooling; debug mode and logging have their own docs; comfortable for teams already on LangChain, steeper for a fresh hackathon team.[^8][^9][^11][^12] | Great if you anticipate growing into a complex agent graph; for one banking agent and a 3‑hour eval slot, LangGraph may be overkill and slightly harder to debug quickly than a custom loop. |
| DSPy (programming‑not‑prompting framework for LM pipelines) | Provides module‑level structure and automatic tracing/evaluation hooks; the Agentic Leaderboard shows strong reliability scores, but DSPy’s abstractions add an extra layer to inspect compared to raw loops or LangGraph’s visual traces.[^14] | Designed to optimize reliability via declarative specs and compilation; the Agentic Leaderboard reports ~92.4% reliability for DSPy agents, reflecting fewer “logic bugs” across tools and steps than many hand‑rolled systems.[^14] | You can express budgets as constraints in the compiled program, but loop control is indirect; you rely on DSPy’s compilation to respect your limits and may need to inspect generated plans to understand behavior.[^14] | Some overhead from compilation and the DSPy runtime; overall latency impact per task is modest (~500ms reported average latency in Agentic benchmarks), but for tight hackathon SLAs it’s still non‑zero versus a simple loop.[^14] | High for a 3‑hour debugging window: you must learn DSPy’s spec language, compilation, and evaluation stack; reward is cleaner abstractions but not instant productivity unless you’ve used it before.[^14] | Strong for teams already invested in DSPy; for a one‑off banking agent, the overhead and learning curve aren’t justified. |
| PydanticAI (type‑safe Python agent framework) | Very strong: every agent input/output/tool argument is a Pydantic model, with validation errors surfaced clearly; integrates with Pydantic Logfire and OpenTelemetry for structured traces and logging.[^15][^16][^17][^18][^19] | Type‑safety is its core value: outputs and tool calls must conform to Pydantic models, catching schema mismatches early; a comparative engineering blog reports PydanticAI catching 23 development‑time bugs that would have reached production in comparable LangGraph/CrewAI implementations.[^1] | Easy to enforce budgets: you can express max steps/time at the agent level and fail fast with clear exceptions; combined with validated arguments, this sharply reduces runaway loops and side‑effect risks.[^15][^16][^19][^2] | Some runtime overhead from validation and DI, but typically small relative to LLM calls; one comparison found equivalent agents requiring fewer lines of code and lower testing cost in PydanticAI than LangGraph or CrewAI, implying reduced operational friction.[^1] | Moderate if you know Pydantic: it feels like “FastAPI for agents”; if you’re already comfortable with Pydantic models and DI, onboarding is fast, otherwise you have to learn its agent abstractions.[^16][^17][^19][^1] | For a Python team, PydanticAI is an attractive compromise: better reliability and structured output than a bare loop, with manageable complexity; it’s a top contender if you can afford a day of ramp‑up before the hackathon. |
| Simple custom state machine (explicit enum of states over a function‑calling model, no framework) | Highly debuggable: each state is a function, you log transitions and tool calls; easy to inspect traces and reproduce errors using normal application logs or OpenTelemetry.[^2] | Same tool‑calling reliability as the underlying model; you can maximize reliability by using strict JSON tooling and Pydantic/Marshmallow models in your own code, without a full agent framework.[^5][^6][^7] | Budgets are trivial: each transition increments a counter, and you add guards like “if step_count ≥ N or elapsed ≥ T, go to ANSWER state immediately”; state machines make it clear where to place these guards.[^3][^2] | Virtually no overhead beyond your own code; you avoid framework runtime costs while retaining explicit control flow.[^2] | Low: most backend engineers know how to implement finite state machines; the only novelty is the LLM function‑calling integration.[^2] | For this hackathon, a bespoke state machine on top of function calling arguably offers the best balance of predictability, speed, and debuggability. |


***

### 1. Frameworks vs hand‑rolled loop on reliability and observability

**Evidence on where ReAct and naive loops break.** Engineering write‑ups on production ReAct agents highlight structural failure modes: if you don’t set `Observation:` as a stop sequence, the model will happily hallucinate its own observation instead of waiting for the real tool output; repeated actions without progress cause loops; and the loop offers no concept of rollback or side‑effect safety without external guards. Qwen’s official guidance for “thinking” models explicitly warns against stopword‑dependent templates like ReAct, recommending Hermes‑style function calling instead and stressing iteration caps, allowlists, argument validation, and idempotency for side‑effecting tools.[^3][^2][^4]

**LangGraph’s observability benefits.** LangGraph adds graph‑structured state with rich observability: state inspection, checkpoint history for every node run, and debug mode that records node order, state evolution, branch decisions, tool calls, retries, and latency; combined with LangSmith or OpenTelemetry tracing, this gives a much clearer picture of how an agent executed than a plain while‑loop. For multi‑node graphs this materially improves root‑cause analysis and auditing (e.g., “which retrieval node mis‑routed?”), but your hackathon agent has a fairly linear loop, so the incremental benefit is smaller.[^8][^9][^10][^11][^12][^13]

**PydanticAI and type‑safety.** PydanticAI’s main differentiator is type‑safe, validated inputs/outputs/tool calls: every boundary is a Pydantic model, and validation happens at development and runtime, preventing silent schema drift. A comparative blog claims that reference implementations in PydanticAI needed fewer lines of code than LangGraph or CrewAI and caught 23 bugs at dev time that would otherwise have reached production, with lower testing cost, suggesting real reliability benefits for structured workflows.[^15][^16][^17][^19][^1]

**DSPy.** DSPy is evaluated on an agentic leaderboard, scoring 83.6 overall and ~92.4% reliability, meaning its compiled programs reduce certain classes of coordination bugs. However, for your single agent, the extra abstraction layer and compilation step are overhead relative to a direct function‑calling loop.[^14]

**Conclusion for Part A Q1.** For your scope (single agent, few tools, clear loop), the marginal reliability and observability benefits of LangGraph or DSPy are real but not decisive; a simple function‑calling state machine plus strong validation and logging gives most of the value with less complexity. PydanticAI is the one framework that clearly improves structured output reliability without heavy orchestration overhead, if you want stronger guarantees and already like Pydantic.[^16][^19][^2][^15][^1]

***

### 2. Evidence that frameworks materially reduce bugs vs simple loops

You asked specifically for “real engineering postmortems or comparisons.” The clearest data point is the PydanticAI vs LangGraph/CrewAI comparison, which reports:

- Equivalent agents implemented in PydanticAI needed ~160 lines of code vs ~280 in LangGraph and ~420 in CrewAI.[^1]
- PydanticAI caught 23 bugs during development that would otherwise have hit production in those competitor frameworks, and testing cost was ~\$390 vs ~\$1,088 for CrewAI in the benchmark scenario.[^1]

This is not a controlled academic experiment, but it is a concrete engineering study showing a type‑safe framework reducing bug incidence and testing cost relative to other frameworks. It still doesn’t directly compare against a hand‑rolled function‑calling loop, but it suggests that strong, schema‑validated boundaries materially reduce production bugs.

Separately, function‑calling benchmarks show that tool‑calling correctness is far from perfect even for frontier models, with top models achieving ~95–99% correctness on single tools but degrading once you add many tools or steps. This means you *must* add your own validation layer (Pydantic models, JSON schema, range checks) whether you use a framework or not; frameworks like PydanticAI simply codify that pattern.[^5][^20][^7]

For LangGraph, multiple engineering blogs emphasize that checkpoint history and debug mode are what make complex agent graphs “manageable” in production, reducing time‑to‑diagnose and preventing silent cost explosions from loops. That’s evidence of reduced *operational* bugs (e.g., runaway loops, mis‑routed retrieval), but again mostly for multi‑agent, multi‑node setups rather than a single agent like yours.[^9][^11][^12][^13][^8]

***

### 3. Best practices for enforcing “must stop and answer with what it has”

Across production agent guidance and ReAct postmortems, several consistent patterns emerge:

- **Hard step budgets.** Agent foundations explicitly recommend a `max_steps` counter plus loop‑detection logic (e.g., repeated identical actions) for ReAct‑style loops; exceeding the budget should force a terminal “answer with current evidence” action rather than another tool call.[^3]
- **Iteration caps in tool‑calling loops.** Qwen’s agent design article stresses always placing an iteration cap and failing explicitly when it’s exceeded, alongside tool allowlists and argument validation.[^2]
- **Timeouts at infrastructure level.** Observability guidance for LangGraph and OpenTelemetry setups describes treating agent execution as a distributed trace with timeouts and alerts; if a request exceeds a latency threshold, you log and abort rather than let the loop continue.[^11][^13]
- **Forced finalization state.** Agentic pattern write‑ups for RAG describe explicit “fallback handler” nodes or states that generate a safe, partial answer (or a refusal) when verification fails or budgets are exhausted, rather than looping indefinitely.[^21][^22]
- **Guarding high‑risk tools.** Production design docs recommend tool allowlists, idempotency keys for side‑effects, and pre‑execution judge checks for deletion/payment tools; high‑risk tools are never executed after a budget breach.[^23][^2]

For your agent, the pragmatic recipe is:

- Maintain `step_count` and `start_time` in state; set caps like `max_steps = 6` and `max_duration = 8–10s` per query.
- Before each tool call, check both; if exceeded, skip new retrieval and run the “answer from current evidence” generation prompt.
- For the self‑verification/judge step, impose its own short timeout and a single retry; if the judge fails repeatedly, return the best answer with a flag in the JSON indicating “verification incomplete.”

All of these are easy to implement in a hand‑rolled state machine or PydanticAI agent.

***

### 4. How production agentic RAG implements self‑verification

Modern agentic RAG systems almost always implement self‑verification as a **separate LLM call acting as a judge** over the generated answer and retrieved context, rather than a purely self‑reflective regeneration.[^24][^25][^22][^23][^21]

Patterns with concrete examples:

- **LLM‑as‑judge faithfulness gate.**
    - A Zylos 2026 engineering study describes a “RAG faithfulness gate” where the pipeline is: retrieve → generate answer → run a judge model (e.g., Lynx‑8B or MiniCheck) that evaluates whether the answer is entailed by the retrieved documents; if the judge says FAIL, you either retry with stronger grounding or return “insufficient information.”[^23]
    - A Hackernoon article on “Self‑Healing RAG with LangGraph and LLM‑as‑Judge” builds a four‑component layer: retrieval validator (pre‑LLM), grounding verifier (post‑LLM judge over answer + docs), retry orchestrator, and fallback handler; the judge model is deliberately separate from the generator to avoid confirmation bias.[^21]
- **Agentic RAG loops with verification tokens.**
    - Self‑RAG (Asai et al., ICLR 2024) uses special tokens like `[ISSUP]` to indicate whether the generated text is supported by evidence; at inference time these tokens control whether more retrieval is needed or whether to accept the answer. This is a form of intrinsic self‑verification, but it’s trained rather than just prompted.[^25][^24]
- **Judge‑based evaluation frameworks.**
    - RAG‑Critic and similar academic systems formalize LLM‑as‑judge methods that assess outputs against rubrics and retrieved documents, often with dual verification (LLM + human) to construct high‑quality error datasets.[^25]
    - MARCH (Multi‑Agent Reinforced Self‑Check) decomposes an answer into claim‑level units: a Solver generates the answer, a Proposer extracts verifiable QA pairs, and a Checker re‑answers those pairs from retrieved documents only, intentionally denying access to the original answer to reduce confirmation bias; this 8B‑model ensemble significantly improves factuality and grounding scores vs standard RAG.[^26]
- **Industry agentic RAG guidance.**
    - FutureAGI’s 2025 write‑up on agentic RAG recommends a judge call that scores faithfulness or groundedness per sentence and optionally enforces citation IDs; sentences without supporting citations are stripped or rewritten.[^22]
    - Talks and blogs on “LLM‑as‑judge for RAG evaluation” note that nearly all high‑performing teams now use judge LLMs to evaluate whether answers are grounded in retrieved context, both offline and often inline for high‑stakes responses.[^27][^28][^22][^23]

Crucially, research and production experience agree that **intrinsic self‑correction (“check your own work” prompts) without external signals tends to hurt performance**; verification should be grounded in retrieval results, tool outputs, or tests, not just another model reflection.[^23]

For your hackathon agent, the simplest robust pattern is:

1. Retrieve PDFs and transaction records.
2. Generate a draft answer with citations to chunk IDs.
3. Run a small judge model (could even be the same provider but different prompt) that:
    - Checks whether each cited sentence is supported by its referenced chunks, and
    - Flags unsupported claims or missing citations.
4. If flags are minor and budget is nearly exhausted, annotate the JSON response with a “verification_warnings” field; if flags are major and budget allows, trigger one more retrieval/regeneration cycle.

***

## Part B — Model Selection

### Frontier models comparison table (2025–2026 data)

*Note: numbers refer to published benchmarks; they vary by task and configuration, but give directional comparisons for banking‑style long‑document RAG and tool‑using agents.*


| model | long‑context accuracy / retrieval fidelity | faithfulness / hallucination rate (grounded QA / summarization) | tool‑calling reliability | native PDF / table understanding | cost (per 1M tokens, indicative) | latency characteristics |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Claude Sonnet 4.6 | Strong on long‑document legal/contract tasks; earlier Claude 3.5 Sonnet scored ~79.4% on FACTS Grounding and excelled at clause identification in dense legal contracts (e.g., 96.1% vs GPT‑4.1’s 89.3%).[^29][^30][^31] | On a Feb 2026 Vectara long‑document dataset (~7,700 docs), Sonnet 4.6 had ~10.6% hallucination, worse than Gemini 2.5 Flash‑Lite (~3.3%) and GPT‑5‑nano (~3.1%), but still in low double digits; other studies show Claude family leading or near‑leading on document‑grounded tasks in HHEM.[^32][^33][^34][^35] | Independent function‑calling studies report Claude Sonnet achieving ~98–99% accuracy on simple tools and high 90s on parallel/nested schemas with strict JSON validation; BFCL and tool‑use blogs repeatedly place Claude at or near the top for schema adherence, with Sonnet nearly matching Opus at much lower cost.[^5][^20][^7][^6] | Claude does not have a native PDF ingestion API, but combined with vendor or OSS PDF parsing, its text understanding over financial/legal tables is reported to be strong; earlier evaluations found it better than GPT‑4.1 on clause extraction, slightly worse on pure numeric table extraction.[^30][^36][^37] | One comparative guide cites Claude Sonnet at ~\$3 input / \$15 output per 1M tokens, cheaper than Opus but more expensive than Gemini Flash or GPT‑4o Mini.[^7] | Long‑context summarization benchmarks show latency slightly higher than GPT‑4.1: ~11.7s vs 8.2s for 128K tokens; lower‑tier Sonnet variants and streaming mitigate this; for banking PDFs at 50–200 pages, latency is acceptable but not best‑in‑class.[^30] |
| Claude Opus 4.7 (judge candidate) | Designed for heavy reasoning and long‑horizon tasks rather than pure throughput; long‑context performance is strong and used in production for complex engineering and legal workflows.[^38][^39][^40] | A 5‑model 2026 hallucination study reports Opus 4.7 with extended thinking at ~5.1% hallucination across mixed tasks, lower than Gemini 3 Pro and many GPT‑5 variants, and AA‑Omniscience benchmarks show Opus family with substantially lower hallucination than GPT‑5.5.[^38][^41] | Tool‑calling reliability is very high: multiple sources cite Opus >99% on standard tool‑use benchmarks and excellent performance on complex multi‑tool agentic tasks (SWE‑bench Verified ~80.8%, OSWorld ~72.7%), indicating strong multi‑step tool orchestration.[^39][^7][^40] | Same PDF/table story as Sonnet: no native PDF API but strong text/table reasoning once parsed, especially for claim‑level analysis and multi‑hop reasoning.[^30][^37][^42] | Expensive: significantly higher price than Sonnet; one benchmark run cost over \$1,100 vs a few hundred for Sonnet and GPT‑5 medium, and per‑token pricing is comparable to other top‑tier frontier models.[^20][^7] | Latency is higher; Opus trades speed for reasoning quality; acceptable for a *single* judge call per query, but not for multiple iterative loops under tight hackathon SLAs.[^38][^39] |
| Gemini 2.5 Pro | Very strong on long‑context, document‑grounded tasks: FACTS Grounding and Vectara HHEM leaderboards place Gemini variants at or near the top (e.g., ~83.6% on FACTS Grounding for Flash Experimental, ~80% for Gemini 1.5 Pro, and low single‑digit hallucination on HHEM).[^29][^33][^34][^35] | On Vectara long‑document datasets, Gemini 2.5 Pro lands around ~7% hallucination, significantly better than Claude Sonnet 4.6 and some GPT‑5 variants; aggregated benchmarks reinforce Gemini’s strength on grounded summarization and search‑grounded factuality.[^32][^33][^34][^35] | Tool‑calling is solid but slightly behind Claude/GPT at the extremes: BFCL and tool‑use guides report Gemini Pro at ~89–93% function‑calling accuracy, with somewhat worse performance on complex multi‑tool sequences; cross‑MCP coordination benchmarks (MCP‑Atlas) show Gemini 3.1 Pro leading at ~69.2%.[^5][^43][^7][^40] | Native PDF and large‑context support (1M–2M tokens in the 2.x generation) make Gemini particularly suited for huge financial/legal documents; table‑understanding research indicates strong multimodal table reasoning when paired with appropriate encoders.[^29][^44][^36][^37][^42] | Cost is mid‑tier: cheaper than Claude Opus, more than Gemini Flash; one guide cites Gemini Flash at ~\$0.075 input / \$0.30 output per 1M tokens, with Pro at higher but still competitive prices.[^7] | Latency is reasonable even at large context sizes; benchmarks describe Gemini Flash as very fast and Pro as somewhat slower but still acceptable; for long banking PDFs, Gemini offers good throughput with strong accuracy.[^29][^34][^45] |
| Gemini 2.5 Flash‑Lite (loop candidate) | Optimized for speed with good long‑document performance; Vectara’s 2026 long‑doc dataset reports ~3.3% hallucination, the lowest among many frontier models tested (including GPT‑5‑nano and Claude Sonnet 4.6).[^32][^34][^35] | Faithfulness on grounded summarization is excellent; HHEM and related leaderboards place Gemini Flash variants in the ~0.7–3.3% hallucination band on document‑grounded tasks, making them ideal for fast retrieval‑reasoning loops.[^32][^34][^35] | Tool‑calling reliability is good for simple calls but weaker on complex multi‑tool sequences; guidance suggests using Flash for many simple tool calls where speed matters, and reserving more capable Pro/Opus/GPT models for complex orchestration.[^43][^45][^39] | Same native PDF/long‑context story as Pro, but tuned for low latency; well‑suited for scanning long statements and contracts, computing aggregates, and feeding evidence to a judge.[^29][^44][^36][^42] | Very cheap: ~\$0.075 input / \$0.30 output per 1M tokens cited in function‑calling comparisons, making it attractive for high‑volume or iterative agent loops.[^7] | Latency is among the best; designed as a “Flash” tier model, one of the fastest options for long‑context RAG loops, which is ideal for your retrieve‑reason‑verify pattern.[^29][^45] |
| GPT‑5‑class (e.g., GPT‑5 / GPT‑5.2 / GPT‑5 Pro) | Strong all‑rounder; GPT‑5 models lead or near‑lead on many agentic and tool‑calling benchmarks, with solid long‑context reasoning; however, on certain grounded tasks (FACTS, long‑doc HHEM) they don’t uniformly beat Gemini or Claude.[^33][^29][^43][^40] | Vectara and other studies show GPT‑5 hallucination rates in low single digits for some document‑grounded tasks but >10% on harder multi‑dimensional factuality tests; aggregated analyses note GPT‑5 doing very well on grounded summarization but not uniquely dominant.[^33][^34][^35][^38] | Tool‑calling reliability is excellent: TAU2‑Bench reports GPT‑5.2 at ~98.7% multi‑turn tool‑calling accuracy, best among evaluated models; BFCL and other guides describe OpenAI’s function‑calling as mature and schema‑friendly.[^5][^20][^43][^7][^40] | PDF/table understanding is good via OpenAI’s multimodal stack; for structured financial tables, GPT‑4.1 previously outperformed Claude on numeric extraction, and GPT‑5 builds on that lineage.[^30][^36][^37] | Pricing is competitive but not the cheapest; GPT‑4o‑class models already had mid‑tier pricing; GPT‑5‑class is higher but offset by strong performance; exact numbers depend on tier (mini vs pro).[^20][^7] | Latency is moderate; reasoning‑enabled GPT‑5 variants can be slower; TAU2‑Bench shows substantial average agent time (~hundreds of seconds) in certain test setups, but for your use, shorter context and fast tiers are available.[^20][^43] |
| Strong open‑weight (e.g., Llama 3.1 405B Instruct) | Good long‑context reasoning but typically below frontier closed models on fine‑grained factuality benchmarks; long‑table and cross‑format table benchmarks show open VLMs making progress but still trailing best closed models.[^29][^37][^46][^42] | Hallucination and faithfulness rates vary widely; some open models have low hallucination on specific tasks but higher on others; AA‑Omniscience and other leaderboards often show higher error rates than top closed models.[^29][^38] | Recent leaderboards rank Llama 3.1 405B and some others as top tool‑calling models overall, but open models generally trail closed ones by 5–22 percentage points on multi‑tool and multi‑turn function‑calling benchmarks, especially as tool count grows.[^5][^47] | Many open models lack native PDF ingestion; you must build your own parsing and chunking pipeline; table understanding is decent but often requires fine‑tuning or custom prompts.[^36][^37][^42] | You pay infra cost rather than API; for a hackathon, self‑hosting a 405B model is impractical and expensive in time, even if raw token cost is “free.” | Latency and ops overhead depend on your hardware and serving stack; realistically not viable to stand up and debug within a 3‑hour evaluation window. |


***

### 5. Interpreting benchmarks for your banking agent

For a banking hackathon agent that:

- Reads long PDFs (terms, statements, product docs).
- Queries structured transaction tables.
- Must return grounded answers with citations and moderate numeric reasoning.
- Runs a self‑verification/judge step.

Key signal from the benchmarks:

- **Long‑context fidelity and ground truth:** FACTS Grounding and Vectara HHEM show Gemini Flash/Pro and Claude Sonnet as especially strong on document‑grounded summarization and long‑context factuality, with Gemini Flash variants leading raw hallucination rates on long documents (~0.7–3.3%) and Claude Sonnet scoring high on clause‑level legal tasks.[^32][^33][^34][^29][^30]
- **Faithfulness/hallucinations on grounded QA:** All frontier models still hallucinate at non‑trivial rates (3–15%+) on grounded tasks, but Gemini 2.x and Claude Opus have some of the lowest rates in recent evaluations; GPT‑5‑class models perform very well on some tasks but not consistently above Gemini or Claude.[^33][^34][^35][^41][^38]
- **Tool‑calling reliability:** Tool‑calling benchmarks (BFCL, TAU2‑Bench, TAU‑bench) place Claude Opus/Sonnet, GPT‑5‑class, and Gemini 3.x all in high‑90% ranges for simple tools, with GPT‑5‑class shining on multi‑turn tool sequences (~98.7% on TAU2‑Bench) and Claude/Opus leading complex agentic tasks (SWE‑bench, OSWorld). For a small tool set (PDF retriever, SQL/transaction query, calculator, judge), all three families are viable.[^20][^43][^7][^40][^5]
- **PDF/table understanding:** There are cross‑format and long‑table benchmarks (TableEval, LongTableBench, TABVERSE, RealHiTBench) showing that table understanding is non‑trivial, but specific per‑model scores are sparse; practical blog comparisons suggest Claude is strong for legal clauses, GPT for numeric tables, and Gemini for very long multimodal documents.[^48][^30][^36][^37][^46][^42]
- **Cost and latency:** Claude Sonnet is mid‑priced (~\$3 / \$15 per 1M tokens) with strong reliability; Gemini Flash is very cheap (~\$0.075 / \$0.30 per 1M tokens) and extremely fast, making it attractive for iterative loops; Opus and GPT‑5 Pro are expensive but acceptable for a single judge call per query.[^45][^7][^20]

Given these, the most hackathon‑friendly combinations are:

- **Fast loop model:** Gemini 2.5 Flash‑Lite or Claude Sonnet 4.6 — fast, relatively cheap, good long‑document faithfulness and function calling.[^34][^35][^7][^32][^45]
- **Judge model:** Claude Opus 4.7 or GPT‑5 Pro — expensive but strong as a final verifier over answer + evidence, used sparingly.[^38][^39][^40]

***

### 6. “Cheap loop + expensive judge” pattern — evidence and trade‑offs

Many production teams now use a **“cheap actor + expensive judge”** pattern: a fast model for the retrieval/reasoning loop and a larger, more reliable model for final verification and gating.[^26][^22][^21][^23]

Evidence and examples:

- **Zylos LLM‑as‑judge study.** This article notes that more than half of surveyed production agent teams use judge LLMs at runtime for hallucination defense and tool‑call verification, often choosing smaller, cheaper judge models for lightweight checks (Lynx‑8B, MiniCheck) and larger ones for periodic deep audits; it emphasizes that intrinsic self‑correction without external signals degrades performance, whereas judge‑based verification improves reliability.[^23]
- **Self‑Healing RAG with LangGraph.** The architecture explicitly separates the generation model and judge model, arguing that using the same model to judge its own output is unreliable; the judge is a different LLM that checks grounding and can trigger retries or fallbacks, acting as a final gate; this is effectively “actor + judge,” sometimes with the judge having different capabilities/budget.[^21]
- **Agentic RAG guidance.** FutureAGI’s write‑up describes a judge call that scores faithfulness, sometimes with a faster “flash” judge for online screening and a slower “small” judge for deeper scoring; they report that adding this judge layer reduces hallucination versus classic one‑shot RAG, with acceptable cost (1–2 extra evals per turn).[^22]
- **MARCH and related multi‑agent frameworks.** MARCH shows that adding dedicated Proposer/Checker agents to verify claim‑level units (even when using relatively small models) substantially improves factuality and grounding; an 8B solver + checker ensemble outperforms standard RAG with larger single models on multiple benchmarks. This is a more sophisticated “cheap-ish actor + separate verifier” variant.[^26]
- **Academic consensus on self‑correction.** Surveys of self‑correction methods conclude that unguided self‑critique using the same model often hurts performance, but frameworks that incorporate external feedback (retrieval, tests) and separation of actor and judge improve reliability; this motivates using a potentially more powerful judge only when needed.[^49][^25][^23]

What’s less clear is a **formal comparison of “one large model throughout” vs “smaller loop + larger judge”** under fixed cost. Most studies compare adding a judge to RAG vs not adding one, or using specialized judges vs generic ones, rather than varying actor size. Still, practical reports show:

- Using a fast model in the loop and a stronger judge once per query keeps latency and cost manageable, while sharply reducing hallucinations compared to no judge.[^22][^21][^23]
- For your case (moderate complexity, banking documents, limited evaluation time), the cost of one Opus/GPT‑5 judge call per query is acceptable, and the benefits (better groundedness, fewer silent hallucinations) are valuable.

So while there isn’t a definitive “cheap loop + expensive judge beats large‑only actor” benchmark, multiple engineering case studies support **actor + judge** architectures, with the judge often being more capable than the loop model and invoked sparingly.[^21][^26][^22][^23]

***

## Recommendation

For this hackathon, I’d recommend **a hand‑coded function‑calling loop implemented as a simple Python state machine, with strict step/time budgets and Pydantic validation**, running on **a fast, low‑cost long‑context model like Gemini 2.5 Flash‑Lite or Claude Sonnet 4.6 for the main retrieval/reasoning, and a single call to a stronger judge model (Claude Opus 4.7 or GPT‑5‑class) to verify grounding before answering.** This stack keeps orchestration simple and debuggable under a 3‑hour window, leverages high tool‑calling reliability and long‑document performance where they matter, and uses the expensive model only as a final gate, aligning with modern agentic RAG best practices without committing you to heavy frameworks.[^39][^7][^2][^32][^34][^38][^5]
<span style="display:none">[^50][^51][^52][^53][^54][^55][^56][^57][^58][^59][^60]</span>

<div align="center">⁂</div>

[^1]: https://agentmarketcap.ai/blog/2026/04/06/pydanticai-python-agent-framework-langgraph-crewai-comparison

[^2]: https://tomodahinata.com/en/blog/qwen3-agent-tool-use-function-calling-qwen-agent-production

[^3]: https://ac.fzhiy.net/agent-post-training-playbook/cheatsheet-agent-foundations-en.html

[^4]: https://www.agentengineering.io/topics/articles/react-loop-unpacked

[^5]: https://presenc.ai/research/ai-agent-tool-calling-accuracy-benchmarks-2026

[^6]: https://opncrafter.space/guide/benchmarking-agent-tools

[^7]: https://deploybase.ai/articles/best-llm-for-function-calling-tool-use-comparison

[^8]: https://www.educative.io/courses/langgraph-from-langchain-user-to-agent-builder/debugging-and-observability

[^9]: https://svgoudar.github.io/Learn-LangGraph/langgraph/11-observability-operations/overview.html

[^10]: https://docs.langchain.com/oss/python/langgraph/observability

[^11]: https://svgoudar.github.io/Learn-LangGraph/langgraph/11-observability-operations/00-logging.html

[^12]: https://svgoudar.github.io/Learn-LangGraph/langgraph/11-observability-operations/05-debug-mode.html

[^13]: https://www.linkedin.com/pulse/langgraph-production-why-observability-evaluation-decide-braga-kor5f

[^14]: https://theagenticleaderboard.com/agent/dspy/

[^15]: https://www.open-source-ai.tech/projects/pydantic-ai

[^16]: https://agentwiki.org/pydantic_ai

[^17]: https://github.com/pydantic/pydantic-ai

[^18]: https://github.com/pydantic/pydantic-ai/blob/main/AGENTS.md

[^19]: https://vantaige.io/ai-tool/pydantic-ai

[^20]: https://www.klavis.ai/blog/function-calling-and-agentic-ai-in-2025-what-the-latest-benchmarks-tell-us-about-model-performance

[^21]: https://hackernoon.com/building-a-self-healing-rag-pipeline-with-langgraph-langchain-and-llm-as-judge

[^22]: https://futureagi.com/blog/agentic-rag-systems-2025/

[^23]: https://zylos.ai/research/2026-04-10-llm-as-judge-production-agent-verification-2026/

[^24]: https://letsdatascience.com/blog/agentic-rag-self-correcting-retrieval

[^25]: https://aclanthology.org/2025.acl-long.179.pdf

[^26]: https://www.emergentmind.com/topics/multi-agent-reinforced-self-check-for-hallucination-march

[^27]: https://www.youtube.com/watch?v=C3P-lnddsRI

[^28]: https://www.youtube.com/watch?v=MZ4mIRQsAhE

[^29]: https://awesomeagents.ai/leaderboards/hallucination-benchmarks-leaderboard/

[^30]: https://www.holysheep.ai/articles/en-gpt-41-vs-claude-35-sonnet-long-context-summarizat-2026-04-12-0025.html

[^31]: https://www.anthropic.com/news/claude-sonnet-4-5

[^32]: https://dikehomme.com/gpt-claude-gemini-hallucination-comparison-en/

[^33]: https://aipromptshub.co/safety/llm-hallucination-rates-comparison

[^34]: https://www.mayhemcode.com/2026/04/vectara-hallucination-leaderboard.html

[^35]: https://chatgptguide.ai/ai-hallucination-rates-report-gpt-claude-gemini/

[^36]: https://aclanthology.org/2025.trl-1.10.pdf

[^37]: https://arxiv.org/pdf/2606.09578.pdf

[^38]: https://www.digitalapplied.com/blog/ai-model-hallucination-rate-benchmarks-2026-study

[^39]: https://www.mindstudio.ai/blog/gemini-3-5-flash-vs-claude-opus-vs-gpt-5-5

[^40]: https://mcpplaygroundonline.com/blog/best-ai-model-for-mcp-tool-calling

[^41]: https://suprmind.ai/hub/claude/vs-other-ai/

[^42]: https://papers.nips.cc/paper_files/paper/2025/file/2a0a3449c661a33ae093b0825705e150-Paper-Conference.pdf

[^43]: https://fleeceai.app/blog/best-ai-model-for-tool-calling-2026

[^44]: https://thecraftman.medium.com/claude-vs-chatgpt-vs-gemini-for-building-ai-agents-in-2026-78515e62611e

[^45]: https://fp8.co/articles/Gemini-3.5-Flash-vs-Claude-Sonnet-vs-GPT-4.1-Mini-Speed-Model-Comparison

[^46]: https://aclanthology.org/2025.findings-acl.371.pdf

[^47]: https://llm-stats.com/leaderboards/best-ai-for-tool-calling

[^48]: https://soloaikit.com/claude-ai-hallucination-rate-vs-competitors-tested/

[^49]: https://openreview.net/pdf?id=AXNRILww9c

[^50]: https://dev.to/kuldeep_paul/how-to-implement-observability-for-ai-agents-with-langgraph-openai-agents-and-crew-ai-5e7k

[^51]: https://www.reddit.com/r/LangChain/comments/1p6rna2/how_do_you_actually_debug_complex_langgraph/

[^52]: https://community.openai.com/t/function-calling-looping-uncontrollably-and-calling-unnecessarily/931730

[^53]: https://www.reddit.com/r/LangChain/comments/150b3id/react_planning_loop_with_function_calling/

[^54]: https://pub.towardsai.net/beyond-the-demo-building-a-rag-system-from-scratch-that-routes-retrieves-and-evaluates-itself-4bb1dc66e524

[^55]: https://www.digitbin.com/chatgpt-claude-gemini-accuracy/

[^56]: https://silentroom.media/the-machine/fact-checking-ceiling-claude-vs-chatgpt-vs-gemini-accuracy

[^57]: https://dikehomme.com/gpt-claude-gemini-hallucination-comparison/

[^58]: https://www.rfeonline.com.au/insights/applied-intelligence/ai-model-hallucination-rates-comparison

[^59]: https://note.com/soyokaze2/n/n87f692d53b7e?hl=en

[^60]: https://www.arsturn.com/blog/claude-sonnet-4-tool-calling-vs-gpt-4-gemini-a-deep-dive


<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

## What actually exists vs. what is mostly unsolved/DIY

There is real work on temporal RAG and conflict-aware retrieval, but it is still early and mostly research-grade or bespoke implementations rather than off‑the‑shelf libraries you can drop into a hackathon project. Legal/compliance vendors solve this with proprietary knowledge-graph pipelines and date/lineage extraction, with a few technical case studies but limited open implementations. Benchmarks for “current fact vs outdated fact” exist, but they are mostly general-knowledge or legal-change detection datasets rather than banking-policy QA corpora.[^1][^2][^3][^4][^5][^6][^7][^8][^9][^10][^11]

For your use case (bank PDFs, conflicting tariffs, internal orders), you will likely have to assemble a DIY stack: robust OCR + metadata/date extraction + a simple lineage graph + temporal filters + a conflict-aware answer layer, borrowing ideas from MemStrata, AionRAG, ConflictRAG, and legal Graph RAG, but implementing them pragmatically for your corpus.[^3][^12][^5][^8][^9][^1]

***

## 1. Techniques for conflicting document version resolution in RAG

### MemStrata: deterministic supersession over a bi‑temporal ledger

MemStrata introduces a retrieval memory that explicitly models temporal validity by storing facts as $(subject, relation, object)$ triples with validity intervals and applying deterministic supersession rules when a fact is contradicted. When a new fact with same $(subject, relation)$ but different object arrives (e.g., “ATM withdrawal limit is 500,000 KZT” vs later “ATM withdrawal limit is 1,000,000 KZT”), the older value is retired in a bi‑temporal ledger (transaction and validity time), so only the currently valid triple is returned to RAG. This eliminates “stale‑fact errors” that standard RAG cannot avoid (RAG served superseded values 15–40% of the time; MemStrata drives this to ~0%).[^1][^3]

**Implementation idea for hackathon:**
Treat each “atomic fact” (e.g., per card type + channel + limit) as a key; when ingestion sees a newer conflicting value, mark the older triple as superseded (valid_to = new_fact.valid_from) and make retrieval filter by validity at “now”. You can do this in a small Postgres table or graph DB rather than a full ledger.

### AionRAG: time‑correct retrieval control

AionRAG treats retrieval as a control problem under knowledge drift. It decides:[^5]

- whether retrieval is needed at all;
- what evidence window (time interval) and hop depth to use;
- filters candidates by time before semantic ranking to avoid mixing historical versions of the same claim;
- then applies a “conflict‑gated” decoding rule when retrieved evidence disagrees with the model prior.[^5]

For banking policies, the key idea is to make “time window selection + time filtering” a first‑class step before embedding similarity, e.g., filter out chunks whose effective interval doesn’t cover the query date (default: today) before scoring semantic relevance.[^5]

### ConflictRAG: explicit conflict types and resolution strategies

ConflictRAG defines a pipeline that:

1. Detects conflicts between retrieved documents using an MLP classifier over embeddings plus selective LLM refinement.[^13]
2. Classifies conflict types: factual, temporal, opinion.[^14][^13]
3. Resolves conflicts with type‑specific strategies. For temporal conflicts, it ranks documents by recency (metadata or LLM‑extracted dates) and prioritizes the latest source while noting temporal evolution.[^13][^14]
4. Uses Entropy‑TOPSIS multi‑criteria decision making (authority, recency, relevance, specificity, consistency) to pick a “winner” source when conflicts are not purely temporal.[^14][^13]

You can adapt this by building a small conflict‑detection step over retrieved chunks (e.g., “two chunks answer the same question with different numbers”), then resolving via date + authority rules and emitting an explanation JSON alongside the answer.

### Prompt‑level temporal conflict resolution

Inferensys publishes a “Temporal Conflict Resolution Prompt for RAG Answers” that defines a structured workflow inside the LLM: extract claims and dates, detect temporal conflicts, resolve them via recency/authority rules, and produce a separate “deprecated information” section. It enforces rules like “never mark a source outdated solely because it is old; only when explicit contradictory evidence exists” and ensures conflicts are either resolved or flagged as unresolved, with reasons.[^15]

This is low‑effort to adopt: you can wrap your generation step in such a prompt so the agent is forced to (a) list conflicting claims, (b) state why one supersedes another (e.g., effective date, explicit ‘supersedes’ clause), and (c) show deprecated values.

### CSRAG: conflict‑suppressed RAG for parametric vs retrieved knowledge

Conflict‑Suppressed RAG (CSRAG) focuses on conflicts between the LLM’s internal parametric knowledge and external context rather than between documents, but the mechanics are relevant. It:[^16]

- Extracts parametric facts relevant to the query.
- Paraphrases retrieved context into diverse statements.
- Uses two logits processors during decoding: a ConflictingKnowledgeSuppressor to damp tokens associated with conflicting parametric facts, and a ContextualBoostProcessor to amplify tokens grounded in retrieved evidence.[^16]

You can borrow this idea for document‑vs‑document conflicts if you paraphrase the selected “current” facts and boost them in decoding while suppressing tokens associated with deprecated facts.

### Reasoning‑Trace‑Augmented RAG

A BITS Pilani + CMU framework (described in a technical LinkedIn write‑up) forces the model to generate a structured XML “thinking block” before answering.[^17]

Pipeline:

1. Micro‑level reasoning: analyze each retrieved document, assign verdict (supports, partially supports, irrelevant), extract key fact, check source quality and date.[^17]
2. Macro‑level conflict analysis: classify conflicts (temporal, opinion, complementary) and explain relationships (e.g., “Doc A older than Doc B; dates indicate an update”).[^17]
3. Grounded synthesis: choose strategy based on conflict type (temporal → prioritize more recent; opinion → present perspectives; complementary → merge).[^17]

This is essentially a structured adjudication step. For a hackathon, you can approximate it with a fixed JSON schema and a few chain‑of‑thought prompts rather than model fine‑tuning.

### Temporal knowledge conflict resolution benchmarks

The ACL 2026 “Temporal Knowledge Conflict Resolution in LLMs” paper introduces WIKIRECENTCHANGES, a benchmark with stable and recently updated facts from Wikidata, used to study how LLMs detect and resolve conflicts between outdated and current information. It shows that LLMs can verbalize temporal reasoning but often fail to act on it, reinforcing the need for explicit conflict‑resolution pipelines rather than relying on the model’s “implicit” recency judgment.[^2]

***

## 2. Legal‑tech / compliance‑tech RAG systems in production

### Regulatory context graphs / Graph RAG for legal norms

Carver’s engineering blog describes a “regulatory context graph” where nodes represent regulators, statutes, rules, guidance, enforcement actions, obligations, products, jurisdictions, and dates; edges represent relations like *amends*, *supersedes*, *clarifies*, *effective_on*, *applies_to*. Every claim and edge has explicit provenance (source artifact, paragraph, version), and time is encoded via validity windows on edges and node attributes, enabling “as‑of” queries.[^6]

Their pipeline:

- Continuous ingestion of regulatory notices, circulars, FAQs, etc., with canonical metadata (publisher, jurisdiction, publication timestamp, identifiers).[^6]
- Parsing and segmentation into sections/paragraphs/claims.[^6]
- Entity extraction + relationship extraction (explicit citations + inferred relationships) labeled with confidence and provenance.[^6]
- Temporal annotation (published date, effective date, compliance deadlines, retroactivity flags).[^6]

Queries such as “what is current capital adequacy requirement for X as of 2024‑01‑01?” traverse the graph and select claims whose effective interval covers the query date and whose supersession chain indicates they are the latest applicable version.[^6]

The SAT‑Graph RAG / Graph RAG for Legal Norms paper generalizes this: it models legal norms as abstract “Works” with versioned “Expressions” and Component Temporal Versions (CTV) that represent the text of each component at a point in time. Legislative changes are modeled as Action nodes that terminate validity of a prior CTV and produce a new one; point‑in‑time retrieval deterministically selects the CTV whose validity interval satisfies $tv.valid\_start \le t < tv.valid\_end$, with policies like SnapshotLast for multi‑change years.[^8][^18][^9][^19]

In production GraphRAG setups, the RAG agent doesn’t retrieve raw PDFs; it retrieves the subgraph (norm + components + CTVs + Actions) relevant to the query date and then asks the LLM to synthesize an answer with explicit provenance and temporal policy disclosure.[^9][^19][^8]

### Knowledge‑graph driven regulatory compliance engine

A meta‑intelligence case study describes a four‑layer architecture for a multilingual financial regulatory compliance engine:[^7]

1. **Regulatory Extraction Layer:** crawls regulatory originals from official websites and gazettes, building a versioned corpus.[^7]
2. **Semantic Parsing Layer:** uses pretrained models to parse provisions into entities and relationships, extracting article numbers, obligations, penalties, and semantic relations.[^7]
3. **Knowledge Graph Layer:** constructs a graph where each provision node includes attributes like original text, structured summary, scope of application, effective date, amendment history, and relationships such as amendment, supersession, supplementation, conflict.[^7]
4. **Application Service Layer:** implements change notifications, impact assessment, and compliance gap analysis.[^7]

When new regulatory documents arrive, an incremental update mechanism extracts changed provisions and updates the graph by adding nodes/edges, modifying attributes, and marking invalidated nodes/relationships. This is essentially production‑grade supersession logic: the current valid state for an obligation is derived from graph traversal, not from raw retrieval.[^7]

### Temporal metadata and hybrid retrieval for legal RAG

A legal RAG whitepaper (short PDF) evaluates hybrid retrieval plus metadata filtering in legal QA. It shows that simple metadata (e.g., validity windows, publication dates) plus hybrid retrieval (lexical + semantic) reduces temporal conflicts by over 50% compared to naive RAG and that “having correct metadata brings the most to the table” versus complex generation reasoning. When metadata is missing or incorrect, aggressive filtering can hurt recall, highlighting the importance of robust metadata extraction.[^20]

### Commercial GRC platforms

Regology’s “Smart Law Library” uses causal citations to tie amendments or proposed rules to specific sections of laws; when a bill or rule updates a section, the library generates alerts and impact views based on citation analysis. Archer’s benchmarking of regulatory date extraction demonstrates that a purpose‑built system can reach 95% correctness on publication, effective, and comment‑close dates across multiple jurisdictions compared to ~44% for a generic LLM workflow, implying a dedicated date‑extraction and normalization pipeline.[^21][^22]

These systems generally:

- Treat “effective date” and “superseded by” as first‑class structured attributes.
- Use rule‑based + ML/LLM extraction from headers, footers, front matter, and specific phrase patterns (“shall come into force on”, “effective as of”).
- Maintain lineage graphs linking new documents to the provisions they affect, enabling version‑aware retrieval for compliance dashboards.

***

## 3. Extracting effective dates, version dates, supersession from messy text/metadata

### Layered extraction pipelines for contracts and legal docs

Contract extraction blogs outline practical pipelines:

- **Schema definition:** explicitly define fields like effective_date, expiry_date, notice_period_days, auto_renewal, contract_value, etc., often via a Pydantic model.[^23][^24]
- **Rules‑based extraction:** use regex, layout‑aware NER, and header/footer parsing for tractable fields (party names in recitals, governing law, execution dates in signature blocks).[^23]
- **LLM extraction for variable fields:** ask an LLM to populate specific schema fields from clause text (renewal, notice periods, payment terms), returning structured JSON mapped to the schema (with None when absent, not hallucinated values).[^25][^23]
- **Validation layer:** enforce logical constraints: effective date precedes expiry; notice periods are plausible; contract value consistent with payment terms; flags low‑confidence fields for human review.[^23]

Local LLM evaluations show ~90% accuracy for effective dates and termination dates in standard templates, with 5–10% misses when dates are buried in complex conditionals or unusual layouts.[^25][^23]

For banking PDFs:

- Extract candidate dates from:
    - Title (“Tariff Sheet No. 3 (Effective from 01.03.2024)”).
    - Preamble (“This Order shall take effect on…”).
    - Signature blocks.
    - Footers and stamps (“Approved on …”, “Enters into force …”).
- Normalize relative dates: e.g., “within 30 days of publication” can be turned into a computed effective window if publication date is known.


### OCR and layout‑aware parsing

Legal document processing platforms (Unstract, DocPeel, Airparser, Nutrient, LegalDocPro) emphasize OCR and layout‑aware parsing as prerequisites: they convert PDFs (including scans) into structured text while preserving headers, footers, tables, and key‑value regions. This enables:[^26][^27][^28][^29][^30]

- Targeted extraction from specific regions (header region for publication date; footer for version numbers).
- Extraction with coordinates, enabling auditing and UI highlighting (e.g., “Key dates: effective, termination, renewal windows, notice periods”, with source coordinates).[^27]


### LLM‑based metadata extraction

A recent metadata extraction paper for contract review uses LLMs as the final component after robust text conversion and chunk selection, generating structured JSON with extracted metadata. They find:[^31]

- Text conversion quality (e.g., Azure Document Intelligence) strongly affects extraction.
- A NER‑enhanced Borda re‑ranking method improves chunk selection for metadata extraction.
- Chain‑of‑Thought prompting and structured tool calling improve extraction accuracy for fields like effective date and jurisdiction.[^31]

Failure modes include hallucinated numeric values when information is not explicitly present, incorrect date type (approval vs effective vs signature), and missed dates in unusual positions; production systems mitigate this with strict schema constraints and cross‑field validation.[^31][^23]

### LLM inference of recency from content

When explicit dates are missing, systems sometimes infer recency from content (e.g., references to more recent laws, product names, or organizational structures), but research shows this is unreliable. Papers on temporal misalignment and conflict resolution show that LLMs often verbalize temporal change (“this may have changed”) but still choose outdated answers unless supported by explicit temporal metadata or retrieval constraints. For a hackathon system, you should treat LLM‑only recency inference as a last resort and always surface uncertainty.[^4][^2]

***

## 4. Document lineage graph vs chunk‑level recency filtering

### Document lineage / context graph approach

Regulatory context graphs and GraphRAG treat legal/regulatory content as a graph of entities and temporally versioned components, with explicit lineage:[^18][^19][^8][^9][^6]

- Nodes: norms, sections, paragraphs, provisions, actions (amend, repeal), effective dates.
- Edges: amends, supersedes, clarifies, references, applies_to, effective_on, repeals.
- Temporal versions (CTV): each component has one or more CTVs, each valid over an interval; actions terminate one CTV and create another.[^8][^18]

Query planning:

1. Canonicalize temporal constraint (e.g., “as of 2022‑05‑01” or default now).
2. Resolve scope (which norm/component applies).
3. Traverse CTV chain to select deterministic version whose validity interval matches the query; use policies like SnapshotLast when multiple changes occur in a window.[^8]
4. Retrieve text units only from that version for RAG; optionally reconstruct provenance DAG of actions that led to the current state.[^8]

Advantages:

- Handles implied amendments via citation graphs: if Circular C references Order B §3 and changes a clause, the graph encodes that; queries automatically follow the chain.[^8][^6]
- Supports as‑of queries naturally: historical states are preserved; “current truth” is not overwritten.
- Provides auditable evidence: lineage chains can be surfaced to explain why a given policy is considered current.


### Chunk‑level recency filtering

Simpler RAG systems perform:

- Metadata‑based recency scoring: rank chunks by publication date or effective date and pick the newest.[^12][^20][^13]
- Temporal filtering: restrict retrieval to chunks within a time window, then do vector similarity.[^20][^5]
- Conflict detection and LLM‑based resolution when multiple chunks disagree (e.g., prefer newer ones).[^15][^12][^13]

While this significantly reduces stale answers when metadata is correct, it fails when:

- Amendment relationships are implicit (new doc mentions old one only in prose).
- Different documents address overlapping but not identical scopes (e.g., one covers platinum cards, another retail cards; naive “newest wins” may misapply limits).
- Effective dates differ from publication dates (e.g., future‑effective rules).

Graph‑based lineage is more robust but heavier to implement; for a hackathon, you can approximate it with:

- A per‑topic table keyed by a canonical identifier (e.g., “card_type + operation + channel”) that stores current and prior values, with effective_from/to and source document IDs.
- Simple edges: supersedes_document_id, amended_fields JSON.

This gives you fact‑level lineage without building a full legal ontology.

***

## 5. Targeted searches: what’s novel vs redundant

### “temporal RAG”

This surfaces:

- MemStrata: bi‑temporal ledger and deterministic supersession, focusing on fact evolution and stale‑fact errors.[^3][^1]
- AionRAG: time‑correct retrieval control, with calibrated decision thresholds and conflict‑gated decoding.[^5]
- EMNLP “Mitigating Temporal Misalignment by Discarding Outdated Facts”: predicting fact duration and discarding outdated evidence to reduce misalignment between model training date, evidence date, and query date.[^4]

Novel: MemStrata’s ledger and AionRAG’s control‑based retrieval plus conflict‑gated decoding; both propose concrete architectures you can mimic. Redundant: generic advice to “add timestamps to metadata” and “prefer newer documents” as seen in many RAG blog posts.[^12]

### “conflicting evidence resolution RAG”

This finds:

- ConflictRAG: explicit conflict detection, type classification, Entropy‑TOPSIS for source credibility, and conflict‑aware RAG score (CARS).[^13][^14]
- CSRAG: decoding‑time dual logits processors to suppress conflicting parametric knowledge and boost contextual evidence.[^16]
- Reasoning‑Trace‑Augmented RAG: structured XML reasoning traces for evidence adjudication.[^17]

Novel: explicit conflict types and metrics (CARS), multi‑criteria source scoring, decoding‑time interventions, reasoning‑trace supervision. Redundant: generic prompts like “if sources disagree, choose the most trustworthy”.

### “document supersession retrieval”

This mostly leads to legal GraphRAG and regulatory context graphs: the ontology‑driven SAT‑Graph RAG and Graph RAG for Legal Norms, plus temporal semantics blogs on GraphRAG. Novel: detailed graph models with Temporal Versions, Actions, and deterministic point‑in‑time retrieval; these go beyond flat vector stores.[^32][^19][^18][^9][^8]

### “regulatory change tracking LLM” / “policy version conflict resolution AI”

This yields:

- RegTrack: benchmark for multi‑class legal change detection on EU regulations (atomic legal units, six change types).[^10]
- LawShift: legal judgment prediction benchmark under statutory revisions with 31 revision types, plus code and dataset on HuggingFace.[^11]
- Knowledge‑graph compliance case study (meta‑intelligence).[^7]
- Regology Smart Law Library and Archer’s date extraction benchmark (production change‑tracking and date extraction).[^22][^21]

Novel: RegTrack and LawShift as structured change‑detection and revision‑aware benchmarks; meta‑intelligence and context‑graph blogs as real engineering write‑ups. Redundant: marketing pages that say “we use AI to track regulatory changes” without technical detail.

### “policy version conflict resolution AI”

Mostly overlaps with ConflictRAG, MemStrata, and general conflict‑aware RAG discussions. Novel content is in the research papers; web listicles tend to add little.[^1][^3][^12][^13]

***

## 6. Benchmarks/datasets for choosing the correct current fact

There are several relevant benchmarks, but none are a perfect out‑of‑the‑box fit for “banking tariff sheets with conflicting PDFs.”

- **WIKIRECENTCHANGES (ACL 2026 temporal conflict paper):**
    - Contains stable and recently changed facts from Wikidata.
    - Designed to study how LLMs resolve conflicts between outdated benchmark labels and up‑to‑date real‑world facts.[^2]
    - Evaluates temporal reasoning and mutability awareness but is not specifically RAG + multi‑document, though retrieval is used to obtain latest facts.
- **MemStrata harness and datasets:**
    - The MemStrata paper releases datasets and evaluation protocols for memory under knowledge evolution, explicitly measuring stale‑fact error rates when facts change and multiple versions coexist.[^3]
    - These test whether the system retires superseded values and return only current ones, closely aligned with your goal, though they focus on API/knowledge scenarios rather than bank PDFs.
- **AionRAG benchmarks:**
    - Uses controlled drift tests and real‑world evolving corpora, including Wikipedia revision histories and U.S. Federal Register policies, to evaluate temporal consistency and faithfulness under knowledge drift.[^5]
    - Benchmarks measure whether the system selects evidence that is valid at the query time, implicitly requiring conflict resolution between historical and current versions.
- **SituatedQA + “Mitigating Temporal Misalignment by Discarding Outdated Facts”:**
    - SituatedQA focuses on time‑sensitive queries; the EMNLP paper measures misalignment between model training date, evidence date, and query date and evaluates strategies to discard outdated facts.[^4]
    - Not specifically multi‑document conflict, but relevant for temporal relevance.
- **RegTrack:**
    - Benchmark for fine‑grained legal change detection on EU regulations; tasks include structural alignment and change classification (six classes).[^10]
    - Useful for training change‑detection modules, but doesn’t directly evaluate QA on conflicting versions.
- **LawShift:**
    - Statutory revision‑oriented legal judgment prediction benchmark with 31 revision types and real cases; dataset and code available on HuggingFace and GitHub.[^11]
    - Evaluates how models handle statutory revisions when predicting judgments, indirectly assessing whether models align with the correct law version.

Overall, there is partial coverage of “choosing current facts under conflict,” especially in MemStrata and AionRAG, but no banking‑specific or tariff‑sheet datasets. You will likely need to create a small internal benchmark: synthetic queries with curated conflicting documents, plus gold labels indicating which fact is currently valid and why.

***

## Techniques table

| Technique | How it works | Source | Maturity |
| :-- | :-- | :-- | :-- |
| MemStrata bi‑temporal ledger | Stores facts as triples with validity intervals; when a new fact contradicts an existing one for same (subject, relation), deterministically retires the old value in a ledger; retrieval only returns temporally valid facts. | MemStrata paper on temporal validity in retrieval memory.[^1][^3] | Research‑only (open harness + datasets) |
| AionRAG time‑correct retrieval | Treats retrieval as control: predicts if retrieval is needed, selects a query‑specific evidence window and hop depth, filters candidates by time before semantic ranking, and applies conflict‑gated decoding when evidence disagrees with model prior. | AionRAG: Time‑Correct RAG under knowledge drift.[^5] | Research‑only |
| ConflictRAG | Two‑stage conflict detection (MLP + selective LLM) over retrieved documents; classifies conflict type (factual, temporal, opinion); resolves temporal conflicts via recency ranking; uses Entropy‑TOPSIS for source credibility and a conflict‑aware RAG score. | ConflictRAG paper.[^13][^14] | Research‑only |
| CSRAG conflict‑suppressed RAG | Extracts parametric facts and paraphrased context; applies dual logits processors to suppress tokens associated with conflicting parametric knowledge and boost context‑grounded tokens during decoding. | CSRAG paper on decoding‑time conflict resolution.[^16] | Research‑only |
| Reasoning‑Trace‑Augmented RAG | Fine‑tunes LLM to generate structured XML reasoning traces before answering: micro‑level doc analysis, macro‑level conflict classification, and conflict‑type‑specific synthesis (temporal → prefer newer). | BITS Pilani + CMU framework (LinkedIn technical summary).[^17] | Research‑only / early prototypes |
| Temporal conflict resolution prompt | Prompt template that forces the LLM to extract claims and dates, identify temporal conflicts, apply resolution rules (prefer most recent effective date, authority), and list deprecated claims with reasons. | Inferensys temporal conflict resolution prompt.[^15] | DIY‑needed; prompt‑level, but production‑usable |
| Layered retrieval \& conflict reranking | Vector retrieval to get many candidates; metadata filters (validity windows, trust), conflict detection (LLM/classifier), and reranking combining trust, freshness, conflict status before passing few chunks to generator. | “RAG in the real world: handling fresh data, conflicts, and source trust.”[^12] | DIY‑needed; pattern used in production RAG stacks |
| SAT‑Graph RAG / Graph RAG for Legal Norms | Ontology‑driven knowledge graph distinguishing legal Works and versioned Expressions; models Component Temporal Versions (CTV) and Action nodes; deterministic point‑in‑time retrieval via CTV validity intervals and policies like SnapshotLast. | Graph RAG for Legal Norms (arXiv + journal).[^8][^18][^9][^19] | Research with strong engineering; some production pilots |
| Regulatory context graph | Regulatory context graph with nodes for statutes, rules, obligations, dates; edges for amends/supersedes/effective_on; ingestion, parsing, entity/relationship extraction, temporal annotation; used to deliver explainable subgraphs. | Carver agents blog on context graphs for regulatory updates.[^6] | Production‑proven (case study) |
| Knowledge‑graph compliance engine | Four‑layer architecture (extraction, semantic parsing, knowledge graph, application services); provision nodes store effective dates, amendment history, supersession relations; incremental updates mark invalidated provisions. | Meta‑intelligence case study on regulatory compliance engine.[^7] | Production‑proven |
| Metadata filtering + hybrid retrieval | Hybrid lexical+dense retrieval combined with temporal metadata filtering; shows simple metadata + hybrid retrieval reduces temporal conflicts more than complex reasoning; warns about recall loss if metadata is wrong. | Legal RAG whitepaper on ablation of retrieval modules.[^20] | Research with applied focus |
| Contract date \& clause extraction pipeline | Schema‑driven extraction with regex/NER for standard dates and LLM for variable clauses; validation layer checks logical consistency and flags low‑confidence extractions. | Contract extraction engineering blogs.[^23][^25] | Production‑proven in contract tools (patterns widely used) |
| LLM‑based metadata extraction | Robust text conversion + NER‑enhanced chunk selection + CoT and structured tool calling LLM for metadata (including effective dates); structured JSON output with validation. | Metadata extraction leveraging LLMs.[^31] | Research‑only but directly implementable |
| RegTrack benchmark | Fine‑grained legal change detection benchmark on EU regulations: atomic units, six change types; tasks include structural alignment and change classification to support version‑aware retrieval. | RegTrack (ACL 2026).[^10] | Research benchmark |
| LawShift benchmark | Legal judgment prediction dataset under statutory revisions with 31 revision types; tests models’ ability to handle legal changes; datasets and code on HuggingFace and GitHub. | LawShift (NeurIPS datasets \& benchmarks track).[^11] | Research benchmark |
| WIKIRECENTCHANGES | Temporally grounded benchmark distinguishing stable vs recently updated facts; evaluates LLMs’ temporal conflict resolution behavior. | ACL 2026 temporal conflict resolution paper.[^2] | Research benchmark |
| MemStrata stale‑fact harness | Harness and datasets measuring stale‑fact error rate under evolving knowledge, demonstrating RAG vs MemStrata under conflicting fact versions. | MemStrata paper.[^3] | Research benchmark/tooling |
| AionRAG drift benchmarks | Controlled drift tests and real corpora (Wikipedia revisions, Federal Register, financial news) to evaluate temporal consistency and faithfulness. | AionRAG paper.[^5] | Research benchmark |


***

## Recommendation

For the hackathon banking agent, treat “current‑fact selection” as a small, explicit subsystem rather than hoping the LLM “does the right thing.” Concretely:

1. **Normalize dates and document fields at ingestion:** use OCR + layout‑aware parsing, regex/NER for standard date positions, and LLM extraction for tricky cases, all mapped into a schema with fields like `publication_date`, `effective_from`, `effective_to`, `document_type`, `supersedes_doc_id`, `applies_to_product`, `applies_to_channel`.[^29][^30][^23]
2. **Build a minimal fact‑level ledger:** for each canonical key (e.g., `card_type + operation + channel`), maintain rows with `value`, `effective_from`, `effective_to`, `source_doc_id`, and `supersedes_row_id`, applying MemStrata‑style deterministic supersession when a new conflicting value appears.[^1][^3]
3. **Add a temporal filter step before retrieval:** given a query, infer the relevant date (default: now) and filter candidate chunks to those whose fact rows are valid at that time, akin to AionRAG’s time filtering and SnapshotLast policy.[^5][^8]
4. **Implement conflict detection + explanation:** when multiple valid candidates remain (e.g., overlapping scopes), run a simple conflict detection (same question, different numeric answer) and a small adjudication LLM prompt (inspired by ConflictRAG and the inferensys temporal‑conflict prompt) to select a winner based on effective dates and authority, while generating a structured explanation listing deprecated facts and their sources.[^15][^12][^13]
5. **Return answers with explicit evidence chain:** always show the chosen fact, its effective interval, and source document; include a “deprecated information” section whenever conflicts were detected, mirroring the temporal‑conflict prompts and legal GraphRAG provenance patterns.[^9][^15][^8]

This gives you a concrete, auditable mechanism for “telling current data from outdated data” using simple tables and prompts, while still being grounded in state‑of‑the‑art temporal RAG and legal GraphRAG ideas.
<span style="display:none">[^33][^34][^35][^36][^37][^38][^39][^40][^41][^42][^43][^44][^45]</span>

<div align="center">⁂</div>

[^1]: https://arxiv.org/html/2606.26511v1

[^2]: https://aclanthology.org/2026.findings-acl.103/

[^3]: https://arxiv.deeppaper.ai/papers/2606.26511v1

[^4]: https://aclanthology.org/2023.emnlp-main.879.pdf

[^5]: https://sciety.org/articles/activity/10.21203/rs.3.rs-8912660/v1

[^6]: https://carveragents.ai/blog/context-graphs-for-regulatory-updates-from-crawled-notices-to-agent-first-intelligence

[^7]: https://www.meta-intelligence.tech/en/case-fintech

[^8]: https://arxiv.org/html/2505.00039

[^9]: https://arxiv.org/abs/2505.00039

[^10]: https://aclanthology.org/2026.acl-srw.68/

[^11]: https://proceedings.neurips.cc/paper_files/paper/2025/file/adf82a0a1d52d93961476458b9566a2b-Paper-Datasets_and_Benchmarks_Track.pdf

[^12]: https://www.paulserban.eu/blog/post/rag-in-the-real-world-handling-fresh-data-conflicts-and-source-trust/

[^13]: https://arxiv.org/html/2605.17301v1

[^14]: https://www.alphaxiv.org/overview/2605.17301

[^15]: https://inferensys.com/prompts/rag-question-answering-prompts/answer-grounding-and-faithful-synthesis-prompts/temporal-conflict-resolution-prompt-for-rag-answers

[^16]: https://openreview.net/pdf/6ea6f2d98d2ed1c9f31dd8cdb7ccb1ca8beed827.pdf

[^17]: https://www.linkedin.com/posts/raphaelmansuy_when-an-ai-retrieves-two-documents-that-contradict-activity-7407661668408864768-F3V7

[^18]: https://arxiv.org/html/2505.00039v3

[^19]: https://journals.sagepub.com/doi/10.3233/FAIA251598

[^20]: https://pub-4a4d4db48b7948f797e2a492d8cd0be8.r2.dev/Project/rag_whitepaper_shorter.pdf

[^21]: https://www.afp.com/en/infos/archerr-proves-purpose-built-ai-beats-general-purpose-llms-regulatory-change-management-95

[^22]: https://www.regology.com/regulatory-change-agent

[^23]: https://subhajitbhar.com/blog/idp/contract-data-extraction/

[^24]: https://guidesfor.dev/langextract-guide-2026/14-project-legal-contracts/

[^25]: https://dev.to/trinh_trankhanhduy_3429/local-llm-for-legal-documents-what-works-what-doesnt-honest-review-20d7

[^26]: https://legaldocpro.com/

[^27]: https://www.nutrient.io/api/data-extraction-api/legal/

[^28]: https://airparser.com/contract-parser/

[^29]: https://docpeel.com/use-cases/contract-data-extraction

[^30]: https://unstract.com/blog/ai-legal-document-data-extraction-processing/

[^31]: https://arxiv.org/html/2510.19334v1

[^32]: https://sergeyvasiliev.substack.com/p/temporal-semantics-in-graphrag-for

[^33]: https://www.themoonlight.io/en/review/when-benchmarks-age-temporal-misalignment-through-large-language-model-factuality-evaluation

[^34]: https://openreview.net/notes/edits/attachment?id=ssIRwxyTzo\&name=pdf

[^35]: https://www.reddit.com/r/LangChain/comments/1sdhmd8/anyone_seeing_rag_break_on_temporally_evolving/

[^36]: https://www.adobe.com/acrobat/business/hub/formatting-legal-documents.html

[^37]: https://contracts.justia.com/contract-clauses/superseding-agreement/

[^38]: https://afterpattern.com/clauses/supersedes-previous-agreements

[^39]: https://fynk.com/en/clauses/supersedes-previous-agreements/

[^40]: https://laws-lois.justice.gc.ca/eng/FAQ/

[^41]: https://www.legislation.gov.uk/changes

[^42]: https://huggingface.co/datasets/alerterra/regulatory_changes

[^43]: https://data360.worldbank.org/en/dataset/ITU_ICT

[^44]: https://aclanthology.org/2026.acl-srw.68.bib

[^45]: https://github.com/shcherbak-ai/contextgem


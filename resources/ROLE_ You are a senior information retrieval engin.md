<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# ROLE: You are a senior information retrieval engineer who has built and

benchmarked retrieval systems specifically for financial document QA
(regulatory filings, tariff/fee documents, multi-hop numerical reasoning
over financial text and tables).

CONTEXT: We are building an agentic RAG system for a banking hackathon.
The corpus is dozens of heterogeneous financial PDFs (regulations,
tariffs, internal orders) plus a separate structured transaction registry.
Questions require multi-hop reasoning: combining a fact from one document
with a fact from another, then linking to transaction records, then
computing a value. The real dataset is not released yet (available Aug 6,
private set Aug 9 with only a 3-hour window to finalize), so we need to
validate our retrieval pipeline NOW using existing open benchmarks that
resemble this task, before real data arrives. Output goes to an engineer
who will implement the retrieval layer and also needs a shortlist of
datasets to download and test against this week.

PART A — RETRIEVAL STRATEGY

1. For multi-hop financial QA specifically, compare: pure dense retrieval,
BM25/sparse-only, hybrid (BM25+dense) with reranking, and any
agentic/iterative retrieval approaches (query decomposition, multi-step
retrieval loops). Which has published superior performance on
financial QA benchmarks specifically, not general QA? Cite the actual
numbers and source.
2. What reranker models/approaches are currently state-of-the-art for
this domain (e.g., cross-encoder rerankers, LLM-based reranking) and is
the accuracy gain worth the added latency for a system under a hard
time budget?
3. How should retrieval handle TABLE-heavy content differently from prose
— is there a documented benefit to separate table-retrieval /
table-QA pathways (e.g., text-to-SQL over extracted tables) versus
treating tables as regular text chunks?
4. What chunking strategies are shown to work best for financial
regulatory/tariff documents specifically (fixed-size vs semantic vs
structure-aware/section-based chunking)? Any benchmarked comparisons?

PART B — OPEN FINANCIAL QA BENCHMARKS/DATASETS

List every open, downloadable dataset relevant as a proxy for this task,
including but not limited to: FinQA, TAT-QA, ConvFinQA, FinanceBench,
FinanceComplexQA, DocFinQA, and any others you find (search actively —
this space has grown fast in 2025-2026). For EACH dataset report:

- What it actually contains (document types, question types, whether it
includes multi-hop questions, whether it includes tables)
- Size (number of documents/questions)
- Where to download it (direct link)
- Whether it includes any notion of document versioning/conflicting
information (most won't — flag which ones, if any, do)
- License/usability constraints
- How closely it resembles OUR task (multi-doc + transaction-linking +
versioning) vs. how much of a stretch it is as a proxy

SOURCE PRIORITY: Papers with published benchmark numbers, official
dataset repos/pages (HuggingFace, GitHub, arXiv), recent (2025-2026)
survey papers that compare multiple financial QA datasets. Deprioritize
blog posts with no benchmark data.

EXCLUDE: non-financial-domain retrieval benchmarks (general open-domain
QA like Natural Questions, HotpotQA) unless directly cited as a
methodology reference for multi-hop technique design.

OUTPUT FORMAT:

- Part A: answer each numbered question with inline citations, end with a
comparison table (approach | reported accuracy/metric | source | added
latency/cost).
- Part B: a table (dataset | contents | size | download link | has
versioning? | license | resemblance to our task: high/medium/low),
followed by a short paragraph recommending which 2-3 datasets are worth
actually downloading and testing against this week.

On current financial text‑and‑table QA benchmarks, a two‑stage hybrid retrieval pipeline (BM25 + dense fusion) with neural reranking is clearly superior to pure dense, pure BM25, and agentic “corrective” retrieval; table‑aware and structure‑aware chunking also beat naive fixed‑size chunking, and the closest proxy datasets for your hackathon are TAT‑QA, DocFinQA, FinanceComplexQA, plus FinQA/ConvFinQA for focused numerical reasoning.[^1][^2][^3][^4][^5]

Below I focus on what you need to implement and test this week.

***

## Part A – Retrieval strategy

### 1. Dense vs BM25 vs hybrid vs agentic/iterative (financial QA only)

The most relevant study is **“From BM25 to Corrective RAG: Benchmarking Retrieval Strategies for Text‑and‑Table Documents” (Akarsu et al., 2026)**, evaluated on **T²‑RAGBench**, a financial QA benchmark built from FinQA, ConvFinQA, and TAT‑DQA (23,088 queries, 7,318 mixed text‑and‑table financial documents).[^6][^2][^1]

Key retrieval results (T²‑RAGBench, financial documents):[^2][^6][^1]

- **BM25 (sparse only)**
    - Recall@5 = 0.644, MRR@3 = 0.411, nDCG@10 = 0.515.[^6][^2]
- **Dense only (OpenAI text‑embedding‑3‑large)**
    - Recall@5 = 0.587, MRR@3 = 0.351, nDCG@10 = 0.466.[^2][^6]
- **Hybrid fusion (BM25 + dense, RRF)**
    - Recall@5 = 0.695, MRR@3 = 0.433, nDCG@10 = 0.551.[^6][^2]
- **Hybrid + neural reranking (Cohere Rerank)**
    - Recall@5 = 0.816, MRR@3 = 0.605, nDCG@10 = 0.683.[^1][^2][^6]

So on **financial text+table QA**, BM25 outperforms state‑of‑the‑art dense retrieval, hybrid fusion beats both, and **hybrid + cross‑encoder reranking is best by a large margin** (Recall@5 +26.7% vs BM25, +39.0% vs dense).[^1][^2]

Agentic / iterative retrieval was also tested:

- The **Corrective RAG (CRAG)** variant, which rewrites queries and adapts retrieval, achieved Recall@5 ≈0.658—**worse than simple hybrid fusion (0.695)** on this financial benchmark.[^7][^2][^6]
- The authors conclude that query expansion (HyDE, multi‑query) and adaptive retrieval give **limited benefit for precise numerical financial queries**, compared with just doing strong hybrid + rerank.[^2][^1]

A separate multi‑hop study over regulatory/financial texts finds BM25 can even beat naive hybrid fusion on multi‑hop questions (ROUGE‑L 0.304 vs 0.239), underscoring how lexical matching of entity names and metric labels is crucial.[^8]

**Practical takeaway for your hackathon corpus (tariffs, regulations, orders):**

- Do **not** use pure dense retrieval as your backbone; BM25 is stronger for this domain.
- Use **BM25 + dense fusion** as your candidate generator, and add a **neural reranker**—this is the best‑documented configuration for financial text‑and‑table QA.
- Agentic query‑rewriting loops (HyDE/CRAG) are *optional* and did **not** surpass hybrid+r in the published financial benchmarks; you can safely treat them as a later optimization, not core infrastructure.[^7][^1][^2]

***

### 2. Rerankers, SOTA and latency trade‑offs

**Models / approaches used on financial QA:**

- **Cohere Rerank v3** (managed cross‑encoder)
    - Used as the reranker in the Akarsu et al. financial benchmark; drove Recall@5 from 0.695 (hybrid RRF) to 0.816 and MRR@3 from 0.433 to 0.605.[^6][^1][^2]
    - Latency: ~30–50 ms p50 for ~100 candidates over API in production practice.[^9]
- **BAAI/bge‑reranker‑v2‑m3 (open‑source cross‑encoder)**
    - Recommended as default reranker in recent hybrid‑search engineering guides; ~40–80 ms on CPU for 100 candidates, ~10–25 ms on GPU.[^9]
    - Gives state‑of‑the‑art reranking quality on general and domain QA benchmarks while keeping latency within a typical 100–200 ms retrieval budget.[^10][^9]
- **LLM‑based reranking (GPT‑style)**
    - A recent RAG system with LLM reranking reports answer correctness improving from 33.5% to 49.0% (15.5‑point gain) when using LLM rerank, but warns that latency and API cost increase substantially.[^11]
    - A production comparison finds LLM reranking yields Precision@5 ≈0.88 with p50 latency ≈420 ms (top‑20 candidates), versus cross‑encoder ≈0.91 Precision@5 with ≈95 ms latency.[^12]

**Cost/latency budgets:**

- A hybrid search guide shows typical p95 breakdown for streaming RAG:[^9]
    - BM25 top‑100: ~5–15 ms.
    - Dense ANN top‑100 (HNSW): ~10–30 ms.
    - RRF fusion + passage hydration: ~6–17 ms combined.
    - Cross‑encoder rerank (100 candidates): ~40–80 ms (often ~50% of the retrieval budget).
- A financial trading blog comparing BM25 vs full RAG pipeline notes BM25 retrieval at 3.2 ms vs RAG at 287 ms average, with most overhead coming from embeddings and reranking.[^13]

**Is reranking “worth it” under a hard time budget?**

- On the financial text‑and‑table benchmark, **reranking adds ~50–80 ms but yields Recall@5 gains of +17–39 percentage points over non‑reranked baselines**, with statistically significant improvements on all metrics.[^1][^2][^9]
- Reranking benefits saturate at relatively shallow depths: re‑ranking top‑50 to top‑100 candidates captures ~90% of possible nDCG gains, so you don’t need 500‑candidate reranking.[^14][^9]

For a hackathon agent:

- If your end‑to‑end LLM budget is ~2–5 seconds, **spending ~70 ms on cross‑encoder reranking is absolutely worth it** for multi‑hop numerical banking questions.
- Prefer **cross‑encoder rerankers (bge‑reranker or Cohere Rerank)** over LLM reranking: similar or better quality at 3–5× lower latency and much lower cost.[^12][^9]
- Only consider **LLM‑based reranking** if you have very small candidate sets (≤20) and are optimizing for maximum precision over a tiny query volume.[^11][^12]

***

### 3. Handling table‑heavy content vs prose

Empirical work on financial table‑and‑text QA shows that **treating tables as structured objects with dedicated reasoning pipelines beats “tables as plain text”**:

- **TAT‑QA (financial table+text QA)** introduces TAGOP, a model that explicitly tags relevant table cells and text spans and then applies symbolic aggregation operators (add, subtract, divide, etc.). TAGOP achieves **58.0% F1**, an **11.1‑point absolute gain** over the previous best baseline on TAT‑QA, while human experts reach 90.8% F1.[^5][^15]
- **FinQA** represents answers as executable programs over structured tables plus pre/post text. The dataset is explicitly designed for multi‑step numerical reasoning over combined structured and unstructured evidence, not simple span extraction.[^16][^17][^18][^19]
- In Akarsu et al.’s error analysis on financial text‑and‑table documents, **73% of retrieval failures were attributed to table structure mismatches**, i.e. relevant numbers were present but not correctly associated with their headers or rows.[^7][^6]

These findings collectively indicate a **documented benefit to table‑aware retrieval and QA pathways**:

- Detecting table boundaries and keeping entire financial tables intact within chunks avoids splitting rows or columns, which breaks the semantics.[^20]
- Using a **table‑specific reasoning layer** (symbolic programs, text‑to‑SQL, or cell‑selection plus aggregation) significantly improves accuracy over treating tables as flat text, especially for multi‑step computations like “fee for account type X across periods Y and Z”.[^15][^16][^5]

For your system:

- **Index tables separately** with metadata (row/column labels, measure types, currencies) and link them to the surrounding prose chunks.
- For questions that clearly target tariffs/fees or balances, route them through a **table‑QA pathway** (symbolic calculator or text‑to‑SQL over the extracted tables) instead of only passing flattened text chunks to the LLM.
- Maintain a unified retrieval ranking so that both table chunks and prose chunks compete, but **handle execution differently** once the right table snippet is retrieved.

***

### 4. Chunking strategies for financial regulatory/tariff documents

On financial reports and regulatory‑style documents, several studies and engineering guides converge on the same message: **structure‑aware, table‑aware, and semantic chunking outperform naive fixed‑size splitting**.

Evidence:

- A 2026 analysis of RAG chunking for financial document analysis reports that **structure‑aware and semantic chunking approaches enhance precision by up to 33.42% compared to naive fixed‑size chunking**, across financial report datasets.[^21]
- NVIDIA’s 2024 chunking benchmark (including financial documents) finds that **financial documents perform best with ~1,024‑token chunks (57.9% accuracy)**, supporting the idea that longer, context‑rich chunks are beneficial for complex reasoning.[^22]
- A finance‑focused chunking guide recommends:[^20]
    - **Section‑based chunking** for 10‑K/10‑Q filings and compliance docs (split by Item sections, policy sections, etc.), with table‑aware splitting for financial statements.
    - Use fixed‑size (512–1024 tokens) only within narrative sections, not across tables or section boundaries.
- A summarisation study over regulations (ReguSum) shows that **section‑aware hierarchical structuring and semantic chunking outperform simple truncation and naive chunking for long regulatory documents**, reinforcing the value of respecting document structure.[^23]
- An empirical evaluation of PDF parsing/chunking on real‑world financial PDFs finds that parser and chunking choices significantly affect downstream tasks, with structure‑aware parsing and chunking yielding better retrieval and summarisation quality.[^24]

**Recommended chunking for your regulatory/tariff PDFs:**

- **Primary strategy: structure‑aware, section‑based chunking**
    - Detect headings, article/section numbers, and tariff tables.
    - Create chunks aligned to sections (e.g., “Fees for consumer accounts”, “Card tariffs”, “Interest rate rules”) with ~512–1024 tokens and 10–20% overlap.
- **Table‑aware chunking:** keep each tariff/fee table intact as one chunk; do *not* split inside tables.[^21][^20]
- **Semantic or hierarchical chunking** for particularly long or cross‑referenced documents (e.g., internal orders referencing multiple sections), to maintain local coherence while allowing multi‑hop retrieval.[^22][^23][^21]

***

### Retrieval approach comparison (financial QA)

| Approach | Reported metrics (financial QA) | Source | Added latency / cost (typical) |
| :-- | :-- | :-- | :-- |
| BM25 / sparse only | Recall@5 = 0.644, MRR@3 = 0.411, nDCG@10 = 0.515 on T²‑RAGBench.[^6][^2] | Akarsu et al. 2026 | ~5–15 ms for top‑100 in OpenSearch; negligible infra cost.[^9][^13] |
| Dense only | Recall@5 = 0.587, MRR@3 = 0.351, nDCG@10 = 0.466 on same benchmark.[^6][^2] | Akarsu et al. 2026 | Embedding + ANN adds ~15–50 ms, plus embedding API cost.[^9][^13] |
| Hybrid (BM25 + dense, RRF) | Recall@5 = 0.695, MRR@3 = 0.433, nDCG@10 = 0.551.[^6][^2] | Akarsu et al. 2026 | BM25 and dense in parallel (~20–40 ms total); small fusion overhead.[^9] |
| Hybrid + cross‑encoder rerank | Recall@5 = 0.816, MRR@3 = 0.605, nDCG@10 = 0.683 (Cohere Rerank).[^6][^1][^2] | Akarsu et al. 2026 | +40–80 ms to rerank ~100 candidates; dominant retrieval cost but still <100 ms.[^9] |
| Agentic / Corrective RAG (CRAG) | Recall@5 ≈0.658 (worse than simple hybrid fusion 0.695) on T²‑RAGBench.[^6][^7] | Akarsu et al. 2026 + blog | Extra query‑rewrite/validation passes; tens of ms overhead, little gain on numerical queries.[^1][^7] |
| LLM‑based reranking | Answer correctness +15.5 points (33.5% → 49.0%), Precision@5 ≈0.88; p50 latency ≈420 ms vs cross‑encoder ≈95 ms.[^11][^12] | RAG reranking studies | Hundreds of ms and significantly higher API cost per query; best reserved for low‑volume, high‑precision use.[^12][^11] |


***

## Part B – Open financial QA benchmarks / datasets

### Dataset overview and suitability table

| Dataset | Contents (docs \& questions) | Size (docs / QAs) | Download link | Has versioning? | License / usability | Resemblance to your task (multi‑doc + transactions + versioning) |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| **FinQA** | Q\&A over S\&P 500 earnings reports; each example has pre‑text, post‑text, and a financial table; answers are executable programs requiring multi‑step numerical reasoning over tables + text.[^16][^17][^18][^19] | ≈2.8k reports, 8,281 Q\&A pairs.[^17][^19] | GitHub: https://github.com/czyssrs/FinQA[^18]; HF: https://huggingface.co/datasets/ibm-research/finqa[^17] | No explicit versioning; single report per example.[^16][^17] | Research use; see repo (academic EMNLP 2021).[^16][^18] | **Medium** – strong numerical table reasoning, but mostly single‑document, no separate transaction registry. |
| **TAT‑QA** | Hybrid table‑and‑text QA over real financial reports (10‑K/10‑Q); each context includes one table + ≥2 paragraphs; questions require multi‑step numerical reasoning over text + tables.[^5][^25][^26][^27][^15] | 182 reports, 2,757 hybrid contexts, 16,552 questions.[^5][^27][^15] | GitHub: https://github.com/NExTplusplus/TAT-QA[^27]; project: https://nextplusplus.github.io/TAT-QA/[^26] | No versioning; each context from a single report.[^5] | CC BY 4.0 (Creative Commons Attribution).[^26][^27] | **High (for text+table reasoning)** – one report at a time, but very close in terms of tariffs/fee‑style table reasoning; no transaction registry. |
| **ConvFinQA** | Conversational multi‑turn numerical QA over financial reports; dialogues decompose FinQA questions into chains of turns referencing prior answers; each example includes pre‑text, post‑text, table, and programs per turn.[^28][^29][^30][^31] | 3,892 conversations, 14,115 questions; 3,037/421/434 train/dev/test conversations.[^29][^28][^30] | GitHub: https://github.com/czyssrs/ConvFinQA[^28]; HF variants (e.g. MehdiHosseiniMoghadam/ConvFinQA).[^32] | No versioning; based on static FinQA reports.[^29] | Research use; EMNLP 2022 dataset.[^29][^28] | **Medium** – excellent for testing multi‑step reasoning chains and conversational state, but still single‑document and no transaction linkage. |
| **DocFinQA** | Long‑context financial reasoning; extends FinQA by attaching **full SEC filings** (full annual reports) to each question, with parsed text and tables in markdown.[^4][^33] | ≈801–1,236 unique SEC filings; 5,735 train, 780 dev, 922 test questions.[^4][^33] | HF: https://huggingface.co/datasets/kensho/DocFinQA[^34][^35] | No explicit versioning; one filing per question, but multi‑page context.[^4] | Research use; ACL 2024 dataset.[^4][^33] | **High** – very close for retrieval: heterogeneous long SEC‑style documents with tables + narrative; still lacks transaction registry but great proxy for regulatory retrieval + multi‑hop reasoning. |
| **FinanceBench** | Open‑book financial QA over SEC 10‑K/10‑Q/8‑K, earnings reports, and call transcripts; questions require locating figures and performing arithmetic (YoY growth, margins, etc.) with evidence spans.[^36][^37][^38] | Full benchmark: 10,231 questions; open‑source sample: 150 annotated cases.[^36][^39][^38] | HF sample: https://huggingface.co/datasets/PatronusAI/financebench[^39]; GitHub sample: https://github.com/patronus-ai/financebench[^36][^38] | No versioning; each question linked to specific document.[^36] | Sample open for research; full dataset requires license from Patronus AI.[^36] | **Medium** – realistic questions over filings with retrieval + numeric reasoning; still mainly single‑document and no transactions, but excellent for end‑to‑end RAG evaluation. |
| **FinanceComplexQA** | Bilingual (Chinese/English) complex QA over diverse financial documents: corporate reports, bank statements, investment strategy reports, fintech research, customer service logs, compliance/audit docs; tasks include retrieval, multi‑hop reasoning, numeric calculation, comparison, planning, summarisation, and evidence‑grounded verification.[^3] | 2,026 unique QA examples in FinComplexQA‑Pro; 4,052 record views; ≈2,083 reference documents.[^3] | HF: https://huggingface.co/datasets/Multilingual-Multimodal-NLP/FinanceComplexQA[^3] | No explicit versioning; documents are static, but include multiple document types and scenes.[^3] | Research use; see HF dataset card.[^3] | **High** – closest to “agentic RAG” over heterogeneous financial PDFs; multi‑doc, multi‑hop reasoning and evidence verification, though still no separate transaction registry. |
| **SEC‑QA** | Systematic evaluation corpus for financial QA built from SEC filings; focuses on **multi‑document, multi‑page, refreshable QA** including inter‑company comparisons, temporal reasoning, and trend analysis over hybrid text+tables.[^40][^41][^42] | Table shows SEC‑QA uses ≈1,315 docs; expanded from 127 to 333 QAs, designed to scale to many more examples.[^40][^41] | Paper: https://aclanthology.org/2025.finnlp-2.15/[^42]; dataset URL is referenced in the paper’s arXiv (2406.14394) but not visible in the snippet—you’ll need to follow the project/repo link from the paper.[^41] | **Refreshable** corpus by design (new QAs added from recent filings), but not explicit “conflicting versions” within the same document set.[^41] | Research use; FinNLP 2025; likely non‑commercial academic use—check project page.[^42][^41] | **High** – designed for multi‑doc, multi‑page reasoning over regulatory filings; closest to your multi‑doc regulatory questions, but still no separate transaction registry. |
| **LOFin** | LOFin is a large‑scale open‑domain financial QA benchmark built on ≈145,000 SEC filings, combined with hierarchical retrieval and evidence curation; includes 1,595 open‑domain questions grounded in filings.[^41] | ≈145k filings; 1,595 QAs mentioned in LOFin description.[^41] | Described in “Hierarchical Retrieval with Evidence Curation for Open‑Domain Financial QA”; dataset link not in snippet, but should be in the paper’s resources.[^41] | No explicit versioning, but built on a continually updated SEC corpus.[^41] | Research use; check paper/project page. | **High** – multi‑doc, open‑domain QA over many filings; strong proxy for long‑range retrieval, but lacks tariff/transaction registry and may be heavier than needed for hackathon. |
| **FiQA (FiQA‑2018 / BEIR)** | Financial opinion mining and QA over news, forums, and other text sources; includes aspect‑based sentiment and factual QA; primarily free‑text, no tables.[^43][^44][^45] | BEIR subset has 648 QA examples for testing; full FiQA has tens of thousands of items.[^43][^45] | HF: https://huggingface.co/datasets/LLukas22/fiqa[^45] or ContextualAI/fiqa2018[^44] | No versioning; free‑text items only.[^43] | CC BY‑NC (non‑commercial academic use).[^45] | **Low** – useful for domain adaptation to financial language, but not table‑heavy, multi‑doc, or transaction‑linked. |
| **PDF‑VQA (finance subset)** | Multimodal PDF QA benchmark including financial documents; focuses on understanding scanned, noisy PDFs with complex layouts (tables, charts, text blocks).[^46] | Financial subset: 140k examples (PDF‑VQA row) across multimodal documents.[^46] | See PDF‑VQA dataset referenced in FinErva/FinVis‑GPT table; link in underlying paper (not in snippet).[^46] | No explicit versioning; PDFs static. | Research use; multimodal benchmark.[^46] | **Medium** – very relevant to PDF parsing and layout issues, but multimodal and broader than your task; does not have explicit transaction registry. |
| **FinErva** | Multimodal financial reasoning dataset with images, structured tables, and text; covers understanding and reasoning tasks, with chain‑of‑thought and ground‑truth annotations.[^46] | 7.54k samples; multimodal (images, tables, text).[^46] | Linked from FinErva paper; details not fully in snippet—follow paper references.[^46] | No versioning. | Research use; see paper. | **Low–Medium** – useful if you care about scanned/visual PDFs; overkill for text‑only hackathon RAG. |

*(Where download links are not visible in snippets, follow the cited paper/project pages; I avoided inventing URLs.)*

***

### Recommended datasets to actually download and test this week

Given your time constraints and the need to validate **retrieval + multi‑hop numerical reasoning over financial text+tables**, I’d prioritize:

1. **TAT‑QA** – best proxy for **tariff/fee‑style questions** over hybrid table+text.[^27][^5]
    - Use it to benchmark candidate generation (BM25 vs dense vs hybrid) and table‑aware reasoning.
2. **DocFinQA** – best proxy for **long, heterogeneous regulatory filings** with text+tables.[^4][^34]
    - Use it to stress‑test chunking strategies, hybrid + rerank pipelines, and long‑context retrieval over SEC‑like documents.
3. **FinanceComplexQA** – best proxy for **agentic, multi‑doc financial QA** over diverse document types (statements, compliance docs, customer records).[^3]
    - Use it to evaluate multi‑hop retrieval loops, cross‑document evidence aggregation, and LLM‑based reasoning traces.

If you have extra time, add **FinQA/ConvFinQA** for focused numerical program‑style reasoning, and **FinanceBench (open subset)** to sanity‑check end‑to‑end retrieval+generation on realistic filing questions.[^18][^39][^36]
<span style="display:none">[^47][^48][^49][^50][^51][^52][^53][^54][^55][^56][^57][^58][^59][^60][^61][^62][^63][^64][^65][^66][^67][^68]</span>

<div align="center">⁂</div>

[^1]: https://arxiv.org/abs/2604.01733

[^2]: https://denser.ai/blog/hybrid-search-for-rag/

[^3]: https://huggingface.co/datasets/Multilingual-Multimodal-NLP/FinanceComplexQA

[^4]: https://arxiv.org/html/2401.06915v3

[^5]: https://arxiv.org/pdf/2105.07624.pdf

[^6]: https://arxiv.org/html/2604.01733v1

[^7]: https://www.linkedin.com/posts/samiaafrinmithila_rag-ai-metrics-activity-7458501024119672832-ePvd

[^8]: https://papers.ssrn.com/sol3/Delivery.cfm/7087840.pdf?abstractid=7087840\&mirid=1

[^9]: https://www.calibreos.com/learn/genai-hybrid-search

[^10]: https://www.emergentmind.com/topics/hybrid-bm25-retrieval

[^11]: https://arxiv.org/html/2603.16877v1

[^12]: https://dev.to/neurolink/5-reranking-strategies-for-production-rag-pipelines-5g4f

[^13]: https://tradegladiator.com/blog/bm25-vs-rag

[^14]: https://www.elastic.co/search-labs/blog/elastic-semantic-reranker-part-3

[^15]: https://aclanthology.org/2022.aacl-main.72.pdf

[^16]: https://arxiv.org/abs/2109.00122

[^17]: https://huggingface.co/datasets/ibm-research/finqa

[^18]: https://github.com/czyssrs/FinQA

[^19]: https://finllm-leaderboard.readthedocs.io/en/latest/datasets/others/finqa.html

[^20]: https://www.finatune.com/en/ai/rag/guides/chunking-strategies-financial-reports

[^21]: https://research.binus.ac.id/airdc/2026/04/retrieval-augmented-generation-chunking-strategies-for-financial-document-analysis-a-systematic-literature-review/

[^22]: https://amirteymoori.com/rag-text-chunking-strategies/

[^23]: https://research.birmingham.ac.uk/en/publications/an-empirical-study-of-long-document-summarisation-methods-under-e/

[^24]: https://dl.acm.org/doi/10.1145/3786583.3786911

[^25]: https://finllm-leaderboard.readthedocs.io/en/latest/datasets/question_answering/tatqa.html

[^26]: https://nextplusplus.github.io/TAT-QA/

[^27]: https://github.com/NExTplusplus/TAT-QA

[^28]: https://github.com/czyssrs/ConvFinQA

[^29]: https://aclanthology.org/2022.emnlp-main.421.pdf

[^30]: https://www.scribd.com/document/895854949/ConvFinQA-Overview

[^31]: https://beancount.io/bean-labs/research-logs/2026/05/15/convfinqa-chain-numerical-reasoning-conversational-finance-qa

[^32]: https://huggingface.co/datasets/MehdiHosseiniMoghadam/ConvFinQA

[^33]: https://beancount.io/bean-labs/research-logs/2026/06/20/docfinqa-long-context-financial-reasoning-dataset

[^34]: https://huggingface.co/datasets/kensho/DocFinQA

[^35]: https://huggingface.co/datasets/kensho/DocFinQA/viewer/default/train

[^36]: https://docs.patronus.ai/docs/research_and_differentiators/financebench

[^37]: https://awesomeagents.ai/leaderboards/finance-llm-leaderboard/

[^38]: https://www.patronus.ai/announcements/patronus-ai-launches-financebench-the-industrys-first-benchmark-for-llm-performance-on-financial-questions

[^39]: https://huggingface.co/datasets/PatronusAI/financebench

[^40]: https://aclanthology.org/2025.finnlp-2.15.pdf

[^41]: https://aclanthology.org/2025.findings-acl.855.pdf

[^42]: https://aclanthology.org/2025.finnlp-2.15/

[^43]: https://langtest.org/docs/pages/benchmarks/legal/fiqa/

[^44]: https://huggingface.co/datasets/ContextualAI/fiqa2018/commit/88551ac384b457feddd19bffc99f1ff597f57523

[^45]: https://www.atyun.com/datasets/info/LLukas22/fiqa.html?lang=en

[^46]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12868138/table/T1/

[^47]: https://medium.com/@oren.dinai/beyond-vanilla-retrieval-technical-deep-dive-into-rag-extensions-for-finance-part-2-2f1c2d6a53bf

[^48]: https://www.youtube.com/watch?v=XvKiTfd6Xvo\&vl=en

[^49]: https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026

[^50]: https://atalupadhyay.wordpress.com/2026/06/10/building-a-production-ready-hybrid-retrieval-system-from-scratch-bm25-dense-embeddings-rrf-re-ranking/

[^51]: https://www.kunwar.page/chapter/060-hybrid-search-and-fusion

[^52]: https://medium.com/@noumannawaz/lesson-8-hybrid-retrieval-bm25-dense-bac3c702318b

[^53]: https://arxiv.org/pdf/2502.20245.pdf

[^54]: https://arxiv.org/abs/2212.09741

[^55]: https://deepwiki.com/fengbinzhu/TAT-LLM/6.2-tat-qa-dataset

[^56]: https://huggingface.co/datasets/next-tat/TAT-QA/commit/c96247f5077eac447f63527fd3dcfdc58bb56d6a

[^57]: https://huggingface.co/datasets/bowang0911/ConvFinQA

[^58]: https://github.com/justinas-kazanavicius/ConvFinQA

[^59]: https://research-api.cbs.dk/ws/portalfiles/portal/98727211/1564909_Thesis_final_version_.pdf

[^60]: https://www.patronus.ai/

[^61]: https://chatpaper.com/de/chatpaper/paper/264067

[^62]: https://www.scribd.com/document/1037913025/2604-01733v1

[^63]: https://huggingface.co/datasets/kensho/DocFinQA/tree/bbcef8279a4e95595eecb63cc5c476f44b55d964

[^64]: https://www.sec.gov/data-research

[^65]: https://opendatalab.com/OpenDataLab/FIQA

[^66]: https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets

[^67]: https://huggingface.co/datasets/TheFinAI/Fino1_Reasoning_Path_FinQA_v2

[^68]: https://huggingface.co/datasets/kensho/DocFinQA/commit/953038005992608415f123e2d2ef640c06dcd8cb


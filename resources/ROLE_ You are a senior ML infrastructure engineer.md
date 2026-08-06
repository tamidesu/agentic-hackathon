<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# ROLE: You are a senior ML infrastructure engineer specializing in document

ingestion pipelines for financial/regulatory AI systems. You are writing a
technical evaluation for an engineering team, not a marketing comparison.

CONTEXT: We are building an agentic RAG system for a banking hackathon. It
must ingest dozens of heterogeneous PDF documents (regulations, tariff
tables, contracts, internal orders/circulars) plus a transaction registry,
under a hard constraint: on evaluation day we get a private dataset and
have approximately 3 hours total to run the full pipeline end-to-end with
NO manual tuning per document. The dataset is synthetic banking data, so
documents may include deliberately messy formatting, scanned-looking
pages, nested tables, and multi-column layouts. Output goes directly to an
engineer for architecture decisions — be concrete, not generic.

RESEARCH QUESTIONS (answer each explicitly):

1. Compare these PDF extraction approaches for exactly this use case:
unstructured.io, LlamaParse, Docling (IBM), Azure Document Intelligence,
AWS Textract, native PDF ingestion in Gemini 2.x/Claude/GPT-4o/5,
and any other actively maintained 2025-2026 tool you find relevant.
For each: table-extraction accuracy, handling of scanned/low-quality
pages, multi-column layout handling, processing speed (pages/sec or
similar), cost per page/document, open-source vs paid, and whether it
preserves layout position (bounding boxes / page numbers) needed for
citation.
2. Specifically for TABLES (tariff grids, limit tables, fee schedules):
which approach has published or benchmarked accuracy numbers? Cite the
actual benchmark or paper, not a vendor claim without a source.
3. What is the realistic throughput for ingesting ~50-100 mixed PDF pages
within a tight time budget (need the full ingestion step to take
minutes, not hours) — which tools are fast enough, and which require
GPU/local infra vs pure API calls?
4. Is there a meaningful accuracy difference between using a dedicated
parsing library (Docling/unstructured) BEFORE feeding text to an LLM,
versus just handing the raw PDF directly to a large multimodal model
(e.g., Gemini/Claude native PDF understanding) and skipping a separate
parsing step entirely? Find any direct comparisons or benchmarks on
this specific question.
5. What open-source repos or reference pipelines exist that combine PDF
parsing with page/chunk-level citation metadata (i.e., not just "parse
text" but "parse text AND keep track of exactly which page/bbox each
chunk came from")? List actual repo links.

SOURCE PRIORITY: Prefer arXiv papers, official technical documentation,
GitHub repos with benchmarks in their README, and engineering blog posts
from companies that built similar systems. Deprioritize vendor marketing
pages and SEO listicles unless they contain unique benchmark data. Flag
the publication date of every source; flag anything older than 12 months
as "may be outdated" explicitly.

EXCLUDE: general "what is RAG" explanations, generic AI industry trend
commentary, anything not specific to PDF parsing/extraction mechanics.

OUTPUT FORMAT:

- A comparison table (tool | table accuracy | speed | cost | OSS?/API? |
citation-metadata support | notes) as the first thing in your response.
- Then a short section per research question above with inline citations
(links).
- End with a single "recommendation for this specific constraint set"
paragraph — but keep this as your opinion clearly labeled as such, not
mixed into the factual findings.

```

| Tool | Table accuracy | Speed | Cost | OSS?/API? | Citation-metadata support | Notes |
|---|---:|---:|---:|---|---|---|
| **Docling** | Strong on structured extraction; in one heterogeneous benchmark it led overall table detection/structure among open tools, and the Docling benchmark/eval repo is built around OmniDocBench. [^1][^2] | CPU-friendly but not the fastest; a reproducible 2026 OSS benchmark measured about 5.97 s/page mean on CPU in one run. [^3] | OSS; compute cost only if self-hosted. | Open source. [^4] | Yes: page number and bounding box are first-class provenance fields. [^5] | Best fit when you need deterministic provenance and can tolerate moderate CPU latency. May be outdated after 12 months. |
| **unstructured.io** | Decent for layout-aware extraction, but table quality varies by strategy; hi-res can return table structure and coordinates. [^4][^6][^7] | Fast mode is roughly text-layer speed; hi-res is materially slower, often cited around seconds/page. [^7] | OSS core + paid platform/API options. [^4] | Both. [^4] | Yes: coordinates and page numbers with hi-res/coordinates enabled. [^6][^7] | Good as a general ingestion layer, less compelling as the sole table parser for messy banking PDFs. May be outdated after 12 months. |
| **LlamaParse** | Strong on hard enterprise PDFs; ParseBench ranks LlamaParse Agentic highest overall and very strong on tables. [^8][^9][^10] | API throughput is designed for production; ParseBench did not publish pages/sec, but cost/perf is competitive. [^9][^11] | Paid API; about $12.50/1k pages for Agentic, $1.25/1k for basic parse. [^11][^12] | API. [^11] | Yes: ParseBench explicitly scores visual grounding; LlamaParse is built to preserve source locations. [^8][^9] | Very strong default if you can use a paid API and want fewer engineering surprises. May be outdated after 12 months. |
| **Azure Document Intelligence** | Good on OCR/layout; table extraction is useful but benchmark evidence is weaker than dedicated table systems. [^13][^14] | Default TPS is 15 on a resource; batch scale is API-bound, not local-GPU-bound. [^14] | Read: about $1.50/1k pages; prebuilt/layout/custom are higher. [^13][^15] | API / container. [^13] | Yes: layout and read outputs include structured page/page-range metadata. [^14] | Reliable enterprise OCR/layout pipeline; not the best pure table benchmark performer. May be outdated after 12 months. |
| **AWS Textract** | Good OCR/forms/tables; especially reliable for invoices/forms, but not known as the strongest heterogeneous table parser. [^16] | API-only throughput; no local infra required. [^16] | Tables: about $15/1k pages; forms+tabels+queries much higher. [^16][^17] | API. [^16] | Yes: returned blocks include page references; suitable for citations. | Often expensive for table-heavy workloads versus OCR-only alternatives. May be outdated after 12 months. |
| **Gemini 2.x native PDF** | Promising for “understanding” PDFs, but published spatial precision is limited; it is not a dedicated parser. [^18] | Fast for direct API Q&A, but throughput depends on model token/image limits rather than a parser pipeline. | Model-token pricing, not per-page parser pricing. | API. | Partial: supports page-level grounding and visual reasoning, but not reliable bbox-level provenance in the docs I found. [^18][^19] | Good for downstream reasoning, weaker as the sole ingestion backbone. May be outdated after 12 months. |
| **Claude PDF support** | Good visual PDF understanding, but explicit note says the Converse API falls back to basic text extraction unless citations are enabled. [^20] | API-only; page images/text are processed per document. [^20][^21] | Model-token pricing. | API. | Page-level citations supported; bbox-level extraction is not the default focus. [^20][^21] | Strong when you need grounded Q&A over PDFs, not a full parser with exact table structure. May be outdated after 12 months. |
| **Marker** | Strong OSS contender in 2026 benchmarks, especially on speed/structure tradeoffs; often competitive with Docling/MinerU in reproducible runs. [^3][^22] | Faster than many DL-heavy parsers at scale; some benchmarks report high page/min on H100. [^22] | OSS. | Open source. | Yes, layout-aware Markdown plus provenance-oriented output. | Worth evaluating if license and environment fit. May be outdated after 12 months. |
| **MinerU / MinerU2.5** | Often near the top on messy-PDF benchmarks, particularly for tables/formulas; needs GPU for the strong path. [^3][^22] | Fast per page in benchmarked runs, but more infra-heavy. [^3][^22] | OSS; self-host GPU. | Open source. | Yes, layout blocks/provenance. | Excellent accuracy candidate if you can run GPU inference. May be outdated after 12 months. |
| **OpenDataLoader PDF** | Claims top benchmark positioning and strong table scores, but source is vendor-hosted and should be treated cautiously. [^23] | Claims ~0.05 s/page local, but that needs independent verification. [^23] | OSS. | Open source. | Yes: bounding boxes and Markdown/JSON. [^23] | Interesting but I’d verify independently before betting a hackathon pipeline on it. |

## Q1

For **your exact banking/regulatory PDF problem**, the practical split is:

- **Best API-first parsing**: LlamaParse and, depending on budget, Claude/Gemini for direct multimodal fallback. ParseBench is the strongest source I found for enterprise-style comparison, and LlamaParse Agentic is the top overall there.[^9][^10]
- **Best OSS provenance-first parser**: Docling, because provenance/page/bbox support is explicit and it has a real evaluation ecosystem.[^2][^5]
- **Best general-purpose OSS ingestion wrapper**: unstructured, but hi-res is slower and table quality is more variable.[^4][^6][^7]
- **Best cloud OCR/layout services**: Azure Document Intelligence and AWS Textract, mainly for dependable OCR and structured extraction at scale.[^13][^14][^16]

## Q2

For **tables specifically**, the strongest published benchmark evidence I found is from **PubTables-1M / Table Transformer (TATR)** and newer multi-tool benchmarks built on **OmniDocBench** and **ParseBench**. TATR reports strong table detection and structure metrics on PubTables-1M, including AP and GriTS scores in the original repository.[^24]

For heterogeneous real-world PDFs, **OmniDocBench** and **ParseBench** are more relevant than old table-only datasets because they include bounding boxes, table recognition annotations, and layout/grounding tasks. OmniDocBench explicitly says it contains table bounding boxes and table recognition annotations; ParseBench evaluates tables with structural record matching and visual grounding.[^8][^25][^9]

Important caveat: some vendor/blog claims mention “table accuracy” numbers, but I would **not** rely on those unless they tie back to a public benchmark. The safest benchmark sources here are PubTables-1M/TATR, OmniDocBench, ParseBench, and the heterogeneous table extraction paper/benchmark I found on arXiv.[^1][^25][^9][^24]

## Q3

For **50–100 mixed PDF pages in minutes**, the tools that are comfortably fast enough are:

- **API-first parsers**: LlamaParse, Azure Document Intelligence, AWS Textract, Claude PDF, Gemini native PDF. These avoid local GPU ops and are operationally simple.[^16][^18][^20][^9][^13]
- **CPU OSS parsers**: Docling and unstructured fast mode can work if the pages are mostly digital PDFs and you only need one pass.[^3][^5][^7]
- **GPU-heavy OSS parsers**: MinerU-style VLM pipelines can be very accurate but are more infra-dependent.[^22][^3]

For your 3-hour evaluation window, the main risk is not raw OCR speed but **retry amplification** on ugly pages. That means a parser with strong default layout/table handling and explicit provenance is more valuable than one that is fast on easy pages but collapses on scans, multi-column layouts, or nested tables. The heterogeneous benchmark I found notes that rule-based methods are much faster, while DL/VLM methods are slower but more robust on messy layouts.[^1]

## Q4

Yes, there is a meaningful difference between **dedicated parsing first** and **raw PDF straight into an LLM**, but it depends on the document type.

The strongest evidence I found is indirect but relevant: **OHRBench** shows OCR/formatting noise materially hurts downstream RAG, and ParseBench scores “visual grounding” separately from table/content fidelity, which is exactly the failure mode of raw-PDF prompting.  Claude’s docs also make a key distinction: in the Converse API, **visual PDF understanding requires citations enabled**, otherwise it falls back to basic text extraction. That strongly suggests the raw-PDF route is not equivalent to a robust parser pipeline.[^20][^26][^9]

My read is:
- For **clean digital PDFs**, raw multimodal PDF input to Gemini/Claude can be good enough for Q&A.
- For **messy banking PDFs with tables, scans, multi-column pages, and citation requirements**, a dedicated parser usually wins because it gives you stable page/bbox metadata and lets the LLM reason over already-normalized structure.[^5][^6][^9]

## Q5

Good open-source references for **parsing with citation metadata**:

- **Docling** — `docling-project/docling`; provenance includes `page_no` and `bbox`.[^5]
- **Unstructured** — `Unstructured-IO/unstructured`; hi-res strategy supports coordinates and page-relative positions.[^6][^4]
- **OmniDocBench** — benchmark repo with page-level bounding boxes and table/layout annotations.[^25][^27]
- **ParseBench** — `run-llama/ParseBench`; benchmark explicitly evaluates visual grounding/source location.[^28][^8][^9]
- **OHR-Bench** — `opendatalab/OHR-Bench`; OCR/RAG benchmark with structured page-level ground truth.[^26]

## Recommendation

**Opinion:** for this exact hackathon constraint set, I would build the ingestion pipeline around **Docling or LlamaParse as the primary parser**, with a **fallback multimodal pass only for failure cases**, and keep **page/bbox provenance stored in the chunk schema from day one**. If you need to self-host and keep costs low, pick Docling; if you want the highest chance of surviving messy private PDFs with minimal tuning, pick LlamaParse Agentic and accept the API cost. For table-heavy banking docs, I would not rely on raw Gemini/Claude PDF ingestion alone.[^10][^9][^5]
<span style="display:none">[^29][^30][^31][^32][^33][^34][^35][^36][^37][^38][^39][^40][^41][^42][^43][^44][^45][^46][^47][^48][^49][^50][^51][^52][^53][^54][^55][^56][^57][^58][^59][^60][^61][^62][^63][^64][^65][^66][^67][^68][^69][^70][^71][^72][^73][^74][^75][^76][^77][^78][^79][^80][^81][^82][^83][^84][^85][^86][^87][^88]</span>

<div align="center">⁂</div>

[^1]: https://www.themoonlight.io/fr/review/benchmarking-table-extraction-from-heterogeneous-scientific-extraction-documents
[^2]: https://github.com/docling-project/docling-eval
[^3]: https://www.whichmodel.pro/articles/2026-06-pdf-parsing-open-source-benchmark/
[^4]: https://github.com/Unstructured-IO/unstructured
[^5]: https://docling-project.github.io/docling/reference/docling_document/
[^6]: https://docs.unstructured.io/api-reference/workflow/nodes/partitioner/partitioner-high-res
[^7]: https://genai4a11.github.io/concepts/unstructured.html
[^8]: https://github.com/run-llama/ParseBench
[^9]: https://arxiv.org/html/2604.08538v3
[^10]: https://www.parsebench.ai/
[^11]: https://developers.llamaindex.ai/llamaparse/general/pricing/
[^12]: https://docuocr.com/llamaparse-pricing
[^13]: https://azure.microsoft.com/en-us/pricing/details/document-intelligence/
[^14]: https://docs.azure.cn/en-us/ai-services/document-intelligence/service-limits?view=doc-intel-4.0.0
[^15]: https://learn.microsoft.com/en-in/answers/questions/5927427/azure-document-intelligence-pricing
[^16]: https://aws.amazon.com/textract/pricing/
[^17]: https://www.braincuber.com/blog/aws-textract-pricing-what-ocr-actually-costs
[^18]: https://winbuzzer.com/2025/04/21/gemini-2-5-pro-appears-to-be-first-ai-model-to-fully-understand-pdf-layouts-enabling-precise-citations-xcxwbn/
[^19]: https://medium.com/@varada883/inspecting-rich-documents-with-gemini-multimodality-and-multimodal-rag-393abf4aa0e6
[^20]: https://platform.claude.com/docs/en/build-with-claude/pdf-support
[^21]: https://aws.amazon.com/about-aws/whats-new/2025/06/citations-api-pdf-claude-models-amazon-bedrock/
[^22]: https://www.youtube.com/watch?v=8RxT5jTcemY
[^23]: https://opendataloader.org/
[^24]: https://github.com/microsoft/table-transformer
[^25]: https://github.com/opendatalab/OmniDocBench/blob/main/README.md
[^26]: https://github.com/opendatalab/OHR-Bench
[^27]: https://github.com/opendatalab/OmniDocBench
[^28]: https://huggingface.co/datasets/llamaindex/ParseBench
[^29]: https://arxiv.org/html/2410.09871v1
[^30]: https://arxiv.org/html/2603.10765v1
[^31]: https://www.firecrawl.dev/blog/best-pdf-parsers
[^32]: https://www.llamaindex.ai/insights/best-ai-pdf-parsers
[^33]: https://ijamjournal.org/ijam/publication/index.php/ijam/article/download/163/154
[^34]: https://github.com/genieincodebottle/parsemypdf
[^35]: https://dev.to/urios/parsing-bank-statement-pdfs-5-tools-compared-for-developers-2026-4b70
[^36]: https://discuss.huggingface.co/t/best-open-source-model-for-parsing-messy-pdfs-on-16gb-ram-cpu-only/168890
[^37]: https://www.docsumo.com/blog/best-document-parsing-tools
[^38]: https://pdfmux.com/blog/pdf-extraction-for-rag-pipeline/
[^39]: https://dev.to/anmolbaranwal/top-11-document-parsing-ai-tools-for-developers-in-2025-4m6a
[^40]: https://www.reddit.com/r/Rag/comments/1univjx/finetuned_a_vlm_for_messypdf_extraction_46_911_on/
[^41]: https://www.reddit.com/r/LangChain/comments/1n0pcgw/best_opensource_tools_for_parsing_pdfs_office/
[^42]: https://ossaihub.com/categories/document-intelligence-parsing/
[^43]: https://aclanthology.org/2025.xllm-1.2.pdf
[^44]: https://arxiv.org/html/2412.02592v1
[^45]: https://arxiv.org/html/2511.16134v1
[^46]: https://arxiv.org/pdf/2303.00716v2.pdf
[^47]: https://openreview.net/pdf/940942c89171e01085f257e7f04e5db4ee82407c.pdf
[^48]: https://pierre.senellart.com/publications/soric2026benchmarking.pdf
[^49]: https://www.alphaxiv.org/benchmarks/shanghai-ai-laboratory/ohrbench
[^50]: https://liner.com/review/ocr-hinders-rag-evaluating-cascading-impact-ocr-on-retrievalaugmented-generation
[^51]: https://deepwiki.com/microsoft/table-transformer/6.1-pubtables-1m
[^52]: https://pub.towardsai.net/ai-innovations-and-insights-27-ocr-hinders-rag-and-ragchecker-ec9cfc35274b
[^53]: https://note.com/all_small_stuff/n/n6986d6b070b7
[^54]: https://run.unl.pt/entities/publication/ebf09470-4460-41d2-ad54-630189a23d4f
[^55]: https://aws.amazon.com/ru/textract/pricing/
[^56]: https://aws.amazon.com/pm/textract/
[^57]: https://learn.microsoft.com/en-us/answers/questions/5665475/what-is-the-cost-of-document-intellegence-service
[^58]: https://www.azure.cn/en-us/pricing/details/form-recognizer/
[^59]: https://www.factualminds.com/tools/amazon-textract-pricing-calculator/
[^60]: https://docuocr.com/blog/azure-document-intelligence-pricing
[^61]: https://docuocr.com/azure-content-understanding-pricing
[^62]: https://aiproductivity.ai/pricing/azure-document-intelligence/
[^63]: https://parsli.co/compare/azure-document-intelligence
[^64]: https://medium.com/@kyandaks/amazon-textract-pricing-explained-and-how-to-track-usage-1969629dc332
[^65]: https://platform.claude.com/docs/ru/build-with-claude/pdf-support
[^66]: https://platform.claude.com/docs/ja/build-with-claude/pdf-support
[^67]: https://platform.claude.com/docs/pt-BR/build-with-claude/pdf-support
[^68]: https://platform.claude.com/docs/de/build-with-claude/pdf-support
[^69]: https://www.llamaindex.ai/blog/adding-document-understanding-to-claude-code
[^70]: https://anablock.com/blog/claude-pdf-processing-document-analysis
[^71]: https://www.datastudios.org/post/claude-ai-and-pdf-reading-in-2025-capabilities-limits-and-workflows
[^72]: https://tools.cooconsbit.com/en/articles/claude-pdf-analysis-guide-en
[^73]: https://www.datastudios.org/post/claude-and-pdf-documents-technical-complete-overview
[^74]: https://www.alibaba.com/product-insights/how-to-use-claude-3-s-document-analysis-to-summarize-academic-pdfs-while-preserving-citation-integrity.html
[^75]: https://news.ycombinator.com/item?id=42952605
[^76]: https://arxiv.org/pdf/2604.08538v2.pdf
[^77]: https://unstructured.io/blog/how-to-parse-a-pdf-part-1
[^78]: https://github.com/Unstructured-IO/unstructured/issues/3100
[^79]: https://github.com/Unstructured-IO/unstructured/blob/main/unstructured/partition/pdf.py
[^80]: https://github.com/Unstructured-IO/unstructured-inference
[^81]: https://github.com/opendatalab/OmniDocBench/blob/main/README_zh-CN.md
[^82]: https://github.com/Unstructured-IO/unstructured/issues/3194
[^83]: https://neelmishra.github.io/blog/mlops/rag/document-processing.html
[^84]: https://www.youtube.com/watch?v=IWhP5lvY_oc
[^85]: https://www.llamaindex.ai/blog/parsebench
[^86]: https://www.llamaindex.ai/blog
[^87]: https://www.codesota.com/browse/computer-vision/document-parsing/parsebench
[^88]: https://apio.sh/apis/llamaparse```


Media Intelligence Agent

The Media Intelligence Agent is an agentic system designed to autonomously collect, analyze, and compare news coverage from multiple sources, generating structured, bias-aware analytical reports on user-specified topics. It combines tool-augmented reasoning, dynamic workflow orchestration, and LLM intelligence to deliver actionable media insights.

Agentic Architecture & Workflow

StateGraph-based Agent Orchestration

Built with LangGraph, the system dynamically manages interactions between reasoning and external tools.

Workflow highlights:

User Query: Receives a topic or question.

Decision Logic: Determines whether to fetch new data or analyze existing content.

Tool Invocation: Calls integrated scrapers as tools for BBC and CNN articles.

LLM Reasoning: Uses a Groq-hosted LLaMA 3.3 70B model guided by a media-analysis system prompt.

Output: Produces structured, source-aware analytical reports.

Dynamic Tool Integration

Scrapers for BBC and CNN are bound to the agent, enabling on-demand data retrieval.

The system handles partial content extraction, fallback parsing, and retry logic intelligently.

Key Capabilities

Multi-source Media Intelligence: Aggregates headlines, summaries, full article text, and URLs.

Bias & Consistency Analysis: Identifies tone, framing, political leanings, and reporting inconsistencies.

Cross-source Comparison: Highlights shared facts, contradictions, and missing perspectives.

Structured Analytical Reporting: Generates concise, professional reports with clear factual takeaways.

Adaptive Reasoning: LLM decides when and which tools to use for efficient, context-aware analysis.

Tech Stack

LLM: Groq-hosted LLaMA 3.3 70B for reasoning

Orchestration: LangGraph for agentic workflow management

Data Extraction Tools: Scrapers for BBC and CNN

Utilities: Pandas for data structuring, dotenv for secure configuration

Use Cases

Journalists and analysts comparing media narratives

Researchers studying bias, framing, and news coverage trends

Developers building tool-augmented RAG systems

This system demonstrates a fully agentic architecture that integrates reasoning, tool invocation, and multi-source media intelligence, delivering autonomous, bias-aware insights in a scalable and transparent manner.

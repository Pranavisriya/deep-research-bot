# Deep Research

A modular research automation tool built with LangChain and LangGraph. Multiple specialized agents work in concert to plan, search, analyze, and synthesize—producing a structured, citation-ready report.

## Prerequisites

- Python 3.12 or higher
- Poetry for dependency management
- API keys for:
  - OpenAI
  - Anthropic
  - Google AI
  - Tavily (Mandatory)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Pranavisriya/deep-research-bot.git
cd deep-research
```

2. Create a virtual python environment in this repo
```bash
conda create -p venv python=3.12 -y
```

Any other method can also be used to create python environment.

3. Activate python environment
```bash
conda activate ./venv
```

4. Install `poetry` in the environment 
```bash
pip install poetry
```

5. Install dependencies using Poetry:
```bash
poetry install
```

6. Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
TAVILY_API_KEY=your_tavily_key
```

## Usage

Change the topic and outline based on your requirements in the `main.py` file.

Run the research workflow:
```bash
python main.py
```

## Features

- Multi-agent pipeline: planner, searcher, analyst, and writer roles orchestrated via LangGraph
- Deep web search: Tavily-powered targeted queries with iterative refinement
- Structured reports: customizable sections and final consolidation
- Human-in-the-loop: optional checkpoints for outline and draft feedback
- Configurable controls: depth, reflections, query caps, temperature
- Detailed logging: progress and decisions traced step-by-step

## Workflow
1. **Plan**
   - Generate/adjust report structure from the topic + outline
   - (Optional) Incorporate human feedback

2. **Research (per section)**
   - Leverage model prior knowledge
   - Run focused Tavily searches
   - Aggregate and rank sources
   - Reflect → generate follow-up queries
   - Synthesize a well-scoped section draft

3. **Curation**
   - Merge all sections
   - Normalize structure & style
   - Emit the final report

This loop can recurse until coverage and coherence criteria are met.

## License

This project is licensed under the terms included in the LICENSE file.

## Author

Pranavi Sriya (pranavisriyavajha9@gmail.com)

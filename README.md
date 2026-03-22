# Drug to Drug Interaction LLM Backend

Here is the comprehensive, finalized master plan for your **Universal Schema-Agnostic Text-to-Cypher Pipeline**.

By upgrading the architecture to handle intent-to-schema mapping, you have evolved this project from a simple "Drug Interaction Tool" into a **Universal Medical Graph Explorer**. Treat the Neo4j database as a completely flexible black box: your data engineering teammates can add new nodes (Patients, Pathogens, Genes) or new relationships (`CAUSES`, `PRESCRIBED_TO`), and this pipeline will instantly understand how to query them without a single line of backend code needing to change.

---



## 🚀 Quick Start (Local Development)

Follow these steps to set up the Universal Medical Graph API on your local machine.

### Prerequisites
* **Python 3.13+** installed on your machine.
* A running **Neo4j Database** (either [Neo4j Desktop](https://neo4j.com/download/) locally or a free cloud instance via [Neo4j AuraDB](https://neo4j.com/cloud/aura/)).
* An **OpenAI API Key** (for the LangChain LLM orchestration).

### Step 1: Clone the Repository
```bash
git clone <your-repository-url>
cd DDI_LLM_backend-advanced_llm_text_cypher
```

### Step 2: Set Up a Virtual Environment

It is highly recommended to isolate your Python dependencies.

```bash
# Create the virtual environment
python -m venv venv

# Activate it (Mac/Linux)
source venv/bin/activate

# Activate it (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies

Install the required packages from the requirements file.

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

1.  Copy the example environment file to create your local `.env` file.

<!-- end list -->

```bash
cp .env.example .env
```

2.  Open `.env` and fill in your actual API keys and database credentials:

<!-- end list -->

```env
OPENAI_API_KEY=sk-your-real-openai-api-key
NEO4J_URI=neo4j+s://your-database-id.databases.neo4j.io  # Or bolt://localhost:7687 for local
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-real-database-password
NEO4J_DATABASE=neo4j
```

### Step 5: Seed the Database (Crucial)

**Important:** This architecture dynamically pulls the database schema on every request. If your Neo4j database is completely empty, the LLM will fail because it won't know what nodes exist\!

Open your Neo4j browser and run this Cypher query to inject sample dummy data:

```cypher
CREATE (d1:Drug {name: 'Metformin'})
CREATE (d2:Drug {name: 'Iodinated Contrast Dye'})
CREATE (diag:Diagnosis {name: 'Type 2 Diabetes', icd_code: 'E11'})
CREATE (d1)-[:INTERACTS_WITH {severity: 'Severe', effect: 'Lactic Acidosis', mechanism: 'Decreased renal clearance'}]->(d2)
CREATE (d1)-[:TREATS {effectiveness: 'High', indication: 'First-line therapy'}]->(diag)
```

### Step 6: Start the Server

Run the FastAPI development server using Uvicorn.

```bash
uvicorn src.api:app --reload --port 8000
```

### Step 7: Test the API

FastAPI provides an automatic, interactive testing dashboard.

1.  Open your browser and navigate to: [http://localhost:8000/docs](https://www.google.com/search?q=http://localhost:8000/docs)
2.  Expand the **POST `/api/query`** endpoint.
3.  Click **"Try it out"** and send a test payload:

<!-- end list -->

```json
{
  "question": "Are there any severe interactions if I take Metformin with contrast dye?"
}
```

4.  Check your terminal to see LangChain dynamically generating the Cypher query, and the browser to see the JSON response\!

### 🧪 Running the Test Suite

To ensure your local setup is completely functional, run the pytest suite (this will test the API, prompt generation, and LangChain configurations in isolation without hitting the live database).

```bash
pytest -v
```

---

## **High-Level Architecture Flow**

1.  **User (React):** Types any medical question (e.g., "What treats Diabetes?" or "What has a dangerous reaction with Advil?").
2.  **Backend (FastAPI):** Forces LangChain to pull the absolute latest schema from Neo4j.
3.  **LLM Orchestrator (LangChain):**
    - Reads the user's intent to identify the core concepts (Disease + Treatment, or Drug + Drug).
    - Reads the fresh schema to find the closest matching Node Labels and Relationship Types.
    - Generates synonyms (Diabetes -> T2D, Advil -> Ibuprofen).
    - Writes a flexible Cypher query returning the target nodes and all edge properties via `properties(r)`.
4.  **Database (Neo4j):** Executes the dynamically generated query and returns the raw JSON.
5.  **Frontend (React):** Receives the JSON, identifies the node types, and maps the unknown properties to the interactive spiderweb graph.

### **Phase 1: Environment & Core Initialization**

The backend setup requires zero hardcoded configuration. It simply connects your LLM provider and the database.

### **Phase 2: The Universal "Intent-Based" Master Prompt**

This prompt forces the LLM to act as a dynamic translator between human medical questions and whatever structure currently exists in the database.

### **Phase 3: Chain Configuration**

Initialize the LLM to execute queries directly and return the raw JSON data without attempting to summarize it into a conversational response.

### **Phase 4: The Real-Time API Endpoint**

This endpoint acts as the bridge. By calling `refresh_schema()` immediately before invoking the chain, the LLM will always be aware of the exact database state at that millisecond.

### **Phase 5: The Frontend "Universal" Visualizer**

Because your graph can now return Drugs, Diagnoses, or Patients, your React UI must be smart enough to style them differently (e.g., Drugs are blue circles, Diagnoses are red squares) and display whatever edge properties exist.

---

This architecture completely separates the concerns of your team. The database modelers can experiment freely with MIMIC-IV and DDInter tables, the frontend developers can build robust mapping functions for Cytoscape.js/D3.js, and this backend pipeline will seamlessly orchestrate the translation between them.

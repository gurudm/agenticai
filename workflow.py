import json
import os

from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.graph import END, StateGraph
from typing import TypedDict


os.environ["ANTHROPIC_API_KEY"] = os.environ.get("CLAUDEKEY")

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

loader = PyPDFLoader("2026-q1-td-transcript.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
chunks = text_splitter.split_documents(pages)



vector_store = Chroma.from_documents(
    persist_directory="./quaterly_db",
    documents=chunks,
    embedding=embeddings
)


# ------ STATE ------
# This object is used to maintain the state of the analysis workflow, storing intermediate results and the current step in the process. It allows for a structured way to track the progress of the analysis and store relevant information at each stage.
# Every node reads from it and write to it.

class AnalysisState(TypedDict):
    question: str                   # Original question
    raw_context: str                # extracted context from documents
    key_metrics: str                # extracted key financial metrics
    performance_analysis: str       # performance analysis based on the context
    risk_analysis: str              # risk analysis based on the context
    final_report: str               # final report combining all analyses
    current_step: str               # current step in the analysis workflow
    risk_level: str                 # overall risk level (High/Medium/Low)
    data_sufficient: bool           # assessment of whether the data is sufficient to answer the question (Sufficient/Insufficient)

# ------ WORKFLOW NODES ------
# Each node is a plain function that takes the current state as input and retures an updated state.


def retrieve_context(state: AnalysisState):
    print(f"Step 1: Retrieving relevant context from documents")

    question = state['question']
    result = vector_store.similarity_search_with_score(question, k=7)
    relevant_result = [item for item, score in result if score < 1.5]

    if not relevant_result:
        return {
            **state,  # ← spread full state first
            "raw_context": "No relevant documents found",
            "current_step": "context_retrieved"
        }
    
    context_parts = []
    for doc in relevant_result:
        pagenum = doc.metadata.get("page", "unknown")
        context_parts.append(f"Page Number: {pagenum}, Context: {doc.page_content}")
    
    return {
        **state,
        "raw_context": '\n'.join(context_parts),
        "current_step": "context_retrieved"
    }

def extract_metrics(state: AnalysisState) -> AnalysisState:
    print("Step 2: Extracting key financial metrics from context")

    context = state['raw_context']

    # Short circuit - no point calling LLM if no context found
    if context == "No relevant documents found":
        print("No relevant context found - marking data as insufficient")
        return {
            **state,
            "key_metrics": "{}",
            "risk_level": "Medium",
            "data_sufficient": False,
            "current_step": "metrics_extracted"
        }

    prompt_text = f"""From the following banking context, extract all specific financial metrics.

        CRITICAL RULES - Set initial_risk_level to High if ANY of these are true:
        - Provision for credit losses (PCL) increasing quarter over quarter
        - Migration from performing to impaired loans mentioned
        - Impaired PCL ratio > 30 basis points
        - Gross impaired loans increasing
        - Capital ratio below 12%

        Return as clean JSON:
        {{
            "revenue": "value or null",
            "roe": "value or null",
            "efficiency_ratio": "value or null",
            "pcl_ratio": "value or null",
            "pcl_change": "increasing or decreasing or stable",
            "impaired_loans_trend": "increasing or decreasing or stable",
            "capital_ratio": "value or null",
            "other_metrics": [],
            "initial_risk_level": "High or Medium or Low",
            "data_sufficient": true
        }}

        Set data_sufficient to false if fewer than 2 metrics found.
        Return only JSON absolutely no extra text.

        Context:
        {context}"""

    print(f"\n>>> PROMPT LENGTH: {len(prompt_text)} chars")
    response = llm.invoke(prompt_text)
    print(f"\n>>> LLM METRICS RESPONSE:\n{response.content}\n")

    try:
        # Strip markdown code fences if present
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]  # get content between fences
            if content.startswith("json"):
                content = content[4:]  # remove the word "json"
        
        response_json = json.loads(content.strip())
        risk_level = response_json.get("initial_risk_level", "Medium")
        data_sufficient = response_json.get("data_sufficient", True)
    except json.JSONDecodeError:
        risk_level = "Medium"
        data_sufficient = True

    return {
        **state,
        "key_metrics": response.content,
        "risk_level": risk_level,
        "data_sufficient": data_sufficient,
        "current_step": "metrics_extracted"
    }

def deep_risk_analysis(state: AnalysisState) -> AnalysisState:
    """High risk branch: detailed risk investigation"""
    print("Step 3a: HIGH RISK detected - running deep risk analysis...")
    
    response = llm.invoke(f"""
    URGENT RISK ANALYSIS REQUIRED.
    The following banking data shows HIGH RISK indicators.
    
    Metrics: {state['key_metrics']}
    Context: {state['raw_context']}
    
    Provide an urgent detailed risk assessment:
    1. Primary risk factors driving HIGH risk rating
    2. Immediate concerns requiring attention
    3. Comparison to regulatory thresholds
    4. Recommended immediate actions
    5. Timeline for risk resolution
    
    Be specific, direct and urgent in tone.
    """)
    
    return {
        **state,
        "risk_analysis": response.content,
        "current_step": "deep_risk_completed"
    }

def standard_analysis(state: AnalysisState) -> AnalysisState:
    """Medium/Low risk branch: standard performance analysis"""
    print("Step 3b: Standard risk level - running standard analysis...")
    
    response = llm.invoke(f"""
    You are a senior banking analyst.
    Based on these metrics: {state['key_metrics']}
    And this context: {state['raw_context']}
    
    Provide a balanced performance assessment:
    - Revenue and profitability trends
    - Return on equity vs industry benchmarks
    - Operational efficiency
    - Minor risk factors to monitor
    - Overall outlook: Positive/Neutral/Cautious
    """)
    
    return {
        **state,
        "performance_analysis": response.content,
        "risk_analysis": "Risk level: Medium/Low - standard monitoring recommended",
        "current_step": "standard_analysis_completed"
    }

def insufficient_data_handler(state: AnalysisState) -> AnalysisState:
    """No data branch: handle gracefully"""
    print("Step 3c: Insufficient data found...")
    
    return {
        **state,
        "performance_analysis": "Insufficient data for analysis",
        "risk_analysis": "Cannot assess risk without sufficient data",
        "final_report": """
        ANALYSIS INCOMPLETE
        
        The system could not find sufficient financial data to answer your question.
        
        Suggestions:
        - Try rephrasing your question with specific terms
        - Ask about specific executives, metrics or time periods
        - Example: 'What did the CEO say about revenue in Q1 2026?'
        """,
        "current_step": "insufficient_data"
    }


def route_by_risk(state: AnalysisState) -> AnalysisState:
    """Determine which analysis patht to take based on the risk level and data sufficiency"""
    print(state['data_sufficient'], state['risk_level'])
    if not state['data_sufficient']:
        return "insufficient_data_handler"
    
    if state['risk_level'] == "High":
        print("High risk level detected, routing to deep risk analysis...")
        return "deep_risk_analysis"
    
    print("Medium/Low risk level detected, routing to standard analysis...")
    
    return "standard_analysis"

def route_after_analysis(state: AnalysisState) -> AnalysisState:
    """After analysis, check if we should generate a report or handle insufficient data"""

    if state.get("current_step") == "insufficient_data":
        return END  # No further action needed, report is already set 
    return "generate_report"


def generate_report(state: AnalysisState) -> AnalysisState:
    """Node 5 Generate final report combining all analyses"""
    print("Step 5: Generating final report combining all analyses")

    response = llm.invoke(f"""
    You are a senior banking analyst compiling an executive report.
    
    Original question: {state['question']}
    
    Key Metrics: {state['key_metrics']}
    Performance Analysis: {state['performance_analysis']}
    Risk Analysis: {state['risk_analysis']}
    
    Compile a concise executive summary report that:
    1. Answers the original question directly
    2. Highlights key metrics
    3. Summarizes performance
    4. Flags top risks
    5. Provides a clear recommendation
    
    Format it professionally as an executive would present it.
    """)
    
    return {
        **state,
        "final_report": response.content,
        "current_step": "report_generated"
    }


# Build the conditional GRAPH
workflow = StateGraph(AnalysisState)

# Add nodes to the graph in the order they should be executed

workflow.add_node("retrieve_context", retrieve_context)
workflow.add_node("extract_metrics", extract_metrics)
workflow.add_node("deep_risk_analysis", deep_risk_analysis)
workflow.add_node("standard_analysis", standard_analysis)
workflow.add_node("insufficient_data_handler", insufficient_data_handler)
workflow.add_node("generate_report", generate_report)

# Add edges to define the flow of the workflow
# Linear edges
workflow.set_entry_point("retrieve_context")
workflow.add_edge("retrieve_context", "extract_metrics")

# conditional routing after metrics extraction
workflow.add_conditional_edges(
    "extract_metrics", # from this node
    route_by_risk, # Call this function to determine the next node based on risk level and data sufficiency
    {               # Map the return values of route_by_risk to the corresponding nodes
        "deep_risk_analysis": "deep_risk_analysis",
        "standard_analysis": "standard_analysis",
        "insufficient_data_handler": "insufficient_data_handler"
    }
)
workflow.add_conditional_edges(
    "deep_risk_analysis",
    route_after_analysis,
    {
        "generate_report": "generate_report",
        END: END
    }
)
workflow.add_conditional_edges(
    "standard_analysis",
    route_after_analysis,
    {
        "generate_report": "generate_report",
        END: END
    }
)
workflow.add_conditional_edges(
    "insufficient_data_handler",
    route_after_analysis,
    {
        "generate_report": "generate_report",
        END: END
    }
)
workflow.add_edge("generate_report", END)

# Compile the graph to create an executable workflow
app = workflow.compile()

print("Banking analysis workflow is ready to execute. Call app with initial state containing the question.")

print("=" * 60)

question = input("Enter the question you want to ask about the bank's performance: ")

# Run the workflow with the initial state containing the question

final_state = app.invoke({
    "question": question,
    "raw_context": "",
    "key_metrics": "",
    "performance_analysis": "",
    "risk_analysis": "",
    "final_report": "",
    "current_step": "started",
    "risk_level": "Medium",
    "data_sufficient": True
})


print(f"\nPath taken: {final_state['current_step']}")
print(f"Risk level detected: {final_state['risk_level']}")
print(f"\nFinal Report:\n{final_state['final_report']}")
print("=" * 60)
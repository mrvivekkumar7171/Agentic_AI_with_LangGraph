# pip install -U langgraph langchain-openai pydantic python-dotenv langsmith

# importing libraries
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langsmith import traceable
from dotenv import load_dotenv
import operator
import string
import random
import os



# ---------- Setup ----------
load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
def generate_userid():
    # generate 4 random letters (upper + lower case allowed)
    letters = ''.join(random.choice(string.ascii_letters) for _ in range(4))
    # generate 4 random digits
    numbers = str(random.randint(1000, 9999))
    return letters + numbers
os.environ["LANGCHAIN_PROJECT"] = "Langgraph ChatBot"

# ---------- Structured schema & model ----------
class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", ge=0, le=10) # to make sure score in integer.

# to get Structured output, we have used the model gpt-4o-mini of openai that support structured output.
structured_model = model.with_structured_output(EvaluationSchema)

# ---------- Sample essay ----------
essay2 = """India and AI Time

Now world change very fast because new tech call Artificial Intel… something (AI). India also want become big in this AI thing. If work hard, India can go top. But if no careful, India go back.

India have many good. We have smart student, many engine-ear, and good IT peoples. Big company like TCS, Infosys, Wipro already use AI. Government also do program “AI for All”. It want AI in farm, doctor place, school and transport.

In farm, AI help farmer know when to put seed, when rain come, how stop bug. In health, AI help doctor see sick early. In school, AI help student learn good. Government office use AI to find bad people and work fast.

But problem come also. First is many villager no have phone or internet. So AI not help them. Second, many people lose job because AI and machine do work. Poor people get more bad.

One more big problem is privacy. AI need big big data. Who take care? India still make data rule. If no strong rule, AI do bad.

India must all people together – govern, school, company and normal people. We teach AI and make sure AI not bad. Also talk to other country and learn from them.

If India use AI good way, we become strong, help poor and make better life. But if only rich use AI, and poor no get, then big bad thing happen.

So, in short, AI time in India have many hope and many danger. We must go right road. AI must help all people, not only some. Then India grow big and world say "good job India".
"""

# ---------- LangGraph state ----------
class UPSCState(TypedDict, total=False):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    # Reducer function to add multiple input without overwritting
    individual_scores: Annotated[List[int], operator.add]  # merges parallel lists # operator.add = + 
    avg_score: float



# ---------- Defining Traced node the workflow functions ----------
# To track function inside each node of the langgraph as it is not done by langsmith by default. So this is additional step.
@traceable(name="evaluate_language_fn", tags=["dimension:language"], metadata={"dimension": "language"})
def evaluate_language(state: UPSCState):
    prompt = (
        "Evaluate the language quality of the following essay and provide feedback "
        "and assign a score out of 10.\n\n" + state["essay"]
    )
    out = structured_model.invoke(prompt)
    
    # For parallel workflow, each nodes must return the required part of the state only.
    # While, for others we can send the whole state. But, to keep it simple, we can just use required part of the state for nodes.
    return {"language_feedback": out.feedback, "individual_scores": [out.score]} # instead of return state

@traceable(name="evaluate_analysis_fn", tags=["dimension:analysis"], metadata={"dimension": "analysis"})
def evaluate_analysis(state: UPSCState):
    prompt = (
        "Evaluate the depth of analysis of the following essay and provide feedback "
        "and assign a score out of 10.\n\n" + state["essay"]
    )
    out = structured_model.invoke(prompt)
    return {"analysis_feedback": out.feedback, "individual_scores": [out.score]}

@traceable(name="evaluate_thought_fn", tags=["dimension:clarity"], metadata={"dimension": "clarity_of_thought"})
def evaluate_thought(state: UPSCState):
    prompt = (
        "Evaluate the clarity of thought of the following essay and provide feedback "
        "and assign a score out of 10.\n\n" + state["essay"]
    )
    out = structured_model.invoke(prompt)
    return {"clarity_feedback": out.feedback, "individual_scores": [out.score]}

@traceable(name="final_evaluation_fn", tags=["aggregate"])
def final_evaluation(state: UPSCState):
    # summary
    prompt = (
        "Based on the following feedback, create a summarized overall feedback.\n\n"
        f"Language feedback: {state.get('language_feedback','')}\n"
        f"Depth of analysis feedback: {state.get('analysis_feedback','')}\n"
        f"Clarity of thought feedback: {state.get('clarity_feedback','')}\n"
    )
    overall = model.invoke(prompt).content
    scores = state.get("individual_scores", []) or []
    # average feedback
    avg = (sum(scores) / len(scores)) if scores else 0.0
    return {"overall_feedback": overall, "avg_score": avg}



# ---------- Build graph ----------
graph = StateGraph(UPSCState)

# nodes
graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)

# nodes : Fan-out → join
graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")
graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")
graph.add_edge("final_evaluation", END)

workflow = graph.compile()



# ---------- Direct invoke (running the workflow) without wrapper ----------
if __name__ == "__main__":
    
    result = workflow.invoke(
        {"essay": essay2},
        config={
            "run_name": f'{generate_userid()}',  # becomes root run name
            "tags": ["essay", "langgraph", "evaluation"],
            "metadata": {
                "essay_length": len(essay2),
                "model": "gpt-4o-mini",
                "dimensions": ["language", "analysis", "clarity"],
            },
        },
    )

    print("\n=== Evaluation Results ===")
    print("Language feedback:\n", result.get("language_feedback", ""), "\n")
    print("Analysis feedback:\n", result.get("analysis_feedback", ""), "\n")
    print("Clarity feedback:\n", result.get("clarity_feedback", ""), "\n")
    print("Overall feedback:\n", result.get("overall_feedback", ""), "\n")
    print("Individual scores:", result.get("individual_scores", []))
    print("Average score:", result.get("avg_score", 0.0))
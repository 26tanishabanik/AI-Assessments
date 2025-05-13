# agents.py - Defines the different agents for the assessment bot
import os
from prompts import (
    CANDIDATE_DETAILS_PROMPT, TAILOR_MCQ_PROMPT, EVALUATION_PROMPT, REPORT_PROMPT,
    TAILOR_OPEN_ENDED_QUESTION_GENERATION_PROMPT, OPEN_ENDED_EVALUATION_PROMPT,
    REFERENCE_BASED_EVALUATION_PROMPT  # Import new prompt
)
import utils # Import the module itself to access its globals

# Agno Core Imports
from agno.agent import Agent
from agno.team import Team # For orchestrating multiple agents
from agno.models.google import Gemini # Import Agno's Gemini model

# Ensure GOOGLE_API_KEY is configured for Agno's Gemini model usage.
# This might be done once globally, e.g., in main.py before any Agno agent init.
# Alternatively, Agno's Gemini class might take an api_key parameter or read env var itself.
# For now, we rely on GOOGLE_API_KEY being in the environment.

# Define the Gemini model instance to be used by agents
# Ensure this model ID is available to you via the API key.
AGNO_GEMINI_MODEL_ID = os.getenv("AGNO_GEMINI_MODEL_ID", "models/gemini-1.5-flash")
# Reference-based evaluation prompt
REFERENCE_BASED_EVALUATION_PROMPT = """
You are an expert tailoring instructor evaluating a candidate's answer.

**Task:** Assess the candidate's answer based *only* on the provided Reference Answer and the Question Asked.

**Input:**
1.  **Reference Answer:** The ideal answer.
2.  **Question Asked:** The question the candidate answered.
3.  **Candidate's Answer:** The answer provided by the candidate.
4.  **Is Correct (90% Threshold):** A pre-calculated boolean indicating if the answer meets a semantic similarity threshold.

**Output Format:**
Provide your evaluation in two parts:

1.  **Assessment:** Start with "Assessment:". Give a brief (1-2 sentence) qualitative judgment of the candidate's understanding *based on the reference answer*. Examples: "Excellent understanding demonstrated.", "Good grasp of the core concepts.", "Shows partial understanding but missed key details.", "Answer is incorrect or irrelevant."
2.  **Explanation:** Start with "Explanation:". Elaborate (2-4 sentences) on *why* you gave that assessment. Compare the candidate's answer to the reference. Mention specific points they got right or wrong *according to the reference*. If the `Is Correct` flag is false, explain the key differences or omissions.

**Important:**
*   Base your evaluation *solely* on the provided Reference Answer. Do not use external knowledge.
*   Focus on the *content* and *accuracy* of the answer compared to the reference.
*   Do not assign a grade; the system handles grading separately based on the correctness threshold.
*   Be objective and provide constructive feedback in the explanation.
"""

# --- CandidateDetailsAgent (Agno Version) ---
candidate_details_agent_instructions = [
    "You are a friendly interviewer conducting an initial screening.",
    "Ask the candidate for ONLY ONE piece of information at a time: full name, then years of experience, then specializations.",
    "Wait for the candidate's response before asking the next question.",
    "Extract the requested information accurately from the candidate's response.",
    "If the candidate provides extra information, acknowledge it briefly but focus on the specific detail you asked for.",
    "Output ONLY the extracted information (e.g., just the name, just the years, just the specialization list)."
]

candidate_details_agno_agent = Agent(
    name="CandidateDetailsAgent",
    role="Collects candidate details (name, experience, specializations) one by one.", # Updated role
    model=Gemini(id=AGNO_GEMINI_MODEL_ID),
    instructions=candidate_details_agent_instructions,
    markdown=True
)


# --- TailorOpenEndedQuestionAgent (Agno Version - KB REMOVED) ---
# This agent is currently NOT USED by app.py, which uses the CSV directly.
# Kept for potential future use, but knowledge base dependency removed.
tailor_question_generation_agent_instructions = [
    "You are an expert in tailoring.",
    # Removed instructions related to searching knowledge base
    "Given a tailoring topic, formulate one clear, open-ended question that can assess a candidate's understanding of that topic.",
    "The question should require more than a yes/no answer.",
    "Output ONLY the generated question."
]

tailor_question_generation_agno_agent = Agent(
    name="TailorQuestionGeneratorAgent",
    role="Generates an open-ended tailoring question based on a given topic.", # Updated role
    model=Gemini(id=AGNO_GEMINI_MODEL_ID),
    # knowledge=utils.get_agno_knowledge_base(), # REMOVED KNOWLEDGE BASE
    instructions=tailor_question_generation_agent_instructions,
    markdown=True
)


# --- EvaluationAgent (Agno Version) ---
evaluation_agent_instructions = [
    REFERENCE_BASED_EVALUATION_PROMPT,
    "Your role is to provide a qualitative assessment of the candidate's answer, not to determine the grade.",
    "The system will automatically assign the grade based on whether the answer meets the 90% correctness threshold.",
    "Focus on explaining why the answer is good or where it falls short compared to the reference answer."
]

evaluation_agno_agent = Agent(
    name="EvaluationAgent",
    role="Evaluates a candidate's answer against a reference answer and provides qualitative assessment.", # Updated role
    model=Gemini(id=AGNO_GEMINI_MODEL_ID),
    instructions=evaluation_agent_instructions,
    markdown=True
)


# --- ReportAgent (Agno Version) ---
report_agent_instructions = [
    "You are an AI assistant tasked with generating a concise assessment report for a tailoring candidate.",
    "You will receive a summary including candidate details, an overall grade, and a performance breakdown for each question (question, candidate answer, reference answer, correctness status, final grade, and qualitative assessment).",
    "Generate a professional report in Markdown format.",
    "The report should include:",
    "  - Candidate's Name, Experience, and Specializations (if provided).",
    "  - The Overall Grade, prominently displayed.",
    "  - A 'Question Performance Summary' section.",
    "    - For each question, briefly mention the topic/question, the candidate's final grade for it, and a very brief summary (1 sentence) of the qualitative assessment provided.",
    "  - A concluding 'Recommendations' section (1-2 sentences) based on the overall performance (e.g., 'Strong candidate, recommend proceeding.', 'Solid skills, potential for growth in [area].', 'Needs improvement in [area].').",
    "Keep the report clear, concise, and professional."
]

report_agno_agent = Agent(
    name="ReportAgent",
    role="Generates a final assessment report summarizing candidate performance.", # Added role
    model=Gemini(id=AGNO_GEMINI_MODEL_ID),
    instructions=report_agent_instructions,
    markdown=True
)

# The old get_gemini_model(), start_chat_session() and AgnoGeminiModelWrapper are no longer needed
# as Agno's Gemini class handles model interaction directly.
# Our global _chat_session is also superseded by Agno agent's internal state management if we fully use Agno's .run() or .print_response().

# --- Orchestration Logic (to be moved/used in main.py) ---
# The old `run_..._agent` functions will be replaced by calls to these Agno agents.
# For example, main.py will now do something like:
# candidate_details_output_str = candidate_details_agno_agent.run("Initiate candidate interaction.")
# Then, parse candidate_details_output_str to get the structured info.
# This part needs careful thought on how to pass data between agent steps if not using a formal Agno Team for full orchestration. 
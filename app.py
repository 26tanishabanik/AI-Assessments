import streamlit as st
import os
import utils
import google.generativeai as genai
from textwrap import dedent
import re
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import logging
import traceback
import asyncio
import sys

from agents import (
    evaluation_agno_agent,
    report_agno_agent
)


st.set_page_config(
    page_title="Tailoring Assessment Bot",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create a logger for this module
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*Tried to instantiate class '__path__._path'.*")
# warnings.filterwarnings("ignore", category=RuntimeWarning)
# warnings.filterwarnings("ignore", category=UserWarning)

# Patch asyncio for Streamlit
if sys.platform == 'darwin':
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        logger.warning("nest_asyncio not installed. Some async features might not work properly.")
        pass


st.markdown("""
<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.stApp {
    max-width: 800px; /* Consistent width for chat-like feel */
    margin: 0 auto;
    background-color: #0e1117; /* Darker background for the app */
}
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 10px; /* Space between messages */
    padding-bottom: 150px; /* Adjusted for potentially taller input area */
}
.chat-message {
    padding: 0.75rem 1rem;
    border-radius: 18px; /* More rounded bubbles */
    margin-bottom: 0.5rem;
    display: inline-block; /* Allow messages to not take full width */
    max-width: 75%; /* Max width of a message bubble */
    word-wrap: break-word; /* Break long words */
    line-height: 1.4;
}
.chat-message.user {
    background-color: #007bff; /* ChatGPT blue for user */
    color: white;
    align-self: flex-end; /* User messages on the right */
    border-bottom-right-radius: 5px; /* Slightly different rounding for user */
}
.chat-message.bot {
    background-color: #343a40; /* Darker grey for bot */
    color: white;
    align-self: flex-start; /* Bot messages on the left */
    border-bottom-left-radius: 5px; /* Slightly different rounding for bot */
}

.chat-input-form-container { /* This is the fixed bar at the bottom */
    position: fixed; 
    bottom: 0;
    left: 0;
    right: 0;
    background-color: #0e1117; 
    border-top: 1px solid #303030;
    z-index: 1000;
    box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
}
.chat-input-form { /* This div is inside the container, for centering form elements */
    max-width: 800px;
    margin: 0 auto; 
    padding: 0.75rem 1rem; /* Adjusted padding */
}

.stTextArea textarea {
    background-color: #2b313e;
    color: white;
    border-radius: 18px;
    border: 1px solid #343a40;
    /* min-height is now controlled by Streamlit's default or the height param */
    max-height: 150px; 
    padding: 0.75rem 1rem;
}
.stButton>button {
    background-color: #007bff;
    color: white;
    border: none;
    border-radius: 18px;
    padding: 0.6rem 1rem;
    height: 100%; /* Make button same height as text area if in columns */
    width: 100%; /* Make button take full width of its column */
}
.evaluation-container { /* Keep previous styling for evaluation */
    background-color: #1e1e1e;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state with error handling and new flow variables."""
    try:
        # if 'step' not in st.session_state: # Deprecating old step logic
        #     st.session_state.step = 1
        if 'collected_details' not in st.session_state:
            st.session_state.collected_details = {
                "full_name": None,
                # "years_of_experience": None, # Removed
                # "specializations": None      # Removed
            }
        if 'asked_questions' not in st.session_state: # Will store all Qs asked across grades
            st.session_state.asked_questions = []
        if 'current_question_idx' not in st.session_state: # Re-purposing for overall question count if needed, or specific to grade.
            st.session_state.current_question_idx = 0
        if 'all_questions' not in st.session_state: # This might be deprecated if we fetch per grade
            st.session_state.all_questions = []
        if 'evaluations' not in st.session_state: # Stores individual LLM evaluations
            st.session_state.evaluations = []
        # if 'overall_grade' not in st.session_state: # This will be determined by last_correct_grade
        #     st.session_state.overall_grade = None
        if 'final_report' not in st.session_state:
            st.session_state.final_report = None
        if 'api_key_configured' not in st.session_state:
            st.session_state.api_key_configured = False
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # New flow stage and variables for sequential grade-based assessment
        if 'flow_stage' not in st.session_state:
            st.session_state.flow_stage = 'welcome_config_api' # Initial stage
        
        # Removed old adaptive flow variables:
        # manuf_exp_answer, machine_knowledge_answer, known_machines_list, job_task_questions_asked_count, max_job_task_questions

        # New state for simplified sequential flow
        if 'grade_order' not in st.session_state:
            st.session_state.grade_order = ['C', 'B', 'B+', 'A', 'A*'] # Define grade progression
        if 'current_grade_level_idx' not in st.session_state: # Index for st.session_state.grade_order
             st.session_state.current_grade_level_idx = 0 
        if 'current_grade_being_tested' not in st.session_state:
             st.session_state.current_grade_being_tested = st.session_state.grade_order[0] # Start with 'C'
        if 'last_correct_grade' not in st.session_state:
            st.session_state.last_correct_grade = None 
        if 'questions_for_current_grade' not in st.session_state:
            st.session_state.questions_for_current_grade = []
        if 'question_index_within_grade' not in st.session_state:
            st.session_state.question_index_within_grade = 0
        if 'max_questions_per_grade' not in st.session_state:
            st.session_state.max_questions_per_grade = 1 # Ask 1 question per grade level for now

        if 'is_evaluating_answer' not in st.session_state: # New state for UI feedback
            st.session_state.is_evaluating_answer = False

        if 'last_bot_message_id' not in st.session_state: # To help replace temporary messages
            st.session_state.last_bot_message_id = None 

        env_api_key = os.getenv("GOOGLE_API_KEY")
        if env_api_key and not st.session_state.api_key_configured:
            configure_api_key(env_api_key)
    except Exception as e:
        logger.error(f"Error initializing session state: {str(e)}\n{traceback.format_exc()}")
        st.error("An error occurred while initializing the application. Please refresh the page.")

def log_error(error_msg, error_traceback=None):
    """Enhanced error logging with more context"""
    try:
        logger.error(f"{error_msg}\n{error_traceback if error_traceback else ''}")
        st.error("An error occurred. Please try again or contact support if the issue persists.")
    except Exception as e:
        print(f"Critical error in log_error: {str(e)}")

def display_chat_message(message, is_user=False):
    """Display a chat message with appropriate styling"""
    user_class = "user" if is_user else "bot"
    if is_user:
        st.markdown(f"<div style='display: flex; justify-content: flex-end;'><div class='chat-message {user_class}'> {message}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='display: flex; justify-content: flex-start;'><div class='chat-message {user_class}'>{message}</div></div>", unsafe_allow_html=True)

def display_evaluation(eval_item, index):
    """Display evaluation results in a cleaner format"""
    with st.container():
        st.markdown(f"""
        <div class="evaluation-container">
            <h4>Question {index + 1}</h4>
            <p><strong>Your Answer:</strong> {eval_item['candidate_answer']}</p>
            <p><strong>Status:</strong> {'✅ CORRECT' if eval_item['is_correct'] else '❌ INCORRECT'}</p>
            <p><strong>Grade:</strong> {eval_item['final_grade']}</p>
            <p><strong>Assessment:</strong> {eval_item['llm_assessment']}</p>
        </div>
        """, unsafe_allow_html=True)


def configure_api_key(api_key):
    try:
        genai.configure(api_key=api_key)
        st.session_state.api_key_configured = True
        logger.info("Google API Key configured successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to configure Google API Key: {e}")
        st.error(f"Failed to configure Google API Key: {e}")
        return False


def parse_evaluation_response(agent_response_content):
    assessment = "Could not parse assessment."
    explanation = agent_response_content
    try:
        assessment_match = re.search(r"Assessment:(.*?)(?:Explanation:|$)", agent_response_content, re.IGNORECASE | re.DOTALL)
        if assessment_match:
            assessment = assessment_match.group(1).strip()
        
        explanation_match = re.search(r"Explanation:(.*)", agent_response_content, re.IGNORECASE | re.DOTALL)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        elif assessment_match and assessment != "Could not parse assessment.":
            explanation = "Assessment provided without separate explanation label."
    except Exception as e:
        st.error(f"Could not parse evaluation response: {e}")
    return {"llm_assessment": assessment, "llm_explanation": explanation}


def check_answer_correctness(candidate_answer, reference_answer):
    try:
        embedder = utils.get_embedder_instance()
        if embedder:
            candidate_embedding = embedder.encode([candidate_answer], show_progress_bar=False)[0]
            reference_embedding = embedder.encode([reference_answer], show_progress_bar=False)[0]
            
            similarity = cosine_similarity([candidate_embedding], [reference_embedding])[0][0]
            
            is_correct = similarity > 0.9
            return is_correct, similarity
    except Exception as e:
        st.warning(f"Error during similarity calculation: {e}")
        st.info("Falling back to keyword matching...")
    
    try:
        candidate_answer = candidate_answer.lower()
        reference_answer = reference_answer.lower()
        
        ref_keywords = reference_answer.split()
        
        match_count = sum(1 for kw in ref_keywords if kw in candidate_answer)
        match_score = match_count / len(ref_keywords) if ref_keywords else 0
        
        is_correct = match_score > 0.9
        return is_correct, match_score
    except Exception as e:
        st.error(f"Error during keyword matching: {e}")
    
    return False, 0.0


def generate_report():
    if not st.session_state.evaluations:
        st.error("No evaluations available to generate a report.")
        return None
    
    grade_map = {"A*": 5, "A": 4, "B+": 3, "B": 2, "C": 1}
    total_points = sum(grade_map.get(eval_item['final_grade'], 1) for eval_item in st.session_state.evaluations)
    avg_points = total_points / len(st.session_state.evaluations)
    
    if avg_points >= 4.5:
        overall_grade = "A*"
    elif avg_points >= 3.5:
        overall_grade = "A"
    elif avg_points >= 2.5:
        overall_grade = "B+"
    elif avg_points >= 1.5:
        overall_grade = "B"
    else:
        overall_grade = "C"
    
    st.session_state.overall_grade = overall_grade
    
    report_input_summary = f"Candidate Details:\nFull Name: {st.session_state.collected_details.get('full_name', 'N/A')}" #Experience: {st.session_state.collected_details.get('years_of_experience', 'N/A')}\nSpecializations: {st.session_state.collected_details.get('specializations', 'N/A')}\n\n"
    report_input_summary += f"Achieved Grade Level: {st.session_state.get('last_correct_grade', 'Not Assessed')}\n\n"
    report_input_summary += "Question Performance Summary:\n"
    
    for i, eval_item in enumerate(st.session_state.evaluations):
        correctness_status = "✓ CORRECT" if eval_item.get('is_correct', False) else "✗ INCORRECT"
        report_input_summary += (
            f"  Q{i+1}: {eval_item['generated_question']}\n"
            f"    Candidate's Answer: {eval_item['candidate_answer']}\n"
            f"    Reference Answer: {eval_item['reference_answer']}\n"
            f"    Status: {correctness_status}\n"
            f"    Final Grade: {eval_item['final_grade']} (Original question grade: {eval_item['original_grade']})\n"
            f"    Assessment: {eval_item['llm_assessment']}\n\n"
        )
    
    progression_feedback_prompt = "Based on the assessment summary, provide brief feedback on how the candidate can progress and their future earning potential."
    final_report_prompt = f"Based on the following information, generate the assessment report. Make sure the grade is prominently featured in the report. Include any insights on progression or earning potential if possible, otherwise focus on the assessment summary:\n\n{report_input_summary}\n\n{progression_feedback_prompt}"

    try:
        final_report_response = report_agno_agent.run(final_report_prompt)
        return final_report_response.content
    except Exception as e:
        st.error(f"Error generating report: {e}")
        return None


def app():
    """Main application function with new adaptive flow."""
    try:
        initialize_session_state()
        
        st.title("Tailoring Skills Assessment Bot (Simplified Flow)")
        
        if st.session_state.flow_stage == 'welcome_config_api':
            st.write("Welcome to the Tailoring Skills Assessment Bot. This tool evaluates tailoring candidates based on their answers to industry-specific questions.")
            
            with st.container():
                st.subheader("Setup")
                
                if st.session_state.api_key_configured:
                    # st.success("Google API Key configured.")
                    api_key_display = "Already configured"
                else:
                    api_key_display = st.text_input("Enter your Google API Key:", type="password", 
                                           help="Or set GOOGLE_API_KEY environment variable")
                
                if st.button("Start Assessment"):
                    proceed_with_assessment = True
                    if not st.session_state.api_key_configured:
                        if api_key_display and api_key_display != "Already configured":
                            proceed_with_assessment = configure_api_key(api_key_display)
                        else:
                            st.error("Please enter a valid Google API Key or set GOOGLE_API_KEY environment variable.")
                            proceed_with_assessment = False
                    
                    if proceed_with_assessment:
                        try:
                            with st.spinner("Loading assessment data..."):
                                csv_loaded = utils.load_grading_data()
                                if csv_loaded is None or csv_loaded.empty:
                                    st.error("Failed to load the CSV file with questions or it's empty. Please check app.log and CSV path in utils.py.")
                                    proceed_with_assessment = False
                        except Exception as e:
                            log_error("Error loading grading data", traceback.format_exc())
                            proceed_with_assessment = False
                        
                        if proceed_with_assessment:
                            st.session_state.flow_stage = 'collect_candidate_name' # Changed from collect_candidate_details
                            st.rerun()
        
        elif st.session_state.flow_stage == 'collect_candidate_name': # Changed from collect_candidate_details
            st.subheader("Candidate Details")
            
            full_name = st.text_input("Full Name:", value=st.session_state.collected_details.get("full_name", ""))
            # years_experience = st.text_input("Years of Experience in Tailoring:", value=st.session_state.collected_details.get("years_of_experience", "")) # Removed
            # specializations = st.text_area("Specializations in Tailoring:", value=st.session_state.collected_details.get("specializations", "")) # Removed
            
            if st.button("Start Assessment"): # Changed button text
                if full_name:
                    st.session_state.collected_details["full_name"] = full_name
                    # st.session_state.collected_details["years_of_experience"] = years_experience # Removed
                    # st.session_state.collected_details["specializations"] = specializations # Removed
                    
                    # Initialize for the new sequential flow
                    st.session_state.current_grade_level_idx = 0
                    st.session_state.current_grade_being_tested = st.session_state.grade_order[0]
                    st.session_state.last_correct_grade = None
                    st.session_state.questions_for_current_grade = []
                    st.session_state.question_index_within_grade = 0
                    st.session_state.asked_questions = [] # Clear any previous session's questions
                    st.session_state.evaluations = []   # Clear any previous evaluations
                    st.session_state.chat_history = []  # Clear chat history

                    st.session_state.flow_stage = 'sequential_assessment_qna' # New Q&A stage
                    st.rerun()
                else:
                    st.error("Please enter your full name.")
        
        elif st.session_state.flow_stage == 'sequential_assessment_qna':
            st.subheader(f"Assessment: Grade {st.session_state.current_grade_being_tested} Questions")

            # --- 1. Load Questions for Current Grade ---
            if not st.session_state.questions_for_current_grade:
                with st.spinner(f"Loading questions for Grade {st.session_state.current_grade_being_tested}..."):
                    logger.info(f"Fetching questions for grade: {st.session_state.current_grade_being_tested}")
                    st.session_state.questions_for_current_grade = utils.get_questions_from_csv(
                        num_questions=st.session_state.max_questions_per_grade,
                        grades=[st.session_state.current_grade_being_tested] 
                        # Removed is_entry_level and machine_types as they are not part of this simplified flow's Q selection
                    )
                    st.session_state.question_index_within_grade = 0 # Reset index for new set of questions

                    if not st.session_state.questions_for_current_grade:
                        logger.warning(f"No questions found for grade {st.session_state.current_grade_being_tested}. Ending assessment.")
                        st.warning(f"No questions available for Grade {st.session_state.current_grade_being_tested}. Proceeding to evaluation overview.")
                        st.session_state.flow_stage = 'evaluation_overview'
                        st.rerun()
                        return # Stop further processing in this run

            # --- 2. Display Chat History ---
            st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
            for msg_idx, msg_data in enumerate(st.session_state.chat_history):
                display_chat_message(msg_data["content"], msg_data["is_user"])
            st.markdown("</div>", unsafe_allow_html=True)

            # Evaluation occurs here, triggered by is_evaluating_answer flag
            if st.session_state.get('is_evaluating_answer', False):
                # This block runs *after* the user has submitted an answer, a temporary bot message was added, and a rerun was triggered.
                
                # Retrieve the question that was just answered.
                if not st.session_state.questions_for_current_grade or \
                    not (0 <= st.session_state.question_index_within_grade < len(st.session_state.questions_for_current_grade)):
                    logger.error("State inconsistency: Trying to evaluate but current question data is missing or index is out of bounds.")
                    st.session_state.is_evaluating_answer = False
                    st.session_state.flow_stage = 'evaluation_overview' # Safety exit
                    st.rerun()
                    return

                current_q_data = st.session_state.questions_for_current_grade[st.session_state.question_index_within_grade]
                
                # Retrieve the last user answer from chat_history
                last_user_answer = ""
                # Find the most recent user message that isn't the temporary bot message
                for i in range(len(st.session_state.chat_history) - 1, -1, -1):
                    if st.session_state.chat_history[i]["is_user"]:
                        last_user_answer = st.session_state.chat_history[i]["content"]
                        break
                
                if not last_user_answer:
                    logger.error("Could not retrieve last user answer for evaluation.")
                    st.session_state.is_evaluating_answer = False 
                    st.rerun() 
                    return

                question_text_key = "Situational questions" 
                if question_text_key not in current_q_data:
                    question_text_key = "Situational questions "
                
                question_text_content = current_q_data.get(question_text_key, 'Error: Question text not found.')
                reference_answer = current_q_data.get("Answers", "Error: Reference answer not found.")
                
                _, similarity_score = check_answer_correctness(last_user_answer, reference_answer)
                PROGRESSION_THRESHOLD = 0.6 
                is_correct_for_progression = similarity_score >= PROGRESSION_THRESHOLD

                logger.info(f"Q: {question_text_content}, Answer: {last_user_answer}, Ref: {reference_answer}, Similarity: {similarity_score}, CorrectForProg: {is_correct_for_progression}")

                st.session_state.asked_questions.append({
                    "generated_question": question_text_content,
                    "candidate_answer": last_user_answer,
                    "reference_answer": reference_answer,
                    "original_grade": current_q_data.get("GRADE", st.session_state.current_grade_being_tested),
                    "machine": current_q_data.get("Machine ", ""),
                    "material": current_q_data.get("Material ", ""),
                    "job": current_q_data.get("Job ", ""),
                    "product": current_q_data.get("Product", ""),
                    "similarity_score": similarity_score,
                    "is_correct_for_progression": is_correct_for_progression
                })

                # Remove the temporary "Checking..." message
                if st.session_state.chat_history and st.session_state.chat_history[-1].get("is_temporary_status"):
                    st.session_state.chat_history.pop()

                # Determine next bot message and state transitions
                if is_correct_for_progression:
                    st.session_state.last_correct_grade = st.session_state.current_grade_being_tested
                    st.session_state.question_index_within_grade += 1 

                    if st.session_state.question_index_within_grade >= st.session_state.max_questions_per_grade or \
                        st.session_state.question_index_within_grade >= len(st.session_state.questions_for_current_grade):
                        st.session_state.current_grade_level_idx += 1
                        if st.session_state.current_grade_level_idx < len(st.session_state.grade_order):
                            next_grade_name = st.session_state.grade_order[st.session_state.current_grade_level_idx]
                            st.session_state.chat_history.append({"role": "assistant", "content": f"That's correct! Moving to Grade {next_grade_name} questions.", "is_user": False})
                            st.session_state.current_grade_being_tested = next_grade_name
                            st.session_state.questions_for_current_grade = [] 
                            st.session_state.question_index_within_grade = 0
                        else:
                            st.session_state.chat_history.append({"role": "assistant", "content": "Excellent! You've completed all available grade levels. Proceeding to evaluation overview.", "is_user": False})
                            st.session_state.flow_stage = 'evaluation_overview' # New stage
                else: 
                    st.session_state.chat_history.append({"role": "assistant", "content": "Okay, let's proceed to the evaluation overview.", "is_user": False})
                    st.session_state.flow_stage = 'evaluation_overview' # New stage
            
                st.session_state.is_evaluating_answer = False 
                st.rerun() # Rerun to display bot's actual feedback & then next question or summary
            
            # Display question and input form if not evaluating and not yet moved to summary
            if not st.session_state.get('is_evaluating_answer', False) and st.session_state.flow_stage == 'sequential_assessment_qna':
                
                # Load Questions for Current Grade (if list is empty for the current grade)
                if not st.session_state.questions_for_current_grade:
                    with st.spinner(f"Loading questions for Grade {st.session_state.current_grade_being_tested}..."): # Spinner can be here for loading Qs
                        logger.info(f"Fetching questions for grade: {st.session_state.current_grade_being_tested} (in display logic)")
                        st.session_state.questions_for_current_grade = utils.get_questions_from_csv(
                            num_questions=st.session_state.max_questions_per_grade,
                            grades=[st.session_state.current_grade_being_tested]
                        )
                        # question_index_within_grade should be 0 if we just loaded for a new grade

                        if not st.session_state.questions_for_current_grade:
                            logger.warning(f"No questions found for grade {st.session_state.current_grade_being_tested} during Q display phase. Ending assessment.")
                            if not (st.session_state.chat_history and "No questions available" in st.session_state.chat_history[-1]["content"]):
                                st.session_state.chat_history.append({"role":"assistant", "content":f"No questions available for Grade {st.session_state.current_grade_being_tested}. We'll proceed to the summary based on answers so far.", "is_user":False})
                            st.session_state.flow_stage = 'evaluation_overview' # Changed to new stage
                            st.rerun()
                            return
                
                # Display Current Question and Get Input
                if st.session_state.questions_for_current_grade and \
                   0 <= st.session_state.question_index_within_grade < len(st.session_state.questions_for_current_grade):
                    
                    current_q_data = st.session_state.questions_for_current_grade[st.session_state.question_index_within_grade]
                    question_text_key = "Situational questions"
                    if question_text_key not in current_q_data:
                        question_text_key = "Situational questions " 
                    
                    question_text_display = f"Question for Grade {st.session_state.current_grade_being_tested} ({st.session_state.question_index_within_grade + 1}/{len(st.session_state.questions_for_current_grade)}): {current_q_data.get(question_text_key, 'Error: Question text not found.')}"

                    # Add question to chat history only if it's not already the last message (to avoid duplicates after reruns)
                    if not st.session_state.chat_history or st.session_state.chat_history[-1]["content"] != question_text_display:
                        # Also ensure the last message wasn't the temporary status one if history is short
                        is_last_temp = st.session_state.chat_history and st.session_state.chat_history[-1].get("is_temporary_status")
                        if not is_last_temp:
                            st.session_state.chat_history.append({"role": "assistant", "content": question_text_display, "is_user": False})
                            st.rerun() 

                    # Chat input form
                    st.markdown("<div class='chat-input-form-container'>", unsafe_allow_html=True)
                    form_key = f"answer_form_{st.session_state.current_grade_being_tested}_{st.session_state.question_index_within_grade}"
                    with st.form(key=form_key, clear_on_submit=True):
                        st.markdown("<div class='chat-input-form'>", unsafe_allow_html=True)
                        cols = st.columns([4, 1])
                        with cols[0]:
                            candidate_answer_input = st.text_area(
                                "Your Answer:",
                                key=f"answer_input_{form_key}",
                                height=70,
                                placeholder="Type your answer here..."
                            )
                        with cols[1]:
                            submit_button = st.form_submit_button("Send")
                        st.markdown("</div>", unsafe_allow_html=True) 
                    st.markdown("</div>", unsafe_allow_html=True) 

                    if submit_button and candidate_answer_input.strip():
                        st.session_state.chat_history.append({"role": "user", "content": candidate_answer_input, "is_user": True})
                        # Add temporary bot message for feedback
                        st.session_state.chat_history.append({"role": "assistant", "content": "<em>Checking your answer...</em>", "is_user": False, "is_temporary_status": True})
                        st.session_state.is_evaluating_answer = True 
                        st.rerun()
                
                elif st.session_state.flow_stage == 'sequential_assessment_qna': 
                    logger.info("In sequential_assessment_qna, but no current question to display or issue with question indexing. Moving to summary.")
                    if not (st.session_state.chat_history and "Proceeding to summary" in st.session_state.chat_history[-1]["content"]):
                         st.session_state.chat_history.append({"role":"assistant", "content":"There are no more questions for this assessment path. Proceeding to evaluation overview.", "is_user":False})
                    st.session_state.flow_stage = 'evaluation_overview' # Changed to new stage
                    st.rerun()
            
            # Debug information (can be removed for production)
            # st.write(f"DEBUG: is_evaluating_answer: {st.session_state.get('is_evaluating_answer', False)}")

            # Safety break: If stuck or too many questions, provide an out (though flow should handle it)
            if len(st.session_state.asked_questions) > 10: # Arbitrary limit
                 if st.button("DEV: Force End Q&A and go to Summary (Safety)"):
                    st.session_state.flow_stage = 'evaluation_overview'
                    st.rerun()
        
        # This stage will be refactored into evaluation_overview, perform_llm_evaluation, and display_final_report
        # For now, let's create the new stages and comment out/remove assessment_summary_report's direct logic
        elif st.session_state.flow_stage == 'evaluation_overview':
            st.subheader("Assessment Q&A Complete")
            final_achieved_grade = st.session_state.get('last_correct_grade', "Not Assessed / Below C")
            st.metric("Achieved Grade Level", final_achieved_grade)

            if not st.session_state.asked_questions:
                st.warning("No questions were answered in this session.")
                if st.button("Start New Assessment"):
                    # Simplified reset for now, full reset is in display_final_report
                    keys_to_clear = ['flow_stage', 'collected_details', 'asked_questions', 'current_question_idx', 'all_questions', 'evaluations', 'final_report', 'chat_history', 'current_grade_level_idx', 'current_grade_being_tested', 'last_correct_grade', 'questions_for_current_grade', 'question_index_within_grade']
                    for key in keys_to_clear:
                        if key in st.session_state: del st.session_state[key]
                    initialize_session_state()
                    st.rerun()
                return # Stop further processing

            # Display evaluations if already done, otherwise button to trigger them
            if st.session_state.evaluations:
                st.markdown("### Detailed Feedback on Answers")
                for i, eval_item in enumerate(st.session_state.evaluations):
                    display_evaluation(eval_item, i)
                
                if st.button("Generate Final Summary Report"):
                    st.session_state.flow_stage = 'display_final_report'
                    st.rerun()
            else:
                if st.button("Get Detailed Feedback & Evaluation"):
                    st.session_state.flow_stage = 'perform_llm_evaluation'
                    st.rerun()
            
            if st.button("Start New Assessment (from Overview)"):
                keys_to_clear = ['flow_stage', 'collected_details', 'asked_questions', 'current_question_idx', 'all_questions', 'evaluations', 'final_report', 'chat_history', 'current_grade_level_idx', 'current_grade_being_tested', 'last_correct_grade', 'questions_for_current_grade', 'question_index_within_grade']
                for key in keys_to_clear:
                    if key in st.session_state: del st.session_state[key]
                initialize_session_state()
                st.rerun()

        elif st.session_state.flow_stage == 'perform_llm_evaluation':
            st.subheader("Processing Feedback")
            if not st.session_state.evaluations: # Only run if not already evaluated
                with st.spinner("Compiling detailed feedback on your answers..."):
                    if not st.session_state.asked_questions:
                        st.error("No questions were asked, cannot perform evaluation.")
                        st.session_state.flow_stage = 'evaluation_overview' # Go back
                        st.rerun()
                        return

                    for item in st.session_state.asked_questions: 
                        try:
                            eval_input_text = dedent(f"""
                            Reference Answer: {item['reference_answer']}
                            Question Asked: {item['generated_question']}
                            Candidate's Answer: {item['candidate_answer']}
                            Expected Grade for this question: {item['original_grade']} 
                            Candidate's recorded correctness for this question: {'Correct' if item.get('is_correct_for_progression') else 'Incorrect'}
                            """) 
                            
                            evaluation_response = evaluation_agno_agent.run(eval_input_text)
                            parsed_eval = parse_evaluation_response(evaluation_response.content)
                            
                            st.session_state.evaluations.append({
                                "machine": item.get("machine", ""),
                                "material": item.get("material", ""),
                                "job": item.get("job", ""),
                                "product": item.get("product", ""),
                                "generated_question": item["generated_question"],
                                "candidate_answer": item["candidate_answer"],
                                "reference_answer": item["reference_answer"],
                                "similarity_score": item.get("similarity_score", 0.0), 
                                "llm_assessment": parsed_eval["llm_assessment"],
                                "llm_explanation": parsed_eval["llm_explanation"],
                                "is_correct": item.get('is_correct_for_progression', False), 
                                "original_grade": item["original_grade"], 
                                "final_grade": item["original_grade"] 
                            })
                        except Exception as e:
                            log_error(f"Error generating LLM evaluation for Q: {item.get('generated_question', 'Unknown Q')}", traceback.format_exc())
                
                logger.info("LLM Evaluations complete.")
                st.session_state.flow_stage = 'evaluation_overview' # Go back to overview to display them
                st.rerun()
            else:
                # Evaluations already exist, so just go to overview to display them / offer report generation
                logger.info("Evaluations already exist, redirecting to evaluation_overview.")
                st.session_state.flow_stage = 'evaluation_overview'
                st.rerun()

        elif st.session_state.flow_stage == 'display_final_report':
            st.subheader("Final Assessment Report")
            if not st.session_state.final_report:
                 with st.spinner("Generating final assessment report..."):
                    try:
                        st.session_state.final_report = generate_report() 
                    except Exception as e:
                        log_error("Error generating final report", traceback.format_exc())
            
            if st.session_state.final_report:
                st.markdown(st.session_state.final_report)
                
                st.download_button(
                    label="Download Report",
                    data=st.session_state.final_report,
                    file_name=f"{st.session_state.collected_details.get('full_name', 'candidate').replace(' ', '_')}_assessment_report.md",
                    mime="text/markdown"
                )
            else:
                st.error("Could not generate the final report text.")

            if st.button("Start New Assessment"):
                keys_to_clear = ['flow_stage', 'collected_details', 'asked_questions', 'current_question_idx', 'all_questions', 'evaluations', 'final_report', 'chat_history', 'current_grade_level_idx', 'current_grade_being_tested', 'last_correct_grade', 'questions_for_current_grade', 'question_index_within_grade']
                for key in keys_to_clear:
                    if key in st.session_state: del st.session_state[key]
                initialize_session_state() 
                st.rerun()

        else:
            st.error(f"Unknown flow stage: {st.session_state.flow_stage}")
            if st.button("Restart Assessment"): # Changed button text for clarity
                # Clear all session state except API key
                api_key_status = st.session_state.get('api_key_configured', False)
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.session_state.api_key_configured = api_key_status
                initialize_session_state() # Re-initialize with defaults
                st.rerun()
    except Exception as e:
        logger.error(f"Critical error in main app: {str(e)}\n{traceback.format_exc()}")
        st.error("A critical error occurred. Please check the logs (app.log) or contact support.")

if __name__ == "__main__":
    try:
        if sys.platform == 'darwin':
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError as ex:
                if "There is no current event loop in thread" in str(ex):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                else:
                    raise
        
        app()
    except Exception as e:
        logger.critical(f"Fatal error preventing app start: {str(e)}\n{traceback.format_exc()}")
        try:
            st.error("A fatal error occurred preventing the app from starting. Please check the console logs or contact support.")
        except Exception:
            print(f"FATAL ERROR (print fallback): {str(e)}") 
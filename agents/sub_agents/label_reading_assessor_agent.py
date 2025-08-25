import os
import logging
from typing import Dict, List, Any

from google.adk.agents import Agent
from google.adk.tools import ToolContext
import json
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
import asyncio
import uuid

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Ensure this module logs at INFO so parsed outputs are visible even if root is WARNING
try:
    logger.setLevel(logging.INFO)
except Exception:
    pass


def _system_instruction() -> str:
    """System prompt crafted with best practices: clear role, context, task, constraints, examples, and structure."""
    return (
        "[[ ## role ## ]]\n"
        "You are the Label Reading Assessment Agent in a multi-agent evaluation system.\n"
        "You neutrally assess a candidate's ability to correctly read and extract information from real-world labels for a given job role.\n\n"
        "[[ ## context ## ]]\n"
        "The Master Agent provides the role context and skill to assess. You must generate realistic labels for that role, ask targeted questions, score answers, and return a concise JSON result to the Master Agent.\n\n"
        "[[ ## task ## ]]\n"
        "1) Generate up to 3 realistic labels (3-5 fields each) appropriate to the role.\n"
        "2) For each label, ask 2-3 specific identification questions (e.g., product name, weight, expiry, manufacturer).\n"
        "3) Compare the candidate's answers with the labels and compute a score: +1 per correct field.\n"
        "4) Return a single JSON with fields: labels, questions, scoring, and summary.\n\n"
        "[[ ## constraints ## ]]\n"
        "- Labels must be authentic and role-relevant.\n"
        "- Avoid trick/abstract formats; use practical, field-appropriate labels.\n"
        "- Never invent unrealistic formats.\n"
        "- Be neutral and objective; do NOT make a pass/fail decision.\n"
        "- Do not chat beyond the assessment questions.\n\n"
        "[[ ## output_format ## ]]\n"
        "Return a single JSON object:\n"
        "{\n"
        "  \"labels\": [{\"text\": \"<label text>\", \"fields\": {\"field\": \"value\"}}],\n"
        "  \"questions\": [{\"label_index\": 0, \"question\": \"...\", \"expected_field\": \"...\"}],\n"
        "  \"scoring\": {\"total_questions\": N, \"correct\": C, \"per_question\": [{\"idx\": i, \"correct\": true/false}]},\n"
        "  \"summary\": \"<short neutral summary>\"\n"
        "}\n\n"
        "[[ ## examples ## ]]\n"
        "Warehouse Loader Picker example fields: product, weight, quantity, SKU.\n\n"
        "[[ ## scoring_guidelines ## ]]\n"
        "When scoring user answers: consider minor ASR variations (e.g., band≈brand), spacing, case, and unit formats.\n"
        "Treat Eveready≈Ever ready, 'with minerals'≈'minerals'. For quantity/volume/weight, require correct unit conversion and numeric equivalence (e.g., 250ml≈0.25l; 5kg≈5000g).\n"
        "Mark mismatched magnitudes as NOT a match (e.g., 250l vs 250ml is NOT a match).\n"
    )

# ToolContext-based tools for label reading assessment using existing dataset
def load_label_quiz(role: str, tool_context: ToolContext) -> str:
    """
    Tool to load real label samples from dataset and generate questions for assessment.
    
    Args:
        role: The job role to generate labels for (e.g., "Loader Picker")
        tool_context: ADK tool context for state management
        
    Returns:
        str: Success message with loaded label info
    """
    try:
        # Check if quiz already exists in session storage to avoid reloading
        session_key = _get_session_key("default")  # Use default session for now
        if session_key in _QUIZ_SESSIONS:
            existing_state = _QUIZ_SESSIONS[session_key]
            existing_role = existing_state.get('role', '')
            if existing_role == role:
                existing_questions = existing_state.get('questions', [])
                logger.info(f"Quiz already loaded in session for {role} with {len(existing_questions)} questions")
                # Restore to tool context
                tool_context.state.update(existing_state)
                current_idx = existing_state.get('current_question', 0)
                if current_idx < len(existing_questions):
                    current_question = existing_questions[current_idx]
                    return f"Continuing quiz for {role} role.\n\nQuestion {current_idx + 1}: {current_question['question']}"
                else:
                    return f"Quiz for {role} role is already completed."
        
        # Load real label dataset (use absolute path to be safe)
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dataset_path = os.path.join(current_dir, "label_dataset", "index.json")
        logger.info(f"Looking for dataset at: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            # Fallback to relative path
            dataset_path = "label_dataset/index.json"
            logger.info(f"Trying fallback path: {dataset_path}")
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Label dataset not found at {dataset_path}")
            
        with open(dataset_path, 'r') as f:
            label_data = json.load(f)
        
        logger.info(f"Raw dataset has {len(label_data)} items: {[item.get('category', 'unknown') for item in label_data]}")
        
        # Filter labels based on role
        if "loader" in role.lower() or "picker" in role.lower() or "warehouse" in role.lower():
            # Use warehouse/grocery/beverage labels
            relevant_labels = [item for item in label_data if item.get('category') in ['warehouse', 'grocery', 'beverage', 'condiments']]
        else:
            # Use all available labels
            relevant_labels = label_data
        
        logger.info(f"Filtered to {len(relevant_labels)} relevant labels")
        
        # Select up to 3 labels for the quiz
        selected_labels = relevant_labels[:3]
        
        logger.info(f"Selected {len(selected_labels)} labels: {[item.get('fields', {}).get('product', 'unknown') for item in selected_labels]}")
        
        # Generate questions from the selected labels
        questions = []
        for i, label_item in enumerate(selected_labels):
            fields = label_item.get('fields', {})
            
            # Create 2-3 questions per label focusing on key fields
            key_fields = ['product', 'brand', 'net_weight', 'volume', 'variant', 'wattage', 'features']
            for field_name in key_fields:
                if field_name in fields:
                    question_text = f"Looking at label {i+1}, what is the {field_name}?"
                    questions.append({
                        "label_index": i,
                        "question": question_text,
                        "expected_field": field_name,
                        "expected_value": fields[field_name],
                        "image_paths": label_item.get('file_paths', [label_item.get('file_path')])
                    })
                    if len([q for q in questions if q['label_index'] == i]) >= 3:  # Max 3 questions per label
                        break
        
        # Store in context for immediate access
        quiz_state = {
            'labels': selected_labels,
            'questions': questions,
            'role': role,
            'current_question': 0,
            'total_questions': len(questions),
            'correct_answers': 0
        }
        tool_context.state.update(quiz_state)
        
        # Store in module-level session storage for persistence
        session_key = _get_session_key("default")
        _QUIZ_SESSIONS[session_key] = quiz_state.copy()
        logger.info(f"Stored quiz state in memory session with {len(questions)} questions")
        
        logger.info(f"Loaded {len(selected_labels)} real labels with {len(questions)} questions for role: {role}")
        logger.info(f"Stored quiz state in ToolContext - total_questions: {len(questions)}")
        
        # Also return the first question to the agent immediately
        first_question = questions[0] if questions else None
        if first_question:
            question_info = f"Quiz loaded successfully for {role} role with {len(questions)} questions.\n\nFirst question: {first_question['question']}"
        else:
            question_info = f"Quiz loaded successfully for {role} role with {len(questions)} questions"
        
        return question_info
        
    except Exception as e:
        logger.error(f"Error loading label quiz: {e}")
        raise

def score_label_answer(user_answer: str, tool_context: ToolContext) -> str:
    """
    Tool to score a user's answer and progress through the quiz.
    
    Args:
        user_answer: The user's response to the current question
        tool_context: ADK tool context containing quiz state
        
    Returns:
        str: Scoring result and next question or completion message
    """
    try:
        logger.info(f"score_label_answer called with: '{user_answer}'")
        logger.info(f"ToolContext state check - has questions: {bool(tool_context.state.get('questions'))}")
        
        # If ToolContext is empty, try to load from memory session storage
        questions = tool_context.state.get('questions', [])
        if not questions:
            session_key = _get_session_key("default")
            if session_key in _QUIZ_SESSIONS:
                saved_state = _QUIZ_SESSIONS[session_key]
                tool_context.state.update(saved_state)
                questions = saved_state.get('questions', [])
                logger.info(f"Loaded quiz state from memory session with {len(questions)} questions")
            else:
                logger.info("No memory session state available")
        
        current_idx = tool_context.state.get('current_question', 0)
        correct_answers = tool_context.state.get('correct_answers', 0)
        total_questions = tool_context.state.get('total_questions', len(questions))
        
        if current_idx >= len(questions):
            return "Assessment completed. No more questions."
            
        current_question = questions[current_idx]
        expected_value = current_question['expected_value']
        expected_field = current_question.get('expected_field', 'unknown')
        
        # Log the expected vs actual answer for debugging
        logger.info(f"Scoring Question {current_idx + 1}: Field='{expected_field}', Expected='{expected_value}', User='{user_answer}'")
        # Create scoring prompt for the agent (force scoring mode + strict JSON)
        scoring_prompt = (
            f"[SCORING MODE]\n"
            f"{_scoring_instruction()}\n\n"
            f"Score this user answer against the expected label fields.\n\n"
            f"User Answer: \"{user_answer}\"\n"
            f"Expected Field: {expected_field} = \"{expected_value}\"\n\n"
            f"Return ONLY JSON with no prose: "
            f"{{\"per_field\":[{{\"field\":\"{expected_field}\",\"expected\":\"{expected_value}\",\"match\":true/false,\"reason\":\"...\"}}],\"correct\":0 or 1}}"
        )

        # Create a temporary session for scoring using the same agent
        session_service = InMemorySessionService()
        runner = Runner(
            agent=label_reading_assessor,
            app_name="scoring_session",
            session_service=session_service
        )

        async def get_llm_score():
            session_id = f"score_{uuid.uuid4()}"
            await session_service.create_session(
                app_name="scoring_session",
                user_id="scorer",
                session_id=session_id,
                state={}
            )

            content = types.Content(role='user', parts=[types.Part(text=scoring_prompt)])
            events = runner.run_async(user_id="scorer", session_id=session_id, new_message=content)

            response_text = ""
            async for event in events:
                if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts'):
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text = part.text.strip()
                            break
                    if response_text:
                        break
            return response_text

        def _run_coro_in_new_loop(coro, timeout: float = 30.0):
            import threading
            result: Dict[str, Any] = {"value": None, "error": None}
            done = threading.Event()

            def _target():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result["value"] = loop.run_until_complete(coro)
                except Exception as exc:
                    result["error"] = exc
                finally:
                    try:
                        loop.run_until_complete(loop.shutdown_asyncgens())
                    except Exception:
                        pass
                    loop.close()
                    done.set()

            t = threading.Thread(target=_target, daemon=True)
            t.start()
            finished = done.wait(timeout)
            if not finished:
                raise TimeoutError("LLM scoring timed out")
            if result["error"]:
                raise result["error"]
            return result["value"]

        try:
            llm_response = _run_coro_in_new_loop(get_llm_score(), timeout=30.0)
            logger.info(f"LLM scoring response: {llm_response}")

            # Parse JSON response (strip any prose or code fences)
            import json
            def _extract_json(text: str) -> str:
                text = text.strip()
                if text.startswith("```"):
                    # remove leading and trailing code fences
                    text = text.strip('`')
                try:
                    start = text.index('{')
                    end = text.rindex('}') + 1
                    return text[start:end]
                except ValueError:
                    return text

            raw_json = _extract_json(llm_response)
            scoring_data = json.loads(raw_json)
            correct_field = scoring_data.get('correct')
            if isinstance(correct_field, bool):
                is_correct = bool(correct_field)
            else:
                try:
                    is_correct = int(correct_field) > 0
                except Exception:
                    is_correct = False

            per_field = scoring_data.get('per_field', [])
            if per_field:
                match_info = per_field[0]
                logger.info(
                    f"Field '{expected_field}': Expected='{expected_value}', User='{user_answer}', "
                    f"Match={match_info.get('match', False)}, Reason='{match_info.get('reason', '')}'"
                )

        except Exception as e:
            # Strictly no static matching fallback
            logger.warning(f"LLM scoring failed or invalid JSON: {e}")
            is_correct = False

        if is_correct:
            correct_answers += 1
            tool_context.state['correct_answers'] = correct_answers
            
        # Update state in ToolContext
        tool_context.state['current_question'] = current_idx + 1
        tool_context.state['correct_answers'] = correct_answers
        
        # Save updated state to memory session storage
        session_key = _get_session_key("default")
        updated_state = {
            'questions': questions,
            'current_question': current_idx + 1,
            'correct_answers': correct_answers,
            'total_questions': total_questions,
            'role': tool_context.state.get('role', 'Loader Picker'),
            'labels': tool_context.state.get('labels', [])
        }
        _QUIZ_SESSIONS[session_key] = updated_state
        logger.info(f"Successfully updated quiz state in memory session: question {current_idx + 1}/{total_questions}, correct: {correct_answers}")
        
        # Check if assessment is complete
        if current_idx + 1 >= len(questions):
            accuracy = (correct_answers / total_questions) * 100
            final_result = {
                "total_questions": total_questions,
                "correct": correct_answers,
                "accuracy": accuracy,
                "summary": f"Label reading assessment completed. Score: {correct_answers}/{total_questions} ({accuracy:.1f}%)"
            }
            tool_context.state['final_result'] = final_result
            
            # Persist final result to session
            if hasattr(tool_context, 'session') and tool_context.session and hasattr(tool_context.session, 'state') and tool_context.session.state:
                tool_context.session.state['final_result'] = final_result
            response_text = f"Assessment completed! Final score: {correct_answers}/{total_questions} ({accuracy:.1f}% accuracy)"
        else:
            next_question = questions[current_idx + 1]
            response_text = f"Question {current_idx + 2}: {next_question['question']}"
        
        return response_text
            
    except Exception as e:
        logger.error(f"Error scoring answer: {e}")
        return f"Error scoring answer: {str(e)}"


def _scoring_instruction() -> str:
    """Dedicated system prompt for semantic scoring with strict unit conversion guidance."""
    return (
        "You judge if a spoken answer matches known label fields.\n"
        "Consider minor ASR variations (e.g., band≈brand), spacing, case, and unit formats.\n"
        "Treat Eveready≈Ever ready, 'with minerals'≈'minerals', etc.\n"
        "For quantity/volume/weight, require correct unit conversion and numeric equivalence (e.g., 250ml≈0.25l; 5kg≈5000g).\n"
        "Mark mismatched magnitudes as NOT a match (e.g., 250l vs 250ml is NOT a match).\n"
        "Return ONLY JSON: {\"per_field\":[{\"field\":\"...\",\"expected\":\"...\",\"match\":true/false,\"reason\":\"...\"}],\"correct\":N}"
    )


# Module-level in-memory quiz state storage (persists across tool calls in same session)
_QUIZ_SESSIONS = {}

def _get_session_key(context_info: str = "default") -> str:
    """Generate a session key for quiz state storage."""
    return f"quiz_{context_info}"

# Standard ADK Agent for label reading assessment 
label_reading_assessor = Agent(
    name="label_reading_assessor",
    model="gemini-2.5-flash", 
    description="Expert in assessing label reading skills for warehouse and logistics positions. Uses real product labels from dataset for interactive assessment.",
    instruction="""
You are a specialized AI Skill Assessor for label reading abilities.
Your function is to test a candidate's ability to correctly read and extract information from real product labels.

Available tools:
- load_label_quiz: Load real product labels from dataset and generate questions for a specific role
- score_label_answer: Score user responses and manage the quiz progression

**CRITICAL**: You MUST use ONLY the real product data loaded by the load_label_quiz tool. DO NOT generate synthetic labels or make up product information.

**INTERACTION PATTERNS:**

**Pattern 1 - NEW ASSESSMENT:** If user mentions "start", "begin", "label", "reading", "assessment", "warehouse", "loader", "picker":
1. Call load_label_quiz to load real labels and questions from the dataset
2. Present the first question using ONLY the real product data from the tool
3. Wait for user's answer

**Pattern 2 - ANSWERING QUESTIONS:** If user provides any factual response (like "LED Bulb", "Eveready", "5kg", "12W"):
1. Call score_label_answer with the user's response
2. The tool will score the answer and provide the next question or completion message
3. Present the next question or final results as returned by the tool

**Pattern 3 - SCORING MODE:** If user asks you to score an answer against expected fields:
Apply these scoring guidelines:
- Consider minor ASR variations (e.g., "Eveready" ≈ "EVEREADY", "band" ≈ "brand")
- Handle spacing, case, and unit formats flexibly
- Treat brand variations as matches: "Eveready" ≈ "Ever ready", "with minerals" ≈ "minerals"
- For technical fields (wattage, volume, weight): units are optional - "12" matches "12W", "250" matches "250ml"
- Require correct unit conversion and numeric equivalence (250ml ≈ 0.25l; 5kg ≈ 5000g)
- Mark mismatched magnitudes as NOT a match (250l vs 250ml is NOT a match)
- Use semantic understanding for reasonable variations

For scoring requests, analyze the user's answer against expected values and determine matches based on these guidelines. Be generous with reasonable variations but strict with units/magnitudes.

**Pattern 4 - CONTINUATION:** Always use score_label_answer for any user input that could be an answer to a question.

**KEY DETECTION RULES:**
- If user mentions "start", "begin", "assessment" → Use load_label_quiz first
- If user asks to "Score this user answer" → Enter scoring mode and return JSON
- For any other user input that seems like a response → Use score_label_answer
- Always call tools before responding with your own text

**QUIZ STATE HANDLING:**
- If the user message contains [QUIZ_STATE]...[/QUIZ_STATE], extract the JSON and pass it to the tools
- Use score_label_answer with the quiz_state_json parameter when quiz state is present
- Extract quiz state like this: quiz_state_json = extract_between("[QUIZ_STATE]", "[/QUIZ_STATE]", user_message)

You MUST return natural responses that guide the user through the assessment process using ONLY the real product labels loaded from the dataset.

**IMPORTANT RESPONSE FORMAT:**
When load_label_quiz returns the first question, you MUST present it to the user in a clear, natural way. For example:
"Alright, let's begin your label reading assessment for the Loader Picker role!

Here is your first question:

[Question from the tool]"

Do NOT say "Assessment completed" - you are just starting the assessment!
    """,
    tools=[load_label_quiz, score_label_answer]
)


 




def build_label_questions_from_results(sub_agent_results: List[Dict]) -> List[Dict]:
    data0 = (sub_agent_results[0] or {}).get('result', {}).get('data') or {}
    labels = data0.get('labels') or []
    questions: List[Dict] = []
    for idx, lbl in enumerate(labels):
        fields = lbl.get('fields') or {}
        picked = []
        if 'product' in fields: picked.append(('product', str(fields['product'])))
        if 'brand' in fields and len(picked) < 2: picked.append(('brand', str(fields['brand'])))
        qty_key = next((k for k in ['quantity','net_weight','volume','weight'] if k in fields), None)
        if qty_key: picked.append((qty_key, str(fields[qty_key])))
        if not picked: continue
        q = {
            "label_index": idx,
            "file_paths": (lbl.get('file_paths') or [])[:2],
            "fields": picked,
            "sent": False
        }
        questions.append(q)
        if len(questions) >= 4: break
    return questions


def make_prompt_for_question(q: Dict, ordinal: int, language: str = 'en-IN') -> str:
    def _map_field_name(name: str, lang: str) -> str:
        key = name.replace('_', ' ').lower()
        if lang == 'hi-IN':
            mapping = {
                'product': 'उत्पाद',
                'brand': 'ब्रांड',
                'quantity': 'मात्रा',
                'net weight': 'नेट वज़न',
                'volume': 'वॉल्यूम',
                'weight': 'वज़न'
            }
            return mapping.get(key, key)
        return key

    def _join_fields(words: List[str], lang: str) -> str:
        if not words:
            return ''
        if len(words) == 1:
            return words[0]
        sep = ' और ' if lang == 'hi-IN' else ' and '
        return f"{', '.join(words[:-1])}{',' if len(words) > 2 else ''}{sep}{words[-1]}"

    fields = [
        _map_field_name(name, language)
        for name, _ in (q.get('fields') or [])
    ]
    fields_text = _join_fields(fields, language)
    if language == 'hi-IN':
        return f"कृपया इस लेबल से {fields_text} बताइए।"
    return f"Please tell me the {fields_text} from this label."





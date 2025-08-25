import os
import json
import logging
from typing import Dict, Any

from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not (os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')):
    raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables.")


def _presentation_instruction() -> str:
    return (
        "You are acting as a professional recruiter for entry-level retail and field sales staff.\n"
        "You will be given candidate submissions for three tasks.\n\n"
        "1. Selfie/Short Video Task — Candidate greets a pretend customer.\n"
        "Evaluate ONLY based on what is visible/audible in the video description provided.\n"
        "Score 0–2 for each: Hair Neatness, Clothing Cleanliness & Fit, Footwear Neatness, Posture & Body Language, Eye Contact, Greeting Words, Tone of Voice.\n\n"
        "2. Photo Identification Task — Candidate chooses the most presentable person.\n"
        "If photo_options with image file paths are provided, analyze each image to determine which person (A/B/C/D) is most presentable for retail/sales work. Score 1 if any candidate_choice matches your determined correct choice, else 0.\n\n"
        "3. Scenario‑Based Video MCQ Task — Candidate chooses the best greeting/body language.\n"
        "You will receive candidate_choice and correct_answer; score 1 if they match else 0.\n\n"
        "Output JSON ONLY with keys exactly as: {\n"
        "  \"Video_Evaluation\": {\n"
        "    \"Hair_Neatness\": <0-2>,\n"
        "    \"Clothing_Cleanliness\": <0-2>,\n"
        "    \"Footwear_Neatness\": <0-2>,\n"
        "    \"Posture\": <0-2>,\n"
        "    \"Eye_Contact\": <0-2>,\n"
        "    \"Greeting_Words\": <0-2>,\n"
        "    \"Tone\": <0-2>,\n"
        "    \"Video_Total\": <sum>\n"
        "  },\n"
        "  \"Photo_Identification_Score\": <0 or 1>,\n"
        "  \"Photo_Correct_Choice\": \"<A/B/C/D - which option you determined as most presentable>\",\n"
        "  \"Photo_Reasoning\": \"<Brief explanation of why this person is most presentable>\",\n"
        "  \"Scenario_Video_Score\": <0 or 1>,\n"
        "  \"Overall_Total\": <Video_Total + Photo_Identification_Score + Scenario_Video_Score>,\n"
        "  \"Recommendation\": \"<Hire / Borderline / Reject>\",\n"
        "  \"Comments\": \"<Short feedback>\"\n"
        "}\n\n"
        "Recommendation Rules:\n"
        "- Hire = Overall_Total ≥ 13 AND no score of 0 in Hair_Neatness, Clothing_Cleanliness, or Greeting_Words.\n"
        "- Borderline = Overall_Total 10–12 OR one score of 0 in any video category.\n"
        "- Reject = Overall_Total ≤ 9.\n\n"
        "Input is a single JSON with keys: video_observation (string), photo_task {candidate_choice, correct_choice}, scenario_task {candidate_choice, correct_answer}.\n"
        "Compute deterministically for photo/scenario using equality. For video, infer 0-2 scores from the observation text.\n"
        "Return only raw JSON, no markdown or extra text."
    )


# Define the ADK sub-agent for presentation assessment
presentation_assessor = Agent(
    name="presentation_assessor",
    model="gemini-2.5-flash",
    description="Assesses presentation skills for retail/field sales across grooming and greeting tasks.",
    instruction=_presentation_instruction()
)


class PresentationAssessorAgent:
    """Wrapper to align with existing Application.execute_sub_agent_instruction interface."""

    def __init__(self):
        self.agent = presentation_assessor
        # Prepare ADK runner + in-memory session service
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.agent,
            app_name="presentation_assessor_app",
            session_service=self.session_service
        )

    async def execute(self, _image_data: bytes, task_context: Dict) -> Dict[str, Any]:
        """
        Expected task_context:
        {
          "role": "Retail Sales Associate",
          "skill_to_assess": "Presentation",
          "assessment_type": "presentation_composite",
          "input": {
             "video_observation": "text",
             "photo_task": {"candidate_choice": "A", "correct_choice": "C"},
             "scenario_task": {"candidate_choice": "C", "correct_answer": "C"}
          }
        }
        """
        try:
            payload = task_context.get("input") or {}
            if not isinstance(payload, dict):
                payload = {"video_observation": str(payload)}
            # Call ADK agent to produce structured JSON using Runner
            directive = (
                "EVALUATE_PRESENTATION_JSON: You MUST return ONLY valid JSON with NO markdown, NO explanations, NO extra text.\n\n"
                "CRITICAL: Analyze the attached images (A, B, C, D) to determine which person is MOST PRESENTABLE for retail/customer service work. "
                "Consider: professional appearance, grooming, clothing appropriateness, approachable demeanor.\n\n"
                "JSON Schema (MANDATORY - include ALL fields):\n"
                "{\n"
                '  "Video_Evaluation": {"Hair_Neatness": 0, "Clothing_Cleanliness": 0, "Footwear_Neatness": 0, "Posture": 0, "Eye_Contact": 0, "Greeting_Words": 0, "Tone": 0, "Video_Total": 0},\n'
                '  "Photo_Identification_Score": 0,\n'
                '  "Photo_Correct_Choice": "A",\n'
                '  "Photo_Reasoning": "Brief explanation",\n'
                '  "Scenario_Video_Score": 0,\n'
                '  "Overall_Total": 0,\n'
                '  "Recommendation": "Reject",\n'
                '  "Comments": "Brief feedback"\n'
                "}\n\n"
                "RETURN ONLY THIS JSON STRUCTURE WITH YOUR VALUES."
            )
            parts = [types.Part(text=directive), types.Part(text=json.dumps(payload))]
            # Attach images if provided
            try:
                p_opts = (payload.get('photo_task') or {}).get('photo_options') or {}
                for opt in ['A','B','C','D']:
                    pth = p_opts.get(opt)
                    if pth and os.path.exists(pth):
                        with open(pth, 'rb') as f:
                            b = f.read()
                        parts.append(types.Part(inline_data=types.Blob(mime_type='image/jpeg', data=b)))
            except Exception:
                pass
            content = types.Content(role='user', parts=parts)
            import uuid as _uuid
            session_id = f"pres_{_uuid.uuid4()}"
            user_id = "presentation_user"
            try:
                sess = await self.session_service.get_session(
                    app_name="presentation_assessor_app",
                    user_id=user_id,
                    session_id=session_id
                )
                if not sess:
                    await self.session_service.create_session(
                        app_name="presentation_assessor_app",
                        user_id=user_id,
                        session_id=session_id,
                        state={"conversation_history": []}
                    )
            except Exception:
                await self.session_service.create_session(
                    app_name="presentation_assessor_app",
                    user_id=user_id,
                    session_id=session_id,
                    state={"conversation_history": []}
                )
            events = self.runner.run_async(user_id=user_id, session_id=session_id, new_message=content)
            text = ""
            async def _collect_first_text():
                nonlocal text
                async for ev in events:
                    try:
                        if getattr(ev, 'content', None) and getattr(ev.content, 'parts', None):
                            # Check all parts for text content, not just the first
                            for part in ev.content.parts:
                                maybe_text = getattr(part, 'text', None)
                                if isinstance(maybe_text, str) and maybe_text.strip():
                                    text = maybe_text.strip()
                                    break
                            if text:
                                break
                    except Exception as e:
                        logger.warning(f"Error extracting text from event: {e}")
                        continue
            await _collect_first_text()
            logger.info(f"PresentationAssessorAgent extracted text length: {len(text)}")
            data: Dict[str, Any] = {}
            try:
                if text:
                    # Strip markdown code blocks if present
                    clean_text = text
                    if text.startswith('```json'):
                        clean_text = text[7:]  # Remove ```json
                    elif text.startswith('```'):
                        clean_text = text[3:]   # Remove ```
                    if clean_text.endswith('```'):
                        clean_text = clean_text[:-3]  # Remove trailing ```
                    clean_text = clean_text.strip()
                    data = json.loads(clean_text)
                    logger.info("PresentationAssessorAgent: Successfully parsed JSON from model")
                else:
                    logger.error("PresentationAssessorAgent: Model returned empty text")
                    return {"status": "error", "message": "Model returned empty response", "fallback_used": True}
            except Exception as e:
                logger.error(f"PresentationAssessorAgent: JSON parse failed for text: {text[:200]}... Error: {e}")
                return {"status": "error", "message": f"Invalid JSON from model: {str(e)}", "fallback_used": True}
            
            # Validate required fields
            required_fields = ["Video_Evaluation", "Photo_Identification_Score", "Photo_Correct_Choice", "Photo_Reasoning"]
            missing_fields = [f for f in required_fields if f not in data]
            if missing_fields:
                logger.error(f"PresentationAssessorAgent: Missing required fields: {missing_fields}")
                return {"status": "error", "message": f"Model response missing fields: {missing_fields}", "fallback_used": True}
            
            return {"status": "success", "data": data, "fallback_used": False}
        except Exception as e:
            logger.error(f"PresentationAssessorAgent failed: {e}")
            return {"status": "error", "message": str(e)}



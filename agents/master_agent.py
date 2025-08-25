import os
import json
import logging
from typing import Dict, Any, List
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from agents.sub_agents.stitching_assessor_agent import stitching_assessor
from agents.sub_agents.label_reading_assessor_agent import label_reading_assessor
from agents.sub_agents.presentation_assessor_agent import presentation_assessor
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not (os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')):
    raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables.")
# ADK handles API key configuration automatically

class MasterAgent:
    """
    The master orchestrator agent, built with Google ADK.
    It follows the logic defined in the master_agent_prompt.md file and is
    controlled by a host application.
    """

    def __init__(self, prompts_dir="agent_prompts"):
        self._load_knowledge(prompts_dir)
        self.agent = self._create_adk_agent()
        # Initialize ADK session service and runner for planning/execution
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.agent,
            app_name="master_agent_app",
            session_service=self.session_service
        )
        # Dedicated session service and runner for Label Reading sub-agent
        self.label_session_service = InMemorySessionService()
        self.label_runner = Runner(
            agent=label_reading_assessor,
            app_name="label_reading_app",
            session_service=self.label_session_service
        )
        # Dedicated session service and runner for Stitching sub-agent
        self.stitch_session_service = InMemorySessionService()
        self.stitch_runner = Runner(
            agent=stitching_assessor,
            app_name="stitching_assessor_app",
            session_service=self.stitch_session_service
        )

    def _load_knowledge(self, prompts_dir: str):
        """Loads the master prompt and all knowledge base files."""
        try:
            with open(os.path.join(prompts_dir, 'master_agent_prompt.md'), 'r') as f:
                self.master_prompt_template = f.read()
            with open(os.path.join(prompts_dir, 'competency_map.json'), 'r') as f:
                self.competency_map = json.load(f)
            with open(os.path.join(prompts_dir, 'sub_agent_library.json'), 'r') as f:
                self.sub_agent_library = json.load(f)
            logger.info("Master Agent knowledge bases loaded successfully.")
        except FileNotFoundError as e:
            logger.error(f"Error loading knowledge base from '{prompts_dir}': {e}")
            raise

    def _create_adk_agent(self) -> Agent:
        """Creates the Google ADK agent with proper configuration using reasoning model and sub-agents.
        Tools are intentionally left empty here; sub-agents own their tool access.
        """
        return Agent(
            model="gemini-2.5-flash",  # Reasoning-capable model for orchestration
            name="master_job_assessor",
            description="Primary interface for users interacting with a multi-agent assessment system for blue-collar roles.",
            instruction=self.master_prompt_template + "\n\n--- KNOWLEDGE BASE ---\n" + 
                       "\nCompetency Map:\n" + json.dumps(self.competency_map, indent=2) +
                       "\n\nSub-Agent Library:\n" + json.dumps(self.sub_agent_library, indent=2) +
                       "\n\nYou can delegate to specialized sub-agents based on the role assessment needed. Use your sub-agents for technical skill evaluation.",
            tools=[],  # No global tools – sub-agents expose their own tools
            sub_agents=[stitching_assessor, label_reading_assessor, presentation_assessor]
        )

    async def _get_adk_response(self, prompt: str) -> Dict[str, Any]:
        """Call the ADK Master Agent to produce a plan; parse robustly into a dict.

        Expected JSON shape (at minimum):
        {
          "response_to_user": string,
          "sub_agent_instructions": [ { "agent_name": string, "task_context": { ... } } ]
        }
        """
        try:
            # Build a single planning message that includes KB context and the user planning request
            planning_text = (
                self.master_prompt_template
                + "\n\n--- KNOWLEDGE BASE ---\n"
                + "\nCompetency Map:\n" + json.dumps(self.competency_map, indent=2)
                + "\n\nSub-Agent Library:\n" + json.dumps(self.sub_agent_library, indent=2)
                + "\n\nTask: Based on the user query below, plan the assessment and return STRICT JSON with keys: response_to_user (string), sub_agent_instructions (array of {agent_name, task_context}).\n"
                + "Return only JSON without markdown.\n\n"
                + "User Planning Request:\n" + prompt
            )
            content = types.Content(role='user', parts=[types.Part(text=planning_text)])
            # Unique session for each planning request
            import uuid as _uuid
            session_id = f"plan_{_uuid.uuid4()}"
            user_id = "planner"
            # Ensure session exists before running
            try:
                sess = await self.session_service.get_session(
                    app_name="master_agent_app",
                    user_id=user_id,
                    session_id=session_id
                )
                if not sess:
                    await self.session_service.create_session(
                        app_name="master_agent_app",
                        user_id=user_id,
                        session_id=session_id,
                        state={"conversation_history": []}
                    )
            except Exception:
                await self.session_service.create_session(
                    app_name="master_agent_app",
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
                            part0 = ev.content.parts[0]
                            maybe_text = getattr(part0, 'text', None)
                            if isinstance(maybe_text, str) and maybe_text.strip():
                                text = maybe_text.strip()
                                break
                    except Exception:
                        continue
            await _collect_first_text()
            parsed: Dict[str, Any] = {}
            if text:
                try:
                    parsed = json.loads(text)
                except Exception:
                    # Fallback: extract first JSON object
                    try:
                        start = text.find('{')
                        end = text.rfind('}')
                        if start != -1 and end != -1 and end > start:
                            parsed = json.loads(text[start:end+1])
                    except Exception:
                        parsed = {}
            if not isinstance(parsed, dict):
                parsed = {}
            # Ensure required keys exist
            if 'response_to_user' not in parsed:
                parsed['response_to_user'] = "I'll prepare your assessment plan now."
            if 'sub_agent_instructions' not in parsed or not isinstance(parsed.get('sub_agent_instructions'), list):
                parsed['sub_agent_instructions'] = []
            # If Presentation skill requested but missing input, explicitly ask user for JSON instead of falling through
            try:
                for instr in parsed.get('sub_agent_instructions', []):
                    ctx = instr.get('task_context', {}) if isinstance(instr, dict) else {}
                    if ctx.get('skill_to_assess') == 'Presentation' and not ctx.get('input'):
                        example = (
                            '{"input":{"video_observation":"Hair tidy, shirt clean and fitted, shoes polished, upright posture, steady eye contact, greets: \"Good afternoon ma’am, welcome\"; tone warm.",' 
                            '"photo_task":{"candidate_choice":"C","correct_choice":"C"},' 
                            '"scenario_task":{"candidate_choice":"C","correct_answer":"C"}}}'
                        )
                        parsed['response_to_user'] = (
                            "To assess presentation, please send a JSON payload with video observations and choices. "
                            f"Example: {example}"
                        )
                        # Clear instructions until input is provided
                        parsed['sub_agent_instructions'] = []
                        break
            except Exception:
                pass
            return parsed
        except Exception as e:
            logger.error(f"Error calling ADK agent: {e}")
            return {
                "response_to_user": "I ran into a planning issue. Please try again in a moment.",
                "sub_agent_instructions": []
            }

    def _handle_plan_request(self, prompt: str) -> Dict[str, Any]:
        """Handle initial assessment planning based on user query and optional image path."""
        user_query = self._extract_user_query(prompt).lower()
        image_path = self._extract_image_path(prompt)
        
        # Extract any file path from user query (simple approach)
        if not image_path:
            import re
            # Look for file paths in the query - try multiple patterns
            patterns = [
                r'(/[^\s]+\.(?:jpg|jpeg|png|gif|bmp|tiff|webp|avif))',  # Unix paths
                r'([a-zA-Z]:[^\s]+\.(?:jpg|jpeg|png|gif|bmp|tiff|webp|avif))',  # Windows paths
                r'(~[^\s]+\.(?:jpg|jpeg|png|gif|bmp|tiff|webp|avif))',  # Home paths
            ]
            
            for pattern in patterns:
                path_match = re.search(pattern, user_query, re.IGNORECASE)
                if path_match:
                    image_path = path_match.group(1)
                    break
        

        
        # Check for greetings first
        greeting_words = ['hello', 'hi', 'hey', 'start', 'help']
        if any(greeting in user_query for greeting in greeting_words) and len(user_query.split()) <= 2:
            return {
                "response_to_user": "Hello! Welcome to our Skill Assessment System. I can evaluate your abilities for various blue-collar roles. Currently I can assess the Tailor role. Which position are you interested in?",
                "sub_agent_instructions": []
            }
        
        # Try to identify role from user query with flexible matching
        identified_role = None
        role_keywords = {
            "Tailor": ["tailor", "tailoring", "sewing", "stitching"],
            "Loader Picker": ["loader picker", "loader", "picker", "warehouse"],
        }
        
        # Look for role keywords in user query
        for role_name, keywords in role_keywords.items():
            if any(keyword in user_query for keyword in keywords):
                identified_role = role_name
                break

        # If no role found in the current query, fall back to the last remembered role (if any)
        if not identified_role and hasattr(self, "last_identified_role"):
            identified_role = getattr(self, "last_identified_role")
        
        # Additional checks for simple role mentions
        if not identified_role and ("tailor role" in user_query or user_query.strip() == "tailor"):
            identified_role = "Tailor"
        
        # If still no role identified, ask for clarification (list all currently supported roles)
        if not identified_role:
            supported_roles = ", ".join(list(self.competency_map.get("roles", {}).keys()))
            return {
                "response_to_user": f"I'd be happy to help you with a job skill assessment! We currently support: {supported_roles}. Which role are you applying for?",
                "sub_agent_instructions": []
            }
        
        # Check if this role is supported
        if identified_role not in self.competency_map.get("roles", {}):
            supported_roles = ", ".join(list(self.competency_map.get("roles", {}).keys()))
            return {
                "response_to_user": f"Thank you for your interest in the {identified_role} role. We currently support: {supported_roles}.",
                "sub_agent_instructions": []
            }
        
        # Persist the identified role for future reference (so that subsequent messages containing only the image can still trigger the assessment)
        self.last_identified_role = identified_role

        # Handle supported role (Tailor)
        role_info = self.competency_map["roles"][identified_role]
        required_skills = role_info["required_skills"]
        
        # Skill-specific orchestration
        sub_agent_instructions = []
        if "Stitching" in required_skills:
            if image_path and image_path.lower() != "none":
                task_context = {
                    "role": identified_role,
                    "skill_to_assess": "Stitching",
                    "assessment_type": "practical_image",
                    "image_path": image_path
                }
                sub_agent_instructions.append({
                    "agent_name": "StitchingAssessorAgent",
                    "task_context": task_context
                })
                return {
                    "response_to_user": f"Perfect! I'll assess your stitching skills for the {identified_role} position using your provided image. Let me analyze your work...",
                    "sub_agent_instructions": sub_agent_instructions
                }
            else:
                return {
                    "response_to_user": f"Great! I can help you assess your skills for the {identified_role} position. Please share a clear image of your stitching work to proceed.",
                    "sub_agent_instructions": []
                }
        
        if "Label Reading" in required_skills:
            task_context = {
                "role": identified_role,
                "skill_to_assess": "Label Reading",
                "assessment_type": "label_reading_quiz"
            }
            sub_agent_instructions.append({
                "agent_name": "LabelReadingAssessorAgent",
                "task_context": task_context
            })
            return {
                "response_to_user": f"Let's begin your label reading assessment for the {identified_role} role. I'll present a few labels and ask quick questions.",
                "sub_agent_instructions": sub_agent_instructions
            }

        # Fallback if no skill path matched
        return {
            "response_to_user": f"I can help with {identified_role}. Tell me more about your experience while I prepare the assessment.",
            "sub_agent_instructions": []
        }
    
    def _handle_final_verdict(self, prompt: str) -> Dict[str, Any]:
        """Handle final decision delivery based on assessment results."""
        if "error" in prompt.lower():
            return {
                "response_to_user": "I was unable to complete the assessment due to a technical issue. Please try again with a clear image of your stitching work.",
                "final_decision_data": {
                    "decision": "INCOMPLETE",
                    "justification": "Assessment could not be completed due to image processing error."
                }
            }
        else:
            return {
                "response_to_user": "Based on your assessment results, I'll need to review your stitching quality. Please ensure you provide a clear image of your work for proper evaluation.",
                "final_decision_data": {
                    "decision": "PENDING",
                    "justification": "Waiting for proper image submission for stitching quality evaluation."
                }
            }
    
    def _extract_user_query(self, prompt: str) -> str:
        """Extract the user query from the full prompt."""
        if "User Query:" in prompt:
            lines = prompt.split('\n')
            for line in lines:
                if line.startswith("User Query:"):
                    return line.replace("User Query:", "").strip().strip('"')
        return prompt

    def _extract_image_path(self, prompt: str) -> str:
        """Extract the image path from the full prompt."""
        # Check for formal Image Path: format
        if "Image Path:" in prompt:
            lines = prompt.split('\n')
            for line in lines:
                if line.startswith("Image Path:"):
                    return line.replace("Image Path:", "").strip().strip('"')
        
        # Check for [path] format in user query
        user_query = self._extract_user_query(prompt)
        if '[' in user_query and ']' in user_query:
            start = user_query.find('[')
            end = user_query.find(']', start)
            if start != -1 and end != -1:
                return user_query[start+1:end].strip()
        
        # Check for natural language image paths
        import re
        # Look for file paths with common image extensions
        path_patterns = [
            r'(/[^\s]+\.(?:jpg|jpeg|png|gif|bmp|tiff|webp|avif))',  # Unix paths
            r'([A-Za-z]:[^\s]+\.(?:jpg|jpeg|png|gif|bmp|tiff|webp|avif))',  # Windows paths
        ]
        
        for pattern in path_patterns:
            matches = re.findall(pattern, user_query, re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        return None

    async def _format_a2a_response(self, agent_text: str, context_id: str = None, user_query: str = "") -> Dict[str, Any]:
        """Format agent response in A2A format with proper FilePart and TextPart structure."""
        try:
            import uuid
            
            # Check if this is the first question or subsequent questions
            is_first_question = ("first question" in agent_text.lower() or 
                               "looking at label 1" in agent_text.lower() or
                               "begin your label reading" in agent_text.lower())
            
            has_question = "question" in agent_text.lower() and "?" in agent_text
            is_completion = ("assessment completed" in agent_text.lower() or 
                           "final score" in agent_text.lower())
            
            # Generate A2A response structure
            message_id = str(uuid.uuid4())
            context_id_value = context_id or "label-reading-test"
            
            # Start with text part
            parts = []
            
            # Add image if this is a question
            if has_question and not is_completion:
                try:
                    dataset_path = "label_dataset/index.json"
                    if os.path.exists(dataset_path):
                        with open(dataset_path, 'r') as f:
                            label_data = json.load(f)
                        
                        # Filter for warehouse/loader picker role
                        relevant_labels = [item for item in label_data if item.get('category') in ['warehouse', 'grocery', 'beverage', 'condiments']]
                        if not relevant_labels:
                            relevant_labels = label_data[:3]
                        
                        # Determine which label to show
                        if is_first_question:
                            current_label = relevant_labels[0]
                        else:
                            # Extract question number to determine label
                            import re
                            question_match = re.search(r'Question (\d+)', agent_text)
                            question_num = int(question_match.group(1)) if question_match else 2
                            label_index = min((question_num - 1) // 3, len(relevant_labels) - 1)
                            current_label = relevant_labels[label_index]
                        
                        # Get image path and convert to URI format - handle both file_path and file_paths
                        image_paths = []
                        if current_label.get('file_path'):
                            image_paths = [current_label.get('file_path')]
                        elif current_label.get('file_paths'):
                            image_paths = current_label.get('file_paths', [])
                        
                        # Add all available images (both front and back) for complete information
                        if image_paths:
                            for image_path in image_paths:
                                image_uri = f"http://localhost:5000/{image_path.replace('label_dataset/', 'label-media/')}"
                                parts.append({
                                    "mediaType": "image/jpeg",
                                    "type": "FilePart", 
                                    "uri": image_uri
                                })
                                logger.info(f"Added image to A2A response: {image_uri}")
                        
                except Exception as e:
                    logger.error(f"Error adding image to A2A response: {e}")
            
            # Add text part
            parts.append({
                "text": agent_text,
                "type": "TextPart"
            })
            
            a2a_response = {
                "id": "label_test",
                "jsonrpc": "2.0",
                "result": {
                    "contextId": context_id_value,
                    "message": {
                        "messageId": message_id,
                        "parts": parts,
                        "role": "agent"
                    }
                }
            }
            
            logger.info(f"Generated A2A response with {len(parts)} parts")
            return a2a_response
            
        except Exception as e:
            logger.error(f"Error formatting A2A response: {e}")
            # Fallback to simple response
            return {
                "response_to_user": agent_text or "Assessment ready.",
                "error": f"A2A formatting error: {str(e)}"
            }


    async def get_plan(self, user_query: str, image_path: str = None) -> Dict[str, Any]:
        """
        **Host-Called Method 1: Plan Assessment.**
        Takes the user query and optional image path, generates an assessment plan. ADK will auto-delegate if needed.
        """
        logger.info(f"Master Agent [Planner]: Getting plan for query: '{user_query}'" + 
                   (f" with image: '{image_path}'" if image_path else " [NO IMAGE]"))
        
        # For A2A and other direct calls, use the deterministic planning logic that properly handles image paths
        if image_path or any(keyword in user_query.lower() for keyword in ['tailor', 'stitching', 'sewing']):
            plan_result = self._handle_plan_request(f"User Query: {user_query}\nImage Path: {image_path or 'None'}")
            return plan_result
        
        # For text-only queries, use ADK response
        task_prompt = (
            f"User Query: \"{user_query}\"\n"
            f"Image Path: \"{image_path or 'None'}\"\n\n"
            "CRITICAL: If the user requests stitching assessment but Image Path is 'None', return empty sub_agent_instructions and ask them to provide an image.\n"
            "Your task is to analyze the user's query and generate the JSON for 'A. Initial Assessment Setup' as described in your instructions."
        )
        return await self._get_adk_response(task_prompt)
    
    async def execute_assessment(self, user_query: str, image_path: str = None, context_id: str = None) -> Dict[str, Any]:
        """
        **Execute Assessment with Natural ADK Delegation**
        Following the ADK pattern from renovation agent - let the master agent naturally delegate.
        """
        logger.info(f"Master Agent [Assessment]: Starting for query: '{user_query}'" + 
                   (f" with image: '{image_path}'" if image_path else " [NO IMAGE]"))
        
        # Use the ADK runner directly like the renovation agent example
        try:
            lower_text = (user_query or "").lower()
            is_stitch_intent = bool(image_path) or any(k in lower_text for k in ['stitch', 'stitching', 'tailor', 'sewing'])
            # Route stitching first if image provided or stitching intent detected
            if is_stitch_intent:
                stitch_session_id = f"stitch_{context_id}" if context_id else f"stitch_ephemeral"
                try:
                    await self.stitch_session_service.create_session(
                        app_name="stitching_assessor_app",
                        user_id="user",
                        session_id=stitch_session_id,
                        state={}
                    )
                except Exception:
                    pass

                stitch_prompt = (
                    "EXECUTE STITCHING ASSESSMENT\n\n"
                    f"User request: {user_query}\n"
                    f"Image file path: {image_path or 'None'}\n\n"
                    "Use retrieve_image_from_path first, then validate_image_data, then return STRICT JSON as per your schema."
                )
                stitch_content = types.Content(role='user', parts=[types.Part(text=stitch_prompt)])
                stitch_events = self.stitch_runner.run_async(user_id="user", session_id=stitch_session_id, new_message=stitch_content)
                stitch_text = ""
                async for ev in stitch_events:
                    try:
                        if getattr(ev, 'content', None) and getattr(ev.content, 'parts', None):
                            for part in ev.content.parts:
                                maybe_text = getattr(part, 'text', None)
                                if isinstance(maybe_text, str) and maybe_text.strip():
                                    stitch_text = maybe_text.strip()
                    except Exception:
                        continue
                logger.info(f"Stitching sub-agent returned: '{stitch_text}'")
                return {"response_to_user": stitch_text or "Assessment completed."}

            # If label reading intent OR continuing an existing label reading session, forward to the dedicated label runner
            is_label_intent = any(k in lower_text for k in ['label', 'reading', 'warehouse', 'loader', 'picker', 'start'])
            is_label_session = context_id and context_id.startswith('label-reading')
            if is_label_intent or is_label_session:
                label_session_id = f"lr_{context_id}" if context_id else f"lr_ephemeral"
                master_session_id = f"assess_{context_id}" if context_id else f"assess_ephemeral"
                
                # Ensure label reading session exists
                try:
                    await self.label_session_service.create_session(
                        app_name="label_reading_app",
                        user_id="user",
                        session_id=label_session_id,
                        state={}
                    )
                except Exception:
                    pass
                
                # Ensure master session exists
                try:
                    await self.session_service.create_session(
                        app_name="master_agent_app",
                        user_id="user",
                        session_id=master_session_id,
                        state={}
                    )
                except Exception:
                    pass
                    
                # Send the raw user message so the sub-agent can decide to call its tools
                label_content = types.Content(role='user', parts=[types.Part(text=user_query)])
                label_events = self.label_runner.run_async(user_id="user", session_id=label_session_id, new_message=label_content)
                # Collect the final text after tool calls complete, not the first partial chunk
                label_text = ""
                async for ev in label_events:
                    try:
                        if getattr(ev, 'content', None) and getattr(ev.content, 'parts', None):
                            for part in ev.content.parts:
                                maybe_text = getattr(part, 'text', None)
                                if isinstance(maybe_text, str) and maybe_text.strip():
                                    label_text = maybe_text.strip()
                    except Exception:
                        continue
                logger.info(f"Label reading agent returned: '{label_text}'")
                # Format A2A response for label reading assessment
                a2a_response = await self._format_a2a_response(label_text, context_id, user_query)
                return a2a_response

            # Use contextId for session persistence, fallback to UUID if not provided
            if context_id:
                session_id = f"assess_{context_id}"
            else:
                import uuid as _uuid
                session_id = f"assess_{_uuid.uuid4()}"
            user_id = "user"
            
            logger.info(f"Master Agent [Assessment]: Using session_id: {session_id}")
        
            # Create session if it doesn't exist (for session persistence)
            try:
                await self.session_service.create_session(
                    app_name="master_agent_app", 
                    user_id=user_id,
                    session_id=session_id,
                    state={}
                )
                logger.info(f"Master Agent [Assessment]: Created NEW session {session_id}")
            except Exception as e:
                # Session might already exist, which is fine for persistence
                logger.info(f"Master Agent [Assessment]: Continuing EXISTING session {session_id}: {e}")
                
                # Check if session exists and has state
                try:
                    session = await self.session_service.get_session(user_id, session_id)
                    if session:
                        logger.info(f"Master Agent [Assessment]: Session {session_id} found with state: {bool(session.state)}")
                except Exception as check_error:
                    logger.info(f"Master Agent [Assessment]: Could not check session state: {check_error}")
                pass
        
            # Explicit execution mode prompt - make it clear we want delegation, not planning JSON
            if image_path:
                prompt = f"""EXECUTE ASSESSMENT (not planning):

User request: {user_query}
Image file path: {image_path}

Use EXECUTION MODE: Delegate to your stitching_assessor sub-agent immediately and provide the natural assessment results. Do NOT return JSON planning format."""
            else:
                prompt = f"""EXECUTE ASSESSMENT (not planning):

User request: {user_query}

Use EXECUTION MODE: Assess the user's request and provide natural responses. Do NOT return JSON planning format."""
            
            # Execute with natural prompt - ADK will handle delegation based on instruction
            content = types.Content(role='user', parts=[types.Part(text=prompt)])
            events = self.runner.run_async(user_id=user_id, session_id=session_id, new_message=content)
            
            # Collect response
            response_text = ""
            async for ev in events:
                if getattr(ev, 'content', None) and getattr(ev.content, 'parts', None):
                    part0 = ev.content.parts[0]
                    maybe_text = getattr(part0, 'text', None)
                    if isinstance(maybe_text, str) and maybe_text.strip():
                        response_text = maybe_text.strip()
                        break
            
            return {"response_to_user": response_text or "Assessment completed."}
                
        except Exception as e:
            logger.error(f"Error in execute_assessment: {e}")
            return {"response_to_user": f"Assessment error: {str(e)}"}

    async def get_final_verdict(self, role: str, sub_agent_results: List[Dict], language: str = 'en-IN') -> Dict[str, Any]:
        """Compute final verdict deterministically using competency map thresholds.
        Falls back to a neutral response if required signals are missing.
        """
        logger.info(f"Master Agent [Judge]: Computing final verdict for role '{role}'.")
        try:
            roles = self.competency_map.get("roles", {})
            role_info = roles.get(role, {})
            thresholds = role_info.get("passing_thresholds", {})

            # Try stitching path
            for item in sub_agent_results:
                data = (item or {}).get('result', {}).get('data', {})
                if isinstance(data, dict) and ("quality_rating" in data or "professional_grade" in data):
                    st = thresholds.get("Stitching", {})
                    min_q = st.get("min_quality_rating", 7)
                    allowed_grades = st.get("required_professional_grade", ["Advanced", "Expert"])
                    rating = data.get("quality_rating", 0)
                    grade = str(data.get("professional_grade", "")).title()
                    passed = (rating >= min_q) and (grade in allowed_grades)
                    decision = "PASS" if passed else "FAIL"
                    if language == 'hi-IN':
                        msg = (
                            f"{role} भूमिका के लिए सिलाई मूल्यांकन पूर्ण। "
                            f"गुणवत्ता {rating}/10, स्तर {grade}. परिणाम: {decision}."
                        )
                    else:
                        msg = (
                            f"Stitching evaluation complete for the {role} role. "
                            f"Quality {rating}/10, level {grade}. Result: {decision}."
                        )
                    return {
                        "response_to_user": msg,
                        "final_decision_data": {
                            "decision": decision,
                            "justification": f"rating>={min_q} and grade in {allowed_grades}"
                        }
                    }

            for item in sub_agent_results:
                data = (item or {}).get('result', {}).get('data', {})
                scoring = data.get("scoring") if isinstance(data, dict) else None
                if isinstance(scoring, dict) and "correct" in scoring and ("total_questions" in scoring or "total_fields" in scoring):
                    correct = int(scoring.get("correct", 0))
                    total = int(scoring.get("total_fields", scoring.get("total_questions", 0)))
                    acc = (correct / total * 100.0) if total > 0 else 0.0
                    lr = thresholds.get("Label Reading", {})
                    min_acc = float(lr.get("min_accuracy", 95))
                    passed = acc >= min_acc
                    decision = "PASS" if passed else "FAIL"
                    if language == 'hi-IN':
                        msg = (
                            f"{role} भूमिका के लिए लेबल पढ़ने का मूल्यांकन पूर्ण। "
                            f"सटीकता {acc:.0f}% ({correct}/{total}). परिणाम: {decision}."
                        )
                    else:
                        msg = (
                            f"Label reading assessment complete for the {role} role. "
                            f"Accuracy {acc:.0f}% ({correct}/{total}). Result: {decision}."
                        )
                    return {
                        "response_to_user": msg,
                        "final_decision_data": {
                            "decision": decision,
                            "justification": f"accuracy>={min_acc}%"
                        }
                    }

            # Fallback neutral response
            return {
                "response_to_user": ("मूल्यांकन पूरा हुआ। हम विवरणों की समीक्षा करेंगे और शीघ्र ही आपसे संपर्क करेंगे।" if language == 'hi-IN' else "Assessment completed. We will review the details and get back to you shortly."),
                "final_decision_data": {
                    "decision": "PENDING",
                    "justification": "insufficient scoring signals"
                }
            }
        except Exception as e:
            logger.error(f"Final verdict computation failed: {e}")
            return {
                "response_to_user": ("तकनीकी समस्या के कारण मैं अंतिम निर्णय नहीं दे सका/सकी।" if language == 'hi-IN' else "I couldn't compute the final decision due to a technical issue."),
                "final_decision_data": {
                    "decision": "INCOMPLETE",
                    "justification": str(e)
                }
            }

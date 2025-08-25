import os
import logging
import asyncio
from typing import Dict, List
from datetime import datetime
from dataclasses import dataclass
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import uuid
import requests
import tempfile
import base64
from agents.master_agent import MasterAgent
import json


load_dotenv()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Session:
    """Data class for session management"""
    session_id: str
    conversation_history: List[Dict] = None
    language: str = "en-IN"
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []


class A2AApp:
    """A2A server for Master Agent orchestration."""
    
    def __init__(self):
        self.host_app = MasterAgent()
        self.app = Flask(__name__)
        self.a2a_contexts: Dict[str, Session] = {}
        self.setup_routes()
        self.setup_a2a_routes()

    def setup_routes(self):
        """Setup basic Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                "status": "healthy", 
                "active_sessions": len(self.a2a_contexts),
                "timestamp": datetime.now().isoformat()
            })

        @self.app.route('/label-media/<path:filepath>', methods=['GET'])
        def serve_label_media(filepath: str):
            """Serve label images from the local dataset folder."""
            return send_from_directory('label_dataset', filepath)

        @self.app.route('/presentation-media/<path:filepath>', methods=['GET'])
        def serve_presentation_media(filepath: str):
            """Serve presentation resources (images/videos) for A2A file parts."""
            return send_from_directory('presentation_resources', filepath)
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application"""
        # Only print on first start, not on reloader restart
        import os
        if not os.environ.get('WERKZEUG_RUN_MAIN'):
            print(f"Starting A2A App on {host}:{port}")
        # Disable auto-reloader to avoid clearing in-memory sessions on file changes
        self.app.run(host=host, port=port, debug=debug, use_reloader=False)

    def _extract_text_and_image_from_parts(self, parts) -> tuple[str, str | None]:
        """A2A: Extract text and an optional image file path from message parts."""
        text = ""
        image_path = None
        try:
            print(f"A2A: Processing {len(parts or [])} parts")
            for i, p in enumerate(parts or []):
                if not isinstance(p, dict):
                    logger.warning(f"A2A: Part {i} is not a dict: {type(p)}")
                    continue
                ptype = (p.get("type") or p.get("mimeType") or "").lower()
                print(f"A2A: Part {i} type='{ptype}', keys={list(p.keys())}")
                
                # TextPart or plain text field
                if "textpart" in ptype or ptype == "text/plain" or ("text" in p and isinstance(p.get("text"), str)):
                    if not text:
                        text = (p.get("text") or p.get("content") or "").strip()
                        print(f"A2A: Extracted text: '{text[:50]}...'")
                
                # FilePart or image attachment
                if "filepart" in ptype or ptype.startswith("image/") or ("uri" in p) or ("inlineData" in p) or ("data" in p) or ("path" in p):
                    print(f"A2A: Found potential file part, type='{ptype}'")
                    uri = p.get("uri")
                    file_path = p.get("path")  # Direct file path
                    inline = p.get("inlineData") or p.get("data") or {}
                    print(f"A2A: uri={bool(uri)}, path={bool(file_path)}, inline_keys={list(inline.keys()) if isinstance(inline, dict) else type(inline)}")
                    
                    if file_path:
                        # Direct file path - just use it
                        if os.path.exists(file_path):
                            image_path = file_path
                            print(f"A2A: Using direct file path: {image_path}")
                        else:
                            print(f"A2A: File path not found: {file_path}")
                    elif uri:
                        print(f"A2A: Processing URI: {str(uri)[:50]}...")
                        if str(uri).startswith("file://"):
                            # File URI
                            local_path = str(uri)[7:]  # Remove file://
                            if os.path.exists(local_path):
                                image_path = local_path
                                print(f"A2A: Using file URI path: {image_path}")
                            else:
                                print(f"A2A: File URI path not found: {local_path}")
                        elif str(uri).startswith("http"):
                            r = requests.get(uri, timeout=20)
                            r.raise_for_status()
                            with tempfile.NamedTemporaryFile(suffix='.img', delete=False) as tmp:
                                tmp.write(r.content)
                                image_path = tmp.name
                                print(f"A2A: Saved HTTP image to {image_path}")
                        elif str(uri).startswith("data:"):
                            b64_data = str(uri).split(",", 1)[-1]
                            with tempfile.NamedTemporaryFile(suffix='.img', delete=False) as tmp:
                                tmp.write(base64.b64decode(b64_data))
                                image_path = tmp.name
                                print(f"A2A: Saved data URI image to {image_path}")
                    elif inline:
                        b64 = inline.get("data") or inline.get("bytes") or ""
                        print(f"A2A: Found inline data, length={len(str(b64))}")
                        if b64:
                            with tempfile.NamedTemporaryFile(suffix='.img', delete=False) as tmp:
                                tmp.write(base64.b64decode(b64))
                                image_path = tmp.name
                                print(f"A2A: Saved inline image to {image_path}")
        except Exception as e:
            logger.error(f"A2A: Image extraction failed: {e}")
        return (text or "").strip(), image_path


    def _a2a_file_part_for_path(self, rel_path: str) -> Dict:
        """Build a single FilePart for a path under presentation_resources/ using public URI."""
        base = request.url_root.rstrip('/')
        uri = f"{base}/presentation-media/{rel_path}"
        mt = 'application/octet-stream'
        lower = rel_path.lower()
        if lower.endswith('.jpg') or lower.endswith('.jpeg'):
            mt = 'image/jpeg'
        elif lower.endswith('.png'):
            mt = 'image/png'
        elif lower.endswith('.webp'):
            mt = 'image/webp'
        elif lower.endswith('.gif'):
            mt = 'image/gif'
        elif lower.endswith('.mp4'):
            mt = 'video/mp4'
        elif lower.endswith('.mov'):
            mt = 'video/quicktime'
        return {"type": "FilePart", "mediaType": mt, "uri": uri}

    def _compose_detailed_assessment(self, sub_agent_results: List[Dict], language: str = 'en-IN') -> str:
        """Format sub-agent results into a concise assessment message."""
        try:
            if not sub_agent_results:
                return ""
            # Prefer first result's data
            data = (sub_agent_results[0] or {}).get('result', {}).get('data')
            if not isinstance(data, dict):
                return ""
            rating = data.get('quality_rating')
            stitch_type = data.get('stitch_type')
            issues = data.get('technical_issues') or []
            tips = data.get('improvement_suggestions') or []
            prof = data.get('professional_grade')
            if language == 'hi-IN':
                parts = ["विस्तृत विश्लेषण:"]
                if rating is not None:
                    parts.append(f"• गुणवत्ता रेटिंग: {rating}/10")
                if stitch_type:
                    parts.append(f"• टाँका प्रकार: {stitch_type}")
                if prof:
                    parts.append(f"• कौशल स्तर: {str(prof).title()}")
                if issues:
                    parts.append("• तकनीकी समस्याएँ:")
                    for it in issues[:3]:
                        parts.append(f"  - {it}")
                if tips:
                    parts.append("• सुधार सुझाव:")
                    for tip in tips[:3]:
                        parts.append(f"  - {tip}")
            else:
                parts = ["Detailed Analysis:"]
                if rating is not None:
                    parts.append(f"• Quality Rating: {rating}/10")
                if stitch_type:
                    parts.append(f"• Stitch Type: {stitch_type}")
                if prof:
                    parts.append(f"• Skill Level: {str(prof).title()}")
                if issues:
                    parts.append("• Technical Issues:")
                    for it in issues[:3]:
                        parts.append(f"  - {it}")
                if tips:
                    parts.append("• Improvement Tips:")
                    for tip in tips[:3]:
                        parts.append(f"  - {tip}")
            return "\n".join(parts)
        except Exception:
            return ""

    # A2A endpoints
    def setup_a2a_routes(self):
        @self.app.route('/.well-known/agent-card.json', methods=['GET'])
        def a2a_agent_card():
            base_url = request.url_root.rstrip('/')
            card = {
                "capabilities": {
                    "pushNotifications": False,
                    "streaming": False
                },
                "defaultInputModes": [
                    "text/plain"
                ],
                "defaultOutputModes": [
                    "text/plain"
                ],
                "description": "A master agent that helps individuals find and apply jobs by orchestrating task delegation to specialized agents. It coordinates job searches, profile creation, outbound calls and training recommendations through dynamic multi-agent workflows.",
                "name": "Master Agent Orchestrator",
                "preferredTransport": "JSONRPC",
                "protocolVersion": "0.3.0",
                "security": [
                    {
                        "apiKey": []
                    }
                ],
                "securitySchemes": {
                    "apiKey": {
                        "description": "API key authentication for accessing the Master Agent's.",
                        "in": "header",
                        "name": "X-API-Key",
                        "type": "apiKey"
                    }
                },
                "skills": [
                    {
                        "description": "Understands blue-collar job seeker requests and delegates tasks like profile creation, job discovery, skill verifications to sub-agents.",
                        "examples": [
                            "Find delivery jobs in Bangalore",
                            "Suggest training programs for drivers"
                        ],
                        "id": "orchestrate_workflows",
                        "name": "Orchestrate Multi-Agent Workflows",
                        "tags": [
                            "blue-collar",
                            "job-seeking",
                            "delegation",
                            "multi-agent",
                            "worker-support"
                        ]
                    }
                ],
                "url": base_url,
                "version": "0.0.1"
            }
            return jsonify(card)

        @self.app.route('/a2a/rpc', methods=['POST'])
        def a2a_rpc():
            # API key auth (optional)
            expected_key = os.getenv('A2A_API_KEY')
            provided_key = request.headers.get('X-API-Key') or request.headers.get('x-api-key')
            if expected_key and provided_key != expected_key:
                return jsonify({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32001, "message": "Unauthorized"}
                }), 401

            data = request.get_json(force=True, silent=True) or {}
            rpc_id = data.get("id")
            if data.get("jsonrpc") != "2.0" or "method" not in data:
                return jsonify({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32600, "message": "Invalid Request"}})

            method = data.get("method")
            params = data.get("params") or {}

            def ok(result):
                return jsonify({"jsonrpc": "2.0", "id": rpc_id, "result": result})

            def err(code, message):
                return jsonify({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}})

            if method == "message/send":
                message = params.get("message") or {}
                parts = message.get("parts") or []
                text, image_path = self._extract_text_and_image_from_parts(parts)
                if not text and isinstance(message.get("text"), str):
                    text = (message.get("text") or "").strip()
                
                print(f"A2A extracted - Text: '{text[:50]}...', Image: {bool(image_path)}")
                if image_path:
                    print(f"A2A: Image path = {image_path}")
                
                lang = params.get("language") or 'en-IN'

                context_id = params.get("contextId") or f"a2a_{str(uuid.uuid4())}"
                sess_key = f"a2a_{context_id}"
                sess = self.a2a_contexts.get(sess_key)
                if not sess:
                    sess = Session(session_id=sess_key)
                    self.a2a_contexts[sess_key] = sess

                if text:
                    sess.conversation_history.append({"user_input": text})

                try:
                    import asyncio
                    # Use ADK delegation for all assessment types (stitching, label reading, etc.)
                    print(f"A2A: Using ADK delegation for assessment - text='{(text or '')[:50]}...', image_path='{image_path}'")
                    assessment_result = asyncio.run(self.host_app.execute_assessment(text or "", image_path, context_id))
                    
                    if isinstance(assessment_result, dict):
                        # Check if this is an A2A formatted response
                        if 'jsonrpc' in assessment_result and 'result' in assessment_result:
                            a2a_message = assessment_result.get('result', {}).get('message', {})
                            reply_parts = a2a_message.get('parts', [])
                            print(f"A2A: Using A2A formatted response with {len(reply_parts)} parts")
                        else:
                            reply_text = assessment_result.get('response_to_user') or "Assessment completed through ADK delegation."
                            reply_parts = [{"type": "TextPart", "text": reply_text}]
                            
                            label_images = assessment_result.get('label_images')
                            if label_images:
                                # Create file parts for the label images
                                try:
                                    file_parts = []
                                    base = request.url_root.rstrip('/')
                                    for img_path in label_images:
                                        if img_path:  # Skip None/empty paths
                                            rel = str(img_path).replace('label_dataset/', '')
                                            uri = f"{base}/label-media/{rel}"
                                            file_parts.append({
                                                "type": "FilePart",
                                                "mediaType": "image/jpeg",
                                                "uri": uri
                                            })
                                    reply_parts = file_parts + reply_parts  # Images first, then text
                                    print(f"A2A: Added {len(file_parts)} image file parts from master agent metadata")
                                except Exception as e:
                                    print(f"A2A: Failed to create file parts from metadata: {e}")
                    else:
                        reply_parts = [{"type": "TextPart", "text": "Assessment completed through ADK delegation."}]

                    if sess.conversation_history:
                        # Store only the final message text for compact history
                        last_text = reply_parts[-1]["text"] if reply_parts else reply_text
                        sess.conversation_history[-1]["agent_response"] = last_text

                    return ok({
                        "message": {
                            "role": "agent",
                            "parts": reply_parts,
                            "messageId": str(uuid.uuid4())
                        },
                        "contextId": context_id
                    })
                except Exception as e:
                    logger.error(f"A2A message/send failed: {e}")
                    return err(-32000, "Server error")

            return err(-32601, "Method not found")

def main():
    """Main entry point"""
    try:
        if not (os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')):
            logger.error("Missing required API key: Please set either GOOGLE_API_KEY or GEMINI_API_KEY")
            print("Tip: Use GOOGLE_API_KEY to avoid SDK warnings when both are set")
            return
        
        # Start application
        app = A2AApp()
        app.run(debug=False)
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

if __name__ == "__main__":
    main()
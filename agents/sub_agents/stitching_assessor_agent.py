import os
import json
import logging
from typing import Dict, Any
from google.adk.agents import Agent
from google.adk.tools import ToolContext
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not (os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')):
    raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables.")
# ADK handles API key configuration automatically

def retrieve_image_from_path(image_path: str, tool_context: ToolContext) -> str:
    """
    Tool to retrieve image data from file path and store in context.
    
    Args:
        image_path: Path to the image file
        tool_context: ADK tool context for state management
        
    Returns:
        str: Success message with image info
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Store image data in tool context for other tools to access
        tool_context.state['image_data'] = image_data
        tool_context.state['image_path'] = image_path
        tool_context.state['image_size'] = len(image_data)
        
        logger.info(f"Retrieved image from path: {image_path}, size: {len(image_data)} bytes")
        return f"Successfully retrieved image from {image_path} ({len(image_data)} bytes)"
    except Exception as e:
        logger.error(f"Error retrieving image from path {image_path}: {str(e)}")
        raise

def validate_image_data(tool_context: ToolContext) -> str:
    """
    Tool to validate image data stored in context and extract basic properties.
    
    Args:
        tool_context: ADK tool context containing image data from previous tool call
        
    Returns:
        str: Validation results message
    """
    try:
        # Get image data from context (set by retrieve_image_from_path)
        image_data = tool_context.state.get('image_data')
        image_path = tool_context.state.get('image_path', 'unknown')
        
        if not image_data:
            raise ValueError("No image data found in context. Call retrieve_image_from_path first.")
            
        # Validate using PIL
        image_io = io.BytesIO(image_data)
        image = Image.open(image_io)
        image.verify()  # Verify the image is valid
        
        # Reopen for processing (verify() closes the file)
        image_io.seek(0)  # Reset to beginning
        image = Image.open(image_io)
        
        validation_result = {
            "valid": True,
            "format": image.format,
            "size": image.size,
            "mode": image.mode,
            "file_size": len(image_data),
            "file_path": image_path
        }
        
        # Store validation results in context
        tool_context.state['validation_result'] = validation_result
        
        logger.info(f"Image validation successful: {validation_result}")
        return f"Image validation successful: {validation_result['format']} format, {validation_result['size']} pixels, {validation_result['file_size']} bytes"
        
    except Exception as e:
        error_msg = f"Image validation failed: {str(e)}"
        logger.error(error_msg)
        tool_context.state['validation_result'] = {"valid": False, "error": str(e)}
        return error_msg

# Define the standalone ADK sub-agent for stitching assessment
stitching_assessor = Agent(
    name="stitching_assessor",
    model="gemini-2.0-flash",
    description="Specialized expert in assessing stitching quality for tailor positions. Analyzes images and provides detailed technical evaluation.",
    instruction="""
You are a specialized AI Skill Assessor with the expert persona of a master tailor.
Your function is to analyze an image of a stitching sample and provide detailed, objective evaluation.

Available tools:
- retrieve_image_from_path: Get image data from a file path and store in context
- validate_image_data: Validate image data stored in context and get properties

When given an image file path, use retrieve_image_from_path first, then validate_image_data, then analyze the stitching quality based on the image data and validation results stored in the tool context.

You MUST return a single, valid JSON object with your findings:

{
    "quality_rating": <integer from 1-10>,
    "stitch_type": "<identified stitch type>",
    "technical_issues": ["<a short, clear issue>"],
    "improvement_suggestions": ["<an actionable tip>"],
    "professional_grade": "<Beginner/Novice/Intermediate/Advanced/Expert>",
    "pass_fail_raw": "<Pass|Fail based on technical merit only>"
}
    """,
    tools=[retrieve_image_from_path, validate_image_data]
)

# Legacy wrapper class for backward compatibility with non-ADK flows
class StitchingAssessorAgent:
    """
    Legacy wrapper for backward compatibility with existing host application.
    Note: For ADK flows, use the stitching_assessor agent directly via master agent delegation.
    """

    def __init__(self):
        self.agent = stitching_assessor
        logger.info("StitchingAssessorAgent: Legacy wrapper initialized. For ADK flows, use master agent delegation instead.")

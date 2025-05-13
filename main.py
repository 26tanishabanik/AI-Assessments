# main.py - Main application logic for the AI Assessment Bot
import os
import utils # For get_agno_knowledge_base
import google.generativeai as genai # For API key configuration
from textwrap import dedent
import re # For parsing agent outputs

from agents import (
    candidate_details_agno_agent,
    tailor_question_generation_agno_agent,
    evaluation_agno_agent,
    report_agno_agent
)

def configure_api_key():
    """Ensures the Google API key is configured."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("[ERROR] GOOGLE_API_KEY environment variable not set.")
        print("Please set it before running the bot.")
        return False
    try:
        genai.configure(api_key=google_api_key)
        print("[INFO] Google API Key configured successfully for genai.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to configure Google API Key: {e}")
        return False

def parse_candidate_details(agent_response_content: str):
    """ Parses the structured summary from candidate_details_agno_agent."""
    details = {
        "full_name": "Not specified",
        "years_of_experience": "Not specified",
        "specializations": "Not specified"
    }
    # Expected summary format is now part of the agent's instructions for its final output.
    # e.g., "Candidate Full Name: [name], Experience: [experience], Specializations: [specializations]"
    try:
        name_match = re.search(r"Candidate Full Name: (.*?)(?:, Experience:|,|$)", agent_response_content, re.IGNORECASE | re.DOTALL)
        if name_match: details["full_name"] = name_match.group(1).strip()
        
        exp_match = re.search(r"Experience: (.*?)(?:, Specializations:|,|$)", agent_response_content, re.IGNORECASE | re.DOTALL)
        if exp_match: details["years_of_experience"] = exp_match.group(1).strip()
        
        spec_match = re.search(r"Specializations: (.*?)(?:$)", agent_response_content, re.IGNORECASE | re.DOTALL)
        if spec_match: details["specializations"] = spec_match.group(1).strip()
        
        # Fallback if the specific summary isn't found but individual extractions were logged by the agent
        if details["full_name"] == "Not specified":
            fb_name = re.search(r"Extracted Name: (.*?)(?:\n|$)", agent_response_content, re.IGNORECASE | re.DOTALL)
            if fb_name: details["full_name"] = fb_name.group(1).strip()
        if details["years_of_experience"] == "Not specified":
            fb_exp = re.search(r"Extracted Experience: (.*?)(?:\n|$)", agent_response_content, re.IGNORECASE | re.DOTALL)
            if fb_exp: details["years_of_experience"] = fb_exp.group(1).strip()
        if details["specializations"] == "Not specified":
            fb_spec = re.search(r"Extracted Specialization: (.*?)(?:\n|$)", agent_response_content, re.IGNORECASE | re.DOTALL)
            if fb_spec: details["specializations"] = fb_spec.group(1).strip()

    except Exception as e:
        print(f"[ERROR] Could not parse candidate details from agent response: {e}. Response: {agent_response_content}")
    return details

def parse_evaluation_response(agent_response_content: str):
    """ Parses Assessment and Explanation from evaluation_agno_agent response."""
    assessment = "Could not parse assessment."
    explanation = agent_response_content # Default to full content
    try:
        # Using re.DOTALL to make . match newlines as explanations can be multi-line
        assessment_match = re.search(r"Assessment:(.*?)(?:Explanation:|$)", agent_response_content, re.IGNORECASE | re.DOTALL)
        if assessment_match:
            assessment = assessment_match.group(1).strip()
        
        explanation_match = re.search(r"Explanation:(.*)", agent_response_content, re.IGNORECASE | re.DOTALL)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        elif assessment_match and assessment != "Could not parse assessment.": # If assessment found but no explicit explanation label
            explanation = "Assessment provided without separate explanation label or format differs."

    except Exception as e:
        print(f"[ERROR] Could not parse evaluation response: {e}. Response: {agent_response_content}")
    return {"llm_assessment": assessment, "llm_explanation": explanation}

if __name__ == "__main__":
    print("--- Starting AI Assessment Bot (Agno Version) ---")

    if not configure_api_key():
        exit()

    print("\n--- Loading Tailoring Assessment Data ---")
    # Load the grading data from CSV
    grading_data = utils.load_grading_data()
    if grading_data is None:
        print("[CRITICAL] Failed to load tailoring assessment questions from CSV. Aborting.")
        exit()
    else:
        print(f"[INFO] Successfully loaded tailoring assessment questions from CSV.")

    # Initialize Agno knowledge base for evaluation agent (optional)
    print("\n--- Initializing Agno Knowledge Base ---")
    agno_kb = utils.get_agno_knowledge_base()
    if not agno_kb:
        print("[WARNING] Failed to initialize Agno Knowledge Base. Will proceed without it for question evaluation.")
    else:
        print("[INFO] Agno Knowledge Base initialized.")

    # --- Step 1: Candidate Details Collection ---
    print("\n--- Step 1: Candidate Details Collection (via Agno Agent) ---")
    
    collected_details = {
        "full_name": None,
        "years_of_experience": None,
        "specializations": None
    }
    
    details_prompts_map = {
        "full_name": "your full name",
        "years_of_experience": "your years of experience in tailoring",
        "specializations": "any specializations you have in tailoring"
    }
    
    extraction_instructions_map = {
        "full_name": "From the candidate response '{user_response}', extract only the full name.",
        "years_of_experience": "From the candidate response '{user_response}', extract the years of experience or level (e.g., beginner, 5 years).",
        "specializations": "From the candidate response '{user_response}', extract any mentioned specializations. If none, state 'None specified.'"
    }

    conversation_history_for_details_agent = ["You are a friendly interviewer. Your goal is to collect candidate details."]

    for detail_key, detail_desc in details_prompts_map.items():
        if collected_details[detail_key] is None: # If detail not yet collected
            print(f"\nCollecting: {detail_desc.replace('your ','')}...")
            
            # Construct prompt for the agent to ask the question
            ask_prompt = f"Current conversation: {' -- '.join(conversation_history_for_details_agent[-3:])}. Based on this, please now ask the candidate specifically for {detail_desc}. Keep your question concise."
            agent_question_response = candidate_details_agno_agent.run(ask_prompt)
            ai_question = agent_question_response.content.strip()
            print(f"AI: {ai_question}")
            conversation_history_for_details_agent.append(f"AI asked: {ai_question}")
            
            user_response = input("You: ").strip()
            conversation_history_for_details_agent.append(f"Candidate responded: {user_response}")
            
            # Construct prompt for the agent to extract the detail from user's response
            extract_prompt = extraction_instructions_map[detail_key].format(user_response=user_response)
            agent_extraction_response = candidate_details_agno_agent.run(extract_prompt)
            extracted_value = agent_extraction_response.content.strip()
            
            collected_details[detail_key] = extracted_value
            print(f"[INFO] Recorded {detail_key}: {extracted_value}")
            conversation_history_for_details_agent.append(f"System recorded {detail_key} as: {extracted_value}")

    print("\n--- Candidate Details Collection Finished ---")
    print(f"Collected Details: {collected_details}")

    # Final check for critical details like name
    if not collected_details.get("full_name") or collected_details.get("full_name") in ["Not specified", "Error"]:
        print("[CRITICAL] Could not obtain a valid candidate name. Aborting.")
        exit()

    # --- Step 2: Tailoring Questions from CSV ---
    asked_questions_and_answers = []
    print("\n--- Step 2: Tailoring Questions from CSV ---")
    
    # Get random questions from CSV file
    csv_questions = utils.get_questions_from_csv(num_questions=3)  # Get 3 questions
    
    if not csv_questions:
        print("[CRITICAL] No questions could be loaded from the CSV file. Aborting assessment.")
        exit()
    
    for i, question_data in enumerate(csv_questions):
        print(f"\nQuestion {i+1} - Grade: {question_data['grade']}")
        print(f"AI: {question_data['question']}")
        candidate_answer = input("You: ").strip()
        
        asked_questions_and_answers.append({
            "grade": question_data['grade'],
            "machine": question_data['machine'],
            "material": question_data['material'],
            "job": question_data['job'],
            "product": question_data['product'],
            "generated_question": question_data['question'],
            "reference_answer": question_data['reference_answer'],
            "candidate_answer": candidate_answer
        })
    
    print("--- Tailoring Questions Finished ---")

    # --- Step 3: Evaluation of Answers ---
    print("\n--- Step 3: Evaluation of Answers ---")
    all_evaluations = []
    
    if asked_questions_and_answers:
        for item in asked_questions_and_answers:
            print(f"\nEvaluating response for question: '{item['generated_question'][:50]}...'")
            
            # Check if the answer is correct using similarity comparison (90% threshold)
            is_correct = False
            try:
                # Get embedder for semantic matching
                embedder = utils.get_embedder_instance()
                if embedder:
                    # Embed both answers
                    candidate_embedding = embedder.encode([item['candidate_answer']])[0]
                    reference_embedding = embedder.encode([item['reference_answer']])[0]
                    
                    # Calculate similarity
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarity = cosine_similarity([candidate_embedding], [reference_embedding])[0][0]
                    
                    # Check if above 90% threshold
                    is_correct = similarity > 0.9
                    print(f"Answer similarity score: {similarity:.2f}")
                else:
                    # Fallback to keyword matching
                    candidate_answer = item['candidate_answer'].lower()
                    reference_answer = item['reference_answer'].lower()
                    
                    # Split reference answer into keywords
                    ref_keywords = reference_answer.split()
                    
                    # Count matching keywords
                    match_count = sum(1 for kw in ref_keywords if kw in candidate_answer)
                    match_score = match_count / len(ref_keywords) if ref_keywords else 0
                    
                    # Check if above 90% threshold
                    is_correct = match_score > 0.9
                    print(f"Keyword match score: {match_score:.2f}")
            except Exception as e:
                print(f"[ERROR] During similarity calculation: {e}")
                # Default to evaluating with LLM only
            
            # Use evaluation agent for qualitative assessment
            if evaluation_agno_agent:
                eval_input_text = dedent(f"""
                Here is the information for evaluation:

                Reference Answer:
                {item['reference_answer']}

                Question Asked:
                {item['generated_question']}

                Candidate's Answer:
                {item['candidate_answer']}
                
                Expected Grade: {item['grade']}
                
                Is the answer correct at 90% threshold: {is_correct}
                """)
                
                try:
                    evaluation_response = evaluation_agno_agent.run(eval_input_text)
                    parsed_eval = parse_evaluation_response(evaluation_response.content)
                    
                    # Assign grade based on correctness
                    final_grade = item['grade'] if is_correct else utils.grade_down(item['grade'])
                    
                    print(f"Original Question Grade: {item['grade']}")
                    print(f"Correctness: {'Correct' if is_correct else 'Incorrect'} (using 90% threshold)")
                    print(f"Final Grade: {final_grade}")
                    print(f"Assessment: {parsed_eval['llm_assessment']}")
                    
                    all_evaluations.append({
                        "machine": item["machine"],
                        "material": item["material"],
                        "job": item["job"],
                        "product": item["product"],
                        "generated_question": item["generated_question"],
                        "candidate_answer": item["candidate_answer"],
                        "reference_answer": item["reference_answer"],
                        "llm_assessment": parsed_eval["llm_assessment"],
                        "llm_explanation": parsed_eval["llm_explanation"],
                        "is_correct": is_correct,
                        "original_grade": item["grade"],
                        "final_grade": final_grade
                    })
                except Exception as e:
                    print(f"[ERROR] During Evaluation Agent: {e}")
                    # Fallback to just using the original grade if evaluation agent fails
                    final_grade = item['grade'] if is_correct else utils.grade_down(item['grade'])
                    all_evaluations.append({
                        "machine": item["machine"],
                        "material": item["material"],
                        "job": item["job"],
                        "product": item["product"],
                        "generated_question": item["generated_question"],
                        "candidate_answer": item["candidate_answer"],
                        "reference_answer": item["reference_answer"],
                        "llm_assessment": "Error in evaluation",
                        "llm_explanation": f"Error during evaluation: {str(e)}",
                        "is_correct": is_correct,
                        "original_grade": item["grade"],
                        "final_grade": final_grade
                    })
            else:
                # No evaluation agent, just use correctness check
                final_grade = item['grade'] if is_correct else utils.grade_down(item['grade'])
                all_evaluations.append({
                    "machine": item["machine"],
                    "material": item["material"],
                    "job": item["job"],
                    "product": item["product"],
                    "generated_question": item["generated_question"],
                    "candidate_answer": item["candidate_answer"],
                    "reference_answer": item["reference_answer"],
                    "llm_assessment": "Direct comparison",
                    "llm_explanation": f"{'Correct' if is_correct else 'Incorrect'} based on 90% similarity threshold",
                    "is_correct": is_correct,
                    "original_grade": item["grade"],
                    "final_grade": final_grade
                })
    else:
        print("No answers to evaluate.")
    
    print("--- Evaluation of Answers Finished ---")

    # Calculate overall grade based on individual grades
    overall_grade = "C"  # Default
    if all_evaluations:
        grade_map = {"A*": 5, "A": 4, "B+": 3, "B": 2, "C": 1}
        total_points = sum(grade_map.get(eval_item['final_grade'], 1) for eval_item in all_evaluations)
        avg_points = total_points / len(all_evaluations)
        
        # Map back to letter grade
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
    print(f"Overall Grade: {overall_grade}")

    # --- Step 4: Generating Final Report ---
    print("\n--- Step 4: Generating Final Report (via Agno Agent) ---")
    report_input_summary = f"Candidate Details:\nFull Name: {collected_details.get('full_name', 'N/A')}\nExperience: {collected_details.get('years_of_experience', 'N/A')}\nSpecializations: {collected_details.get('specializations', 'N/A')}\n\n"
    
    # Add the overall grade to the report
    report_input_summary += f"Overall Grade: {overall_grade}\n\n"
    
    report_input_summary += "Question Performance Summary:\n"
    if all_evaluations:
        for i, eval_item in enumerate(all_evaluations):
            correctness_status = "✓ CORRECT" if eval_item.get('is_correct', False) else "✗ INCORRECT"
            report_input_summary += (
                f"  Q{i+1}: {eval_item['generated_question']}\n"
                f"    Candidate's Answer: {eval_item['candidate_answer']}\n"
                f"    Reference Answer: {eval_item['reference_answer']}\n"
                f"    Status: {correctness_status}\n"
                f"    Final Grade: {eval_item['final_grade']} (Original question grade: {eval_item['original_grade']})\n"
                f"    Assessment: {eval_item['llm_assessment']}\n\n"
            )
    else:
        report_input_summary += "- No questions were evaluated or evaluations available.\n"
    
    print("AI: (Generating final report...)")
    final_report_generation_prompt = f"Based on the following information, generate the assessment report. Make sure the grade is prominently featured in the report:\n\n{report_input_summary}"
    try:
        final_report_response = report_agno_agent.run(final_report_generation_prompt)
        print("\n--- ASSESSMENT REPORT ---")
        print(final_report_response.content)
        print("--- END OF REPORT ---")
    except Exception as e:
        print(f"[ERROR] During Report Agent: {e}")
        # Fallback to a simple report if the agent fails
        print("\n--- ASSESSMENT REPORT (FALLBACK) ---")
        print(f"Candidate: {collected_details.get('full_name', 'N/A')}")
        print(f"Experience: {collected_details.get('years_of_experience', 'N/A')}")
        print(f"Specializations: {collected_details.get('specializations', 'N/A')}")
        print(f"Overall Grade: {overall_grade}")
        print("--- END OF REPORT ---")

    print("\n--- Assessment Bot (Agno Version) workflow complete. ---") 
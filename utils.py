# utils.py - Helper functions
import os
import pandas as pd  # For CSV processing
from sentence_transformers import SentenceTransformer
import logging

# Configure logging for utils
logger = logging.getLogger(__name__)

# Global DataFrame to store loaded grading data
grading_data_df = None
embedder = None

# CSV Path (relative to the project root)
CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'Question Bank - For RAG  - Question & Answer for Phython Code.csv')

def get_embedder_instance():
    """Initializes and returns a SentenceTransformer model."""
    global embedder
    if embedder is None:
        try:
            # Using a lightweight model suitable for sentence similarity
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("SentenceTransformer model 'all-MiniLM-L6-v2' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}", exc_info=True)
            return None
    return embedder

def load_grading_data():
    """
    Loads the grading data from the CSV file into a global pandas DataFrame.
    Returns the DataFrame or None if loading fails.
    """
    global grading_data_df
    try:
        # Construct the absolute path to the CSV file
        # Assumes utils.py is in the root of the project, and data is in a 'data' subfolder.
        
        # Corrected path construction assuming utils.py is in the root
        # and 'data' is a direct subdirectory of the root.
        # CSV_FILE_PATH was already defined globally, so we use it directly.
        
        if not os.path.exists(CSV_FILE_PATH):
            logger.error(f"CSV file not found at: {CSV_FILE_PATH}")
            # Attempt to find it relative to the current working directory if running from project root
            alt_csv_path = os.path.join(os.getcwd(), 'data', 'Question Bank - For RAG  - Question & Answer for Phython Code.csv')
            if os.path.exists(alt_csv_path):
                logger.info(f"Found CSV at alternative path: {alt_csv_path}")
                current_csv_path = alt_csv_path
            else:
                 # Try one level up from utils.py for the data folder (if utils is in a src dir for example)
                alt_csv_path_up = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Question Bank - For RAG  - Question & Answer for Phython Code.csv')
                if os.path.exists(alt_csv_path_up):
                    logger.info(f"Found CSV at alternative path (one dir up): {alt_csv_path_up}")
                    current_csv_path = alt_csv_path_up
                else:
                    logger.error(f"Also tried: {alt_csv_path} and {alt_csv_path_up}, but file not found.")
                    return None
        else:
            current_csv_path = CSV_FILE_PATH
            
        grading_data_df = pd.read_csv(current_csv_path)
        logger.info(f"Successfully loaded grading data from {current_csv_path}. Shape: {grading_data_df.shape}")
        
        # Basic data cleaning: strip whitespace from column names
        grading_data_df.columns = grading_data_df.columns.str.strip()
        
        # Ensure essential columns exist (adjust column names as per your CSV)
        expected_cols = ['GRADE', 'Machine', 'Situational questions', 'Answers']
        missing_cols = [col for col in expected_cols if col not in grading_data_df.columns]
        if missing_cols:
            logger.error(f"Missing expected columns in CSV: {missing_cols}. Available: {list(grading_data_df.columns)}")
            grading_data_df = None # Invalidate df if crucial columns are missing
            return None

        # For 'Machine' column, strip whitespace from its values and handle NaNs
        if 'Machine' in grading_data_df.columns:
            grading_data_df['Machine'] = grading_data_df['Machine'].fillna('').astype(str).str.strip()
        else:
            logger.warning("'Machine' column not found, machine-specific filtering will not work.")


        return grading_data_df
    except FileNotFoundError:
        logger.error(f"Error: The file {CSV_FILE_PATH} (or alternatives) was not found.")
        grading_data_df = None
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"Error: The file {CSV_FILE_PATH} is empty.")
        grading_data_df = None
        return None
    except Exception as e:
        logger.error(f"An error occurred while loading the CSV: {e}", exc_info=True)
        grading_data_df = None
        return None

def get_questions_from_csv(num_questions=3, grades=None, machine_types=None, is_entry_level=False):
    """
    Fetches questions from the CSV based on various criteria.
    - num_questions: Max number of questions to return.
    - grades: A list of grades to filter by (e.g., ['C', 'B']).
    - machine_types: A list of machine types the candidate knows.
    - is_entry_level: Boolean, if True, filters for general/entry-level questions.
    """
    global grading_data_df
    if grading_data_df is None or grading_data_df.empty:
        # Attempt to load data if not already loaded
        logger.warning("grading_data_df not loaded in get_questions_from_csv. Attempting to load.")
        load_grading_data()
        if grading_data_df is None or grading_data_df.empty:
            logger.error("Failed to load grading_data_df in get_questions_from_csv. Cannot fetch questions.")
            return []


    # Ensure required columns are present before proceeding
    required_columns = ['Situational questions', 'Answers', 'GRADE', 'Machine']
    # Check against the actual columns in the dataframe
    actual_columns = grading_data_df.columns
    if not all(col in actual_columns for col in required_columns):
        missing = [col for col in required_columns if col not in actual_columns]
        logger.error(f"Missing one or more required columns for question fetching: {missing}. Available: {list(actual_columns)}")
        return []

    filtered_df = grading_data_df.copy()

    # Entry-level questions: typically lower grades and no specific machine
    if is_entry_level:
        logger.info("Filtering for entry-level questions.")
        if grades:
             filtered_df = filtered_df[filtered_df['GRADE'].isin(grades)]
        else: # Default entry-level grades if not specified
             filtered_df = filtered_df[filtered_df['GRADE'].isin(['C', 'B'])]
        # Also filter for questions where 'Machine' is empty or NaN
        filtered_df = filtered_df[filtered_df['Machine'].fillna('').str.strip() == '']
        logger.info(f"After entry-level filter, {len(filtered_df)} questions remain.")


    # Filter by specific machine types if provided and not entry-level
    elif machine_types: # machine_types is a list of strings
        logger.info(f"Filtering for machine types: {machine_types}")
        # Normalize selected machine types for robust comparison
        normalized_selected_machines = [str(m).lower().strip() for m in machine_types if str(m).strip()]

        if normalized_selected_machines: # Only apply if there are actual machines to filter by
            def machine_match(row_machines_cell):
                if pd.isna(row_machines_cell) or not str(row_machines_cell).strip():
                    return False
                # Normalize CSV machine entries (split by comma, lowercase, strip)
                csv_machines_in_row = [str(m).lower().strip() for m in str(row_machines_cell).split(',')]
                return any(selected_m in csv_machines_in_row for selected_m in normalized_selected_machines)

            filtered_df = filtered_df[filtered_df['Machine'].apply(machine_match)]
            logger.info(f"After machine type filter ({machine_types}), {len(filtered_df)} questions remain.")
        else: # If machine_types list was empty or only contained empty strings
            logger.info("Machine types list was empty, no machine-specific filtering applied.")


    # Filter by grades if provided and not exclusively handled by entry-level logic already
    if grades and not is_entry_level: # Avoid double-filtering grades if entry-level already did it
        logger.info(f"Filtering for grades: {grades}")
        filtered_df = filtered_df[filtered_df['GRADE'].isin(grades)]
        logger.info(f"After grade filter ({grades}), {len(filtered_df)} questions remain.")


    if filtered_df.empty:
        logger.warning(f"No questions found after applying filters: entry_level={is_entry_level}, machines={machine_types}, grades={grades}")
        return []

    # Randomly sample or implement a more sophisticated selection strategy
    num_to_sample = min(num_questions, len(filtered_df))
    if num_to_sample == 0: # No questions to sample
        logger.warning(f"No questions to sample after filtering. num_to_sample is 0.")
        return []
        
    sampled_questions_df = filtered_df.sample(n=num_to_sample, random_state=42) # Added random_state for reproducibility if desired
    
    final_questions = []
    for index, q_row in sampled_questions_df.iterrows():
        # Use .get with a default for all columns to prevent KeyError if a column is unexpectedly missing
        # after a filter operation, though the initial check for required_columns should prevent this.
        final_q = {
            "Situational questions": q_row.get("Situational questions", "N/A Question Text"),
            "Answers": q_row.get("Answers", "N/A Answer Text"),
            "GRADE": q_row.get("GRADE", "N/A Grade"),
            "Machine ": q_row.get("Machine", ""), # Retaining space for app.py compatibility, though df has 'Machine'
            "Material ": q_row.get("Material", ""),
            "Job ": q_row.get("Job", ""),
            "Product": q_row.get("Product", "")
        }
        final_questions.append(final_q)
    
    logger.info(f"Returning {len(final_questions)} questions after sampling.")
    return final_questions

def get_grade_for_answer(candidate_answer, reference_answer=None, question_topic=None):
    """
    Compare candidate's answer to a reference answer or grading data and return the appropriate grade.
    Uses semantic similarity to determine the grade.
    """
    # If no reference answer provided, try to find one from the grading data
    if not reference_answer and question_topic:
        grading_data = load_grading_data()
        if grading_data is not None:
            # Look for a matching topic in the grading data
            for _, row in grading_data.iterrows():
                topic_match = False
                for col in ['Machine', 'Material', 'Job', 'Product']:
                    if col in grading_data.columns and pd.notna(row.get(col, '')) and question_topic.lower() in str(row[col]).lower():
                        topic_match = True
                        break
                
                if topic_match and pd.notna(row.get('Answers', '')):
                    reference_answer = row['Answers']
                    break
    
    # If we still don't have a reference answer, return default grade
    if not reference_answer:
        return "C" # Return default grade if no reference found
    
    # Get embedder for semantic matching
    embedder = get_embedder_instance()
    if embedder is None:
        print("[WARNING] Embedder instance not available. Falling back to keyword matching.")
        return grade_by_keywords(candidate_answer, reference_answer)
    
    try:
        # Embed both answers
        candidate_embedding = embedder.encode([candidate_answer])[0]
        reference_embedding = embedder.encode([reference_answer])[0]
        
        # Calculate similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity([candidate_embedding], [reference_embedding])[0][0]
        
        # Assign grade based on similarity (adjust thresholds as needed)
        if similarity > 0.8:
            return "A*"
        elif similarity > 0.7:
            return "A"
        elif similarity > 0.6:
            return "B+"
        elif similarity > 0.5:
            return "B"
        else:
            return "C"
    except Exception as e:
        print(f"[ERROR] Error calculating grade via similarity: {e}")
        return grade_by_keywords(candidate_answer, reference_answer) # Fallback on error

def grade_by_keywords(candidate_answer, reference_answer):
    """Simple keyword-based grading fallback"""
    try:
        candidate_answer = candidate_answer.lower()
        reference_answer = reference_answer.lower()
        
        # Split reference answer into keywords (simple split)
        ref_keywords = set(reference_answer.split())
        if not ref_keywords:
            return "C" # Cannot grade if reference is empty
            
        # Count matching keywords
        candidate_words = set(candidate_answer.split())
        match_count = len(ref_keywords.intersection(candidate_words))
        match_score = match_count / len(ref_keywords)
        
        # Assign grade based on match score (adjust thresholds as needed)
        if match_score > 0.7:
            return "A*"
        elif match_score > 0.6:
            return "A"
        elif match_score > 0.5:
            return "B+"
        elif match_score > 0.4:
            return "B"
        else:
            return "C"
    except Exception as e:
        print(f"[ERROR] Error during keyword grading: {e}")
        return "C" # Default to C on error

def grade_down(original_grade):
    """Lowers the grade by one step."""
    grade_order = ["A*", "A", "B+", "B", "C"]
    try:
        idx = grade_order.index(original_grade)
        return grade_order[idx + 1] if idx + 1 < len(grade_order) else grade_order[-1]
    except ValueError:
        return "C" # Default to lowest if original grade is unrecognized

def get_all_machine_types():
    """
    Extracts all unique, non-empty machine names from the 'Machine' column 
    of the loaded grading_data_df.
    Returns a sorted list of machine names.
    """
    global grading_data_df
    if grading_data_df is None or grading_data_df.empty:
        # Attempt to load data if not already loaded, as this function might be called independently
        logger.warning("grading_data_df not loaded in get_all_machine_types. Attempting to load.")
        load_grading_data() 
        if grading_data_df is None or grading_data_df.empty:
             logger.error("Failed to load grading_data_df in get_all_machine_types. Cannot get machine types.")
             return [] # Return empty list if still not loaded

    if 'Machine' not in grading_data_df.columns:
        logger.warning("'Machine' column not found in CSV. Cannot get machine types.")
        return []

    try:
        # Get unique machine names, split if multiple machines are in one cell (comma-separated)
        all_machines = set()
        for machines_cell in grading_data_df['Machine'].dropna().unique():
            if machines_cell: # Ensure it's not an empty string after potential prior cleaning
                for machine in machines_cell.split(','):
                    cleaned_machine = machine.strip()
                    if cleaned_machine: # Add only if it's not an empty string after strip
                        all_machines.add(cleaned_machine)
        
        sorted_machines = sorted(list(all_machines))
        logger.info(f"Found machine types: {sorted_machines}")
        return sorted_machines
    except Exception as e:
        logger.error(f"Error extracting machine types: {e}", exc_info=True)
        return []

# If this script is run directly for testing, __file__ might be different.
# For robustness if utils.py is in project root:
if __name__ == '__main__':
    # Test functions
    print(f"Attempting to load data from: {CSV_FILE_PATH}")
    df = load_grading_data()
    if df is not None:
        print("Grading data loaded successfully.")
        print("Available columns:", df.columns.tolist())
        print("Unique machine types:", get_all_machine_types())
        print("Entry-level questions (2):", get_questions_from_csv(num_questions=2, is_entry_level=True, grades=['C']))
        print("Single Needle questions (2, Grade A):", get_questions_from_csv(num_questions=2, machine_types=['Single Needle'], grades=['A']))
        print("Overlock questions (1):", get_questions_from_csv(num_questions=1, machine_types=['Overlock']))
    else:
        print("Failed to load grading data.") 
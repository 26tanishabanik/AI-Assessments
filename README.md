# Tailoring Assessment Bot

This application evaluates tailoring candidates based on their answers to industry-specific questions, using a grading system from A* to C as defined in the question bank CSV.

## Features

- Candidate data collection
- Questions pulled directly from a CSV file
- Answer evaluation against reference answers
- Automatic grading based on 90% similarity threshold
- Comprehensive assessment reports
- Streamlit web interface for easy interaction

## Setup Instructions

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Make sure you have a valid Google API key for Gemini models
   - Export it as an environment variable:
     ```
     export GOOGLE_API_KEY="your_api_key_here"
     ```
   - Or you can enter it in the Streamlit interface when prompted

3. Ensure your question data is in CSV format:
   - Place your question CSV at `data/Question Bank - For RAG  - Question & Answer for Phython Code.csv`
   - The CSV should have columns for Grade, Machine, Material, Job, Product, Situational questions, and Answers

## Running the Application

### Streamlit Web Interface (Recommended)

Run the Streamlit app:
```
streamlit run app.py
```

This will open a web interface in your browser where you can:
1. Enter your Google API key
2. Fill in candidate details
3. Answer tailoring questions
4. See evaluations with similarity scores
5. Get a final assessment report with overall grade

### Command Line Interface

Alternatively, you can use the CLI version:
```
python main.py
```

## Grading Logic

The assessment bot uses a simplified grading approach:
1. Each question has an original grade in the CSV (A*, A, B+, B, or C)
2. If the candidate's answer has ≥90% similarity to the reference answer, they receive the full grade
3. If the similarity is <90%, they receive one grade lower (e.g., A* → A)
4. The overall grade is calculated as a weighted average of individual grades

## Customization

- Change the number of questions in `app.py` by modifying the `num_questions` parameter
- Adjust the similarity threshold (currently 90%) in the code if needed
- Modify the grading scale in the `grade_down` function in `utils.py`

## Troubleshooting

If you encounter issues:
- Check that your Google API key is valid and has access to Gemini models
- Ensure the CSV file is formatted correctly and available in the data directory
- Make sure you have all required dependencies installed 
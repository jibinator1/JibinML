import google.generativeai as genai
import os
import warnings
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
warnings.filterwarnings("ignore")
sys.stderr = open(os.devnull, 'w')

def gemini_answer(context, user_question):
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return "Error: GOOGLE_API_KEY environment variable not set. Please set it."
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')

        refinement_prompt = f"""
        Given the following information:
        {context}
        User Question: {user_question}
        Refine the user's question to be more specific or extract the key information needed to answer it. 
        Provide the refined question or key information in a short, clear statement. 
        Do not add any explanations or additional context.
        """

        refinement_response = model.generate_content(refinement_prompt)
        if not refinement_response.text:
            return "Error: Could not refine the question."

        refined_question = refinement_response.text

        answer_prompt = f"""
        Given the following information:
        {context}

        Refined Question: {refined_question}

        Answer the refined question in less than 2 short sentences. 
        Provide a brief, SHORT, and direct answer with no extra details or explanations. 
        Preferred if data is in points. Do not include any interpretation or recommendations. 
        If GPA is mentioned, it is out of 12.
        Instead of ananymous, replace with Jibin in 3rd person
        """

        answer_response = model.generate_content(answer_prompt)
        if not answer_response.text:
            return "Error: Could not generate an answer."

        return answer_response.text

    except Exception as e:
        return f"An error occurred: {e}"
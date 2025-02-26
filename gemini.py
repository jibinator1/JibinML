import google.generativeai as genai
import os

def gemini_answer(context, user_question):
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return "Error: GOOGLE_API_KEY environment variable not set. Please set it."
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')

        prompt_generation_prompt = f"using this conext{context}, answer this question:{user_question}"

        prompt_generation_response = model.generate_content(prompt_generation_prompt)
        if prompt_generation_response.text:
            improved_prompt_text = prompt_generation_response.text
        else:
            return "Error: Could not generate an improved prompt."

        answer_response = model.generate_content(improved_prompt_text)
        return answer_response.text

    except Exception as e:
        return f"An error occurred: {e}"
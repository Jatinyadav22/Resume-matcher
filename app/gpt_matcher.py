import os
import requests
import json
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
api_key = os.getenv("AIPIPE_API_KEY")  # Change here

if not api_key:
    raise EnvironmentError("AIPIPE_API_KEY not found. Make sure it is in your .env file.")

def get_resume_match_score(resume_text, job_description):
    url = "https://aipipe.org/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4.1-nano",  # or gpt-3.5-turbo if only that is available
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a professional HR assistant. Your job is to evaluate how well a candidate’s resume matches a given job description.\n"
                    "Return a JSON with:\n"
                    "- a total score (0–100)\n"
                    "- sub-scores: skills_match, experience_match, education_match, keyword_match\n"
                    "- a short overall reason\n"
                    "\nThe score must reflect:\n"
                    "- Relevance of skills to JD\n"
                    "- Match in years of experience\n"
                    "- Related education/degrees\n"
                    "- Keyword coverage (e.g., job-specific terms)\n"
                    "\nOnly return valid JSON!"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Job Description:\n{job_description}\n\n"
                    f"Resume:\n{resume_text}\n\n"
                    "Now evaluate the match. Return a JSON like this:\n"
                    "{\n"
                    "  \"score\": 83,\n"
                    "  \"skills_match\": 80,\n"
                    "  \"experience_match\": 90,\n"
                    "  \"education_match\": 75,\n"
                    "  \"keyword_match\": 85,\n"
                    "  \"reason\": \"Strong experience and keyword match; education could be more aligned.\"\n"
                    "}"
                )
            }
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        output = response.json()
        content = output["choices"][0]["message"]["content"]

        result = json.loads(content)
        if isinstance(result, dict) and "score" in result and "reason" in result:
            return result
        else:
            return {"score": 0, "reason": "Invalid format returned by GPT."}

    except requests.exceptions.HTTPError as http_err:
        return {"score": 0, "reason": f"HTTP Error: {str(http_err)} - {response.text}"}
    except requests.exceptions.RequestException as req_err:
        return {"score": 0, "reason": f"Request Error: {str(req_err)}"}
    except json.JSONDecodeError:
        return {"score": 0, "reason": "Failed to parse JSON response from GPT."}
    except Exception as e:
        return {"score": 0, "reason": f"Unexpected error: {str(e)}"}
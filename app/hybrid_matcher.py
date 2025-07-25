from app.cosine_matcher import cosine_match_score
from app.gpt_matcher import get_resume_match_score
from app.key_matcher import keyword_match_score
import re

def extract_years(text):
    matches = re.findall(r'(\d+)\s*\+?\s*years?', text.lower())
    return max([int(m) for m in matches], default=0)

def experience_match_score(resume_text, jd_text):
    resume_exp = extract_years(resume_text)
    jd_exp = extract_years(jd_text)

    if jd_exp == 0:
        return {"score": 0, "reason": "JD does not specify experience requirement."}

    if resume_exp >= jd_exp:
        return {"score": 5, "reason": f"Resume meets experience requirement ({resume_exp} vs {jd_exp})."}
    else:
        return {"score": -5, "reason": f"Resume lacks required experience ({resume_exp} vs {jd_exp})."}

def hybrid_match_score(resume_text, job_description):
    # Cosine similarity
    cosine = cosine_match_score(resume_text, job_description)

    # GPT-based similarity (safe fallback)
    try:
        gpt = get_resume_match_score(resume_text, job_description)
    except Exception:
        gpt = {"score": 0, "reason": "GPT scoring failed or not available."}

    # Experience score
    exp = experience_match_score(resume_text, job_description)

    # Keyword match score
    keyword = keyword_match_score(resume_text, job_description)

    # Weighted final score
    final_score = round(
        (0.3 * cosine["score"]) +
        (0.3 * gpt["score"]) +
        (0.2 * exp["score"]) +
        (0.2 * keyword["score"]),
        2
    )

    reason = (
        f"Hybrid Scoring:\n"
        f"- GPT Score: {gpt['score']} → {gpt['reason']}\n"
        f"- Cosine Score: {cosine['score']} → {cosine['reason']}\n"
        f"- Experience Score: {exp['score']} → {exp['reason']}\n"
        f"- Keyword Score: {keyword['score']} → {keyword.get('reason', '')}"
    )

    return {
        "score": final_score,
        "reason": reason
    }

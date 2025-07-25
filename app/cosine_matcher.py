from sentence_transformers import SentenceTransformer, util
import re

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def extract_years(text):
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*\+?\s*years?', text.lower())
    return max([int(float(m)) for m in matches], default=0)

def keyword_overlap_ratio(resume_text, job_description):
    resume_words = set(re.findall(r'\b\w+\b', resume_text.lower()))
    jd_words = set(re.findall(r'\b\w+\b', job_description.lower()))
    overlap = resume_words & jd_words
    return len(overlap) / len(jd_words) if jd_words else 0

def cosine_match_score(resume_text, job_description):
    # Cosine similarity of entire text
    res_emb = model.encode(resume_text, convert_to_tensor=True)
    jd_emb = model.encode(job_description, convert_to_tensor=True)
    semantic_score = util.cos_sim(res_emb, jd_emb).item() * 100
    semantic_score = round(semantic_score, 2)

    # Experience comparison
    resume_exp = extract_years(resume_text)
    jd_exp = extract_years(job_description)
    exp_score = 0
    if jd_exp > 0:
        if resume_exp >= jd_exp:
            exp_score = 5
            exp_reason = f"Meets experience requirement ({resume_exp} vs {jd_exp})."
        else:
            exp_score = -5
            exp_reason = f"Lacks experience ({resume_exp} vs {jd_exp})."
    else:
        exp_reason = "No experience requirement specified in JD."

    # Keyword overlap
    keyword_score = keyword_overlap_ratio(resume_text, job_description)
    keyword_bonus = 3 if keyword_score > 0.2 else 0
    keyword_reason = "Good keyword match." if keyword_bonus else "Poor keyword match."

    final_score = round(min(max(semantic_score + exp_score + keyword_bonus, 0), 100), 2)

    return {
        "score": final_score,
        "semantic_score": semantic_score,
        "experience_score": exp_score,
        "keyword_score": keyword_bonus,
        "reason": f"{exp_reason} | {keyword_reason}",
        "experience_reason": exp_reason,
        "keyword_reason": keyword_reason
    }

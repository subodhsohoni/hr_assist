from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import warnings
import logging
import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")


class CVRequest(BaseModel):
    cv_summary: str


class ScreeningResponse(BaseModel):
    assessment: str
    decision: Dict[str, Any]


def get_llm() -> ChatOpenAI:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")

    return ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0.2,
        max_tokens=300,
    )


def build_chains():
    # Step 1: Write a plain-text assessment
    assessment_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a hiring manager. Review the candidate CV summary and write a concise plain-text assessment "
                "covering strengths, gaps, and cultural fit. Keep it under 120 words.",
            ),
            ("human", "CV summary:\n{cv_summary}"),
        ]
    )

    # Step 2: Turn that assessment into a structured JSON decision
    decision_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a talent acquisition coordinator. Based on the assessment, return ONLY valid JSON with these exact keys: "
                "shortlist (boolean), interview_round (HR/Technical/Skip), reason (one sentence). No preamble.",
            ),
            ("human", "Assessment:\n{assessment}"),
        ]
    )

    llm = get_llm()
    assessment_chain = assessment_prompt | llm | StrOutputParser()
    decision_chain = decision_prompt | llm | JsonOutputParser()
    chain = assessment_chain | decision_chain
    return assessment_chain, decision_chain, chain


def screen_cv_summary(cv_summary: str) -> Dict[str, Any]:
    assessment_chain, decision_chain, chain = build_chains()

    assessment = assessment_chain.invoke({"cv_summary": cv_summary})
    decision = decision_chain.invoke({"assessment": assessment})
    result = chain.invoke({"cv_summary": cv_summary})

    return {
        "assessment": assessment,
        "decision": result,
    }


app = FastAPI(title="HR Job Screener API", version="1.0.0")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/screen-cv", response_model=ScreeningResponse)
def screen_cv_endpoint(request: CVRequest):
    try:
        result = screen_cv_summary(request.cv_summary)
        return ScreeningResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Screening failed: {str(exc)}") from exc


if __name__ == "__main__":
    if OPENAI_API_KEY:
        # Example CV summary for the job screener
        example_cv_summary = (
            "Candidate: Priya Sharma. 6 years of experience in product analytics and operations at SaaS firms. "
            "Led a 12-person analytics team, improved onboarding conversion by 18%, and has strong cross-functional collaboration."
        )

        print("[STEP 1] Candidate assessment:")
        assessment_chain, decision_chain, chain = build_chains()
        assessment = assessment_chain.invoke({"cv_summary": example_cv_summary})
        print(assessment)
        print()

        print("[STEP 2] Structured hiring decision:")
        decision = decision_chain.invoke({"assessment": assessment})
        print(json.dumps(decision, indent=2))
        print()

        print("[FULL CHAIN] End-to-end job screener:")
        result = chain.invoke({"cv_summary": example_cv_summary})
        print(json.dumps(result, indent=2))
    else:
        print("[SETUP] OPENAI_API_KEY is not configured. Starting the FastAPI service anyway; screening requests will fail until it is added.")

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

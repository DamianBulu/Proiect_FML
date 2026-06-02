from __future__ import annotations
from typing import TypedDict, List

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from AdvisorModel import AdvisorModel
from AnalyserModel import AnalyserModel
from JudgeModel import JudgeModel
from MedicalKnowledgeRepo import MedicalKnowledgeRepo
from UserKnowledgeRepo import UserKnowledgeRepo

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


def _format_chat_history(chat_history: List[dict]) -> List:
    """Convert chat history dicts (role/content) to LangChain message objects."""
    messages = []
    for msg in chat_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages

medicalRepo = MedicalKnowledgeRepo()
userRepo = UserKnowledgeRepo()
judgeModel = JudgeModel()
advisorModel = AdvisorModel()
analyserModel = AnalyserModel()

class AgentState(TypedDict):
    message: str
    chat_history: List[dict]
    message_type: str # "store_data" | "medical_advice" | "both" | "neither"

    extracted_user_info: str

    user_context: str
    medical_knowledge: str

    advice: str
    judge_feedback: str
    judge_approved: bool
    attempt: int

    final_response: str

def classify_message(state: AgentState) -> dict:
    print("\n🔍 [CLASSIFY] Classifying user message intent...")
    print(f"   Message: \"{state['message'][:100]}{'...' if len(state['message']) > 100 else ''}\"")
    #Strict System Prompt for Intent Classification(Triage)
    system_template="""You are a triage assistant in a medical application.
        Your job is to analyze the user's message and determine their intent.
        
        The intent MUST be classified into exactly ONE of these four categories:
        - "medical_advice": The user is asking for health tips, symptoms analysis, or medical information.
        - "store_data": The user is providing personal health data (e.g., age, weight, medical history, current symptoms) to be saved in their profile, without explicitly asking for advice right now.
        - "both": The user is providing personal data AND asking for medical advice in the same message.
        - "neither": The message is unrelated to healthcare, data storage, or is just a general greeting.
        
        Consider the conversation history when classifying the intent.
        
        Respond STRICTLY with a valid JSON object. Do not add any extra text or markdown formatting.
        Expected JSON format:
        {{
            "message_type": "one of the four categories mentioned above"
        }}
        """

    # Build messages: system + chat history + current user message
    formatted_prompt = [SystemMessage(content=system_template)]
    formatted_prompt.extend(_format_chat_history(state.get("chat_history", [])))
    formatted_prompt.append(HumanMessage(content=state["message"]))
    print(f"   📜 Chat history: {len(state.get('chat_history', []))} messages")

    #Invoke the Analyser Model
    print("   ⏳ Calling Analyser model for intent classification...")
    response=analyserModel.getResponse(formatted_prompt)
    raw_response=response.content if hasattr(response,"content") else str(response)
    print(f"   📨 Raw response: {raw_response[:200]}")

    #Parse JSON safely and validate the output
    try:
        clean_json = raw_response.replace("```json", "").replace("```", "").strip()
        parsed=json.loads(clean_json)
        message_type=parsed.get("message_type","neither")

        # Safety check: ensure the model didn't hallucinate a category
        if message_type not in ["store_data","medical_advice","both","neither"]:
            print(f"   ⚠️  Invalid category '{message_type}', falling back to 'neither'")
            message_type="neither"
    except Exception as e:
        #Fallback in case of parsing error
        print(f"   ⚠️  JSON parsing failed: {e}, falling back to 'neither'")
        message_type="neither"

    print(f"   ✅ Intent classified as: '{message_type}'")
    return {"message_type":message_type}

def route_by_intent(state: AgentState) -> str:
    intent = state["message_type"]
    if intent == "store_data" or intent == "both":
        print(f"\n🔀 [ROUTE] Intent='{intent}' → extracting user info")
        return "extract_user_info"
    elif intent == "medical_advice":
        print(f"\n🔀 [ROUTE] Intent='{intent}' → preparing retrieval")
        return "prepare_retrieval"
    else:
        print(f"\n🔀 [ROUTE] Intent='{intent}' → handling as unrelated")
        return "handle_neither"


def route_after_store(state: AgentState) -> str:
    if state["message_type"] == "store_data":
        print("\n🔀 [ROUTE] Data stored, confirming to user")
        return "confirm_store"
    else:
        print("\n🔀 [ROUTE] Data stored, now also preparing medical retrieval")
        return "prepare_retrieval"

def confirm_store(state: AgentState) -> dict:
    print("\n💾 [CONFIRM] Confirming data storage to user")
    return {"final_response": "Ok. I'll remember that"}

def handle_neither(state: AgentState) -> dict:
    print("\n🚫 [NEITHER] Message is not medical-related, returning default response")
    return {"final_response": "Sorry, I can only help with medical questions or storing your health data. If you have a medical question or want to update your health profile, just let me know!"}

def prepare_retrieval(state: AgentState) -> dict:
    print("\n📚 [PREPARE] Preparing to retrieve user context + medical knowledge...")
    return {}

def extract_user_info(state: AgentState) -> dict:
    print("\n🧬 [EXTRACT] Extracting user health data from message...")
    #System Propmpt for Data Extraction
    system_template="""You are a medical data extraction specialist.
        Your job is to extract relevant personal and medical information from the user's message so it can be saved to their long-term health profile.
        
        Look for:
        - Demographics (age, gender, weight, height)
        - Medical history (past diseases, surgeries, allergies)
        - Current symptoms (pain type, duration, severity, location)
        - Current medications or treatments
        
        Consider the conversation history for additional context.
        
        Extract the information clearly and concisely as a summary list. 
        If no relevant medical data is found, simply output "No new medical data provided."
        Do NOT give medical advice here, your ONLY job is to extract facts.
        """

    # Build messages: system + chat history + current user message
    formatted_prompt = [SystemMessage(content=system_template)]
    formatted_prompt.extend(_format_chat_history(state.get("chat_history", [])))
    formatted_prompt.append(HumanMessage(content=state["message"]))
    print(f"   📜 Chat history: {len(state.get('chat_history', []))} messages")

    #Invoke the Analyser Model
    print("   ⏳ Calling Analyser model for data extraction...")
    response=analyserModel.getResponse(formatted_prompt)
    extracted_info=response.content if hasattr(response,"content") else str(response)
    print(f"   ✅ Extracted info: {extracted_info[:200]}{'...' if len(extracted_info) > 200 else ''}")

    return {"extracted_user_info":extracted_info}

def store_user_context(state: AgentState) -> dict:
    print("\n💾 [STORE] Saving extracted info to user profile...")
    #Luăm informația extrasă de Analyser la pasul anterior
    extracted_info=state.get("extracted_user_info","")

    #Salvare fizica in JSON
    userRepo.save_knowledge(extracted_info)
    print("   ✅ User profile updated")
    return {}

def retrieve_user_context(state: AgentState) -> dict:
    print("\n👤 [RETRIEVE] Loading user profile from JSON...")
    #Luăm informația extrasă de Analyser la pasul anterior
    user_context=userRepo.get_knowledge()
    print(f"   ✅ User context: {user_context[:150]}{'...' if len(user_context) > 150 else ''}")

    #Punem datele în starea grafului pentru a fi folosite de Advisor
    return {"user_context":user_context}

def retrieve_medical_knowledge(state: AgentState) -> dict:
    print("\n📖 [RETRIEVE] Searching ChromaDB for relevant medical knowledge...")
    print(f"   Query: \"{state['message'][:100]}{'...' if len(state['message']) > 100 else ''}\"")
    #Cautam in baza medical folosind intrebarea curenta a pacientului
    medical_knowledge=medicalRepo.get_knowledge(state["message"])
    print(f"   ✅ Found medical knowledge ({len(medical_knowledge)} chars)")
    return {"medical_knowledge":medical_knowledge}

def generate_advice(state: AgentState) -> dict:
    attempt=state.get("attempt",0)
    print(f"\n💡 [ADVISOR] Generating medical advice (attempt {attempt + 1}/3)...")

    system_template="""You are an empathetic and professional virtual medical assistant. 
        Your role is to provide basic health advice, medical education, and support based STRICTLY on the provided medical knowledge.
        You are NOT a real doctor. Do not provide final diagnoses or prescribe medications. Always recommend consulting a real doctor for emergencies.
        
        PATIENT INFORMATION (Context):
        {user_context}
        
        RELEVANT MEDICAL KNOWLEDGE:
        {medical_knowledge}
        
        {judge_section}
        
        Answer the user's latest question directly, considering the conversation history and the data provided above. Be concise and clear.
        """

    #Handle Judge feedback loop
    judge_section=""
    if attempt>0 and state.get("judge_feedback"):
        print(f"   ⚠️  Previous attempt was rejected. Judge feedback: {state['judge_feedback'][:150]}")
        judge_section=f"""
            WARNING! Your previous response was rejected by the safety judge.
            Judge Feedback: {state["judge_feedback"]}
            Please correct your response according to this feedback!
        """

    #Build the prompt with chat history
    system_content=system_template.format(
        user_context=state.get("user_context","No patient data available"),
        medical_knowledge=state.get("medical_knowledge","No specific medical knowledge found"),
        judge_section=judge_section,
    )

    formatted_prompt = [SystemMessage(content=system_content)]
    formatted_prompt.extend(_format_chat_history(state.get("chat_history", [])))
    formatted_prompt.append(HumanMessage(content=state["message"]))
    print(f"   📜 Chat history: {len(state.get('chat_history', []))} messages")

    #invoke the Advisor Model
    print("   ⏳ Calling Advisor model...")
    response=advisorModel.getResponse(formatted_prompt)

    #Extract raw text from LangChain AIMessage
    advice_text=response.content if hasattr(response,'content') else str(response)
    print(f"   ✅ Advice generated ({len(advice_text)} chars)")

    return {
        "advice":advice_text,
        "attempt":attempt+1
    }


def judge_advice(state: AgentState) -> dict:
    print("\n⚖️  [JUDGE] Evaluating advice for safety compliance...")
    #1.Strict System Prompt for the Judge
    judge_template="""You are a medical safety auditor (Judge).
        Your task is to evaluate the advice given by a virtual assistant (Advisor) to a patient.
        
        STRICT SAFETY RULES:
        1. The assistant MUST NOT provide a final diagnosis.
        2. The assistant MUST NOT prescribe medications.
        3. If the situation sounds like an emergency (e.g., severe chest pain, heavy bleeding), the assistant MUST recommend calling emergency services or visiting a doctor immediately.
        
        Assistant's Advice to evaluate:
        "{advice}"
        
        Evaluate if this advice strictly follows the rules. 
        You MUST respond STRICTLY with a valid JSON object. Do not include any other text, markdown formatting (like ```json), or explanations outside the JSON.
        
        Expected JSON format:
        {{
            "judge_approved": true or false,
            "judge_feedback": "Your short explanation of why you approved or rejected the advice."
        }}
        """

    prompt=ChatPromptTemplate.from_messages([
        ("system",judge_template)
    ])
    formatted_prompt=prompt.format_messages(advice=state["advice"])

    print("   ⏳ Calling Judge model...")
    response=judgeModel.getResponse(formatted_prompt)

    raw_response=response.content if hasattr(response,"content") else str(response)
    print(f"   📨 Raw judge response: {raw_response[:200]}")

    #Parse the JSON output
    try:
        clean_json=raw_response.replace("```json", "").replace("```", "").strip()
        evaluation=json.loads(clean_json)

        approved=evaluation.get("judge_approved",False)
        feedback=evaluation.get("judge_feedback","Could not parse judge feedback")
    except Exception as e:
        print(f"   ⚠️  Judge JSON parsing failed: {e}")
        approved=False
        feedback=f"Judge parsing error: {str(e)}. Please be safer and more cautious."

    status = "✅ APPROVED" if approved else "❌ REJECTED"
    print(f"   {status} — {feedback[:150]}")

    return {
        "judge_feedback":feedback,
        "judge_approved":approved
    }

def route_after_judge(state: AgentState) -> str:
    if state["judge_approved"]:
        print("\n🔀 [ROUTE] Judge approved → finalizing response")
        return "finalize_response"
    elif state["attempt"]>= 3:
        print("\n🔀 [ROUTE] Max attempts reached (3/3) → fallback response")
        return "fallback_response"
    else:
        print(f"\n🔀 [ROUTE] Judge rejected, retrying (attempt {state['attempt']}/3)")
        return "generate_advice"

def finalize_response(state: AgentState) -> dict:
    print("\n🏁 [DONE] Finalizing approved response")
    return {"final_response": state["advice"]}

def fallback_response(state: AgentState) -> dict:
    print("\n🏁 [FALLBACK] All attempts rejected by judge, returning safe fallback")
    return {"final_response": "Sorry, but I'm not able to help you with that. If you're worried you should go see a doctor."}

def build_graph() -> CompiledStateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("classify_message", classify_message)
    graph.add_node("handle_neither", handle_neither)
    graph.add_node("prepare_retrieval", prepare_retrieval)
    graph.add_node("extract_user_info", extract_user_info)
    graph.add_node("store_user_context", store_user_context)
    graph.add_node("confirm_store", confirm_store)
    graph.add_node("retrieve_user_context", retrieve_user_context)
    graph.add_node("retrieve_medical_knowledge", retrieve_medical_knowledge)
    graph.add_node("generate_advice", generate_advice)
    graph.add_node("judge_advice", judge_advice)
    graph.add_node("finalize_response", finalize_response)
    graph.add_node("fallback_response", fallback_response)

    graph.add_edge(START, "classify_message")
    graph.add_conditional_edges(
        "classify_message",
        route_by_intent,
        {
            "handle_neither": "handle_neither",
            "prepare_retrieval": "prepare_retrieval",
            "extract_user_info": "extract_user_info",
        },
    )

    graph.add_edge("handle_neither", END)
    graph.add_edge("prepare_retrieval", "retrieve_user_context")
    graph.add_edge("prepare_retrieval", "retrieve_medical_knowledge")

    graph.add_edge("extract_user_info", "store_user_context")
    graph.add_conditional_edges(
        "store_user_context",
        route_after_store,
        {
            "confirm_store": "confirm_store",
            "prepare_retrieval": "prepare_retrieval",
        },
    )

    graph.add_edge("confirm_store", END)

    graph.add_edge("retrieve_user_context", "generate_advice")
    graph.add_edge("retrieve_medical_knowledge", "generate_advice")

    graph.add_edge("generate_advice", "judge_advice")
    graph.add_conditional_edges(
        "judge_advice",
        route_after_judge,
        {
            "finalize_response": "finalize_response",
            "generate_advice": "generate_advice",
            "fallback_response": "fallback_response",
        },
    )

    graph.add_edge("finalize_response", END)
    graph.add_edge("fallback_response", END)

    return graph.compile()

class ChatService:

    def __init__(self):
        self.graph = build_graph()

    def getResponse(self, message: str, chat_history: list[dict] | None = None):
        print("\n" + "="*60)
        print("🚀 [START] New request received")
        print(f"   Message: \"{message[:120]}{'...' if len(message) > 120 else ''}\"")
        print(f"   Chat history: {len(chat_history or [])} messages")
        print("="*60)

        initial_state: AgentState = {
            "message": message,
            "chat_history": chat_history or [],
            "message_type": "",
            "extracted_user_info": "",
            "user_context": "",
            "medical_knowledge": "",
            "advice": "",
            "judge_feedback": "",
            "judge_approved": False,
            "attempt": 0,
            "final_response": "",
        }

        result = self.graph.invoke(initial_state)

        final = result["final_response"]
        print("\n" + "="*60)
        print("🏁 [END] Response ready")
        print("="*60 + "\n")
        yield final
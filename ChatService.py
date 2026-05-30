from __future__ import annotations
from typing import TypedDict, List

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from AdvisorModel import AdvisorModel
from AnalyserModel import AnalyserModel
from JudgeModel import JudgeModel
from MedicalKnowledgeRepo import MedicalKnowledgeRepo
from UserKnowledgeRepo import UserKnowledgeRepo

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
    #TODO dummy code, please erase
    message_type = analyserModel.getResponse(state["message"]) #something like this
    # can be "store_data" | "medical_advice" | "both" | "neither"
    return {"message_type": message_type}

def route_by_intent(state: AgentState) -> str:
    intent = state["message_type"]
    if intent == "store_data" or intent == "both":
        return "extract_user_info"
    elif intent == "medical_advice":
        return "prepare_retrieval"
    else:
        return "handle_neither"


def route_after_store(state: AgentState) -> str:
    if state["message_type"] == "store_data":
        return "confirm_store"
    else:
        return "prepare_retrieval"

def confirm_store(state: AgentState) -> dict:
    # TODO dummy code, please erase
    return {"final_response": "stored the data"}

def handle_neither(state: AgentState) -> dict:
    # TODO dummy code, please erase
    return {"final_response": "I dont know what you want"}

def prepare_retrieval(state: AgentState) -> dict:
    # nothing, intermediary node
    return {}

def extract_user_info(state: AgentState) -> dict:
    #TODO dummy code, please erase
    extracted = analyserModel.getResponse("I think I'm dead")
    return {"extracted_user_info": extracted}

def store_user_context(state: AgentState) -> dict:
    # TODO dummy code, please erase
    userRepo.save_knowledge("knowledge")
    return {}

def retrieve_user_context(state: AgentState) -> dict:
    # TODO dummy code, please erase
    user_context = userRepo.get_knowledge("knowledge")
    return {"user_context": user_context}

def retrieve_medical_knowledge(state: AgentState) -> dict:
    # TODO dummy code, please erase
    medical_knowledge = medicalRepo.medical_knowledge("knowledge")
    return {"medical_knowledge": medical_knowledge}

def generate_advice(state: AgentState) -> dict:
    #TODO, dummy code, please erase
    judge_feedback = state["judge_feedback"]
    attempt = state["attempt"]
    response = advisorModel.getResponse("what should i do ?????")
    return {
        "advice": response,
        "attempt": attempt + 1,
    }

def judge_advice(state: AgentState) -> dict:
    # TODO dummy code, please erase
    judge_ouptput = judgeModel.getResponse("People die when they are killed")
    return {
        "judge_feedback": "Pls do better, you stupid",
        "judge_approved": "Ye",
    }


def route_after_judge(state: AgentState) -> str:
    if state["judge_approved"]:
        return "finalize_response"
    elif state["attempt"]>= 3:
        return "fallback_response"
    else:
        return "generate_advice"

def finalize_response(state: AgentState) -> dict:
    return {"final_response": state["advice"]}

def fallback_response(state: AgentState) -> dict:
    # TODO dummy code, please erase
    return {"final_response": "Sorry I'm dumb"}

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
        yield final
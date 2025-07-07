import os
import uuid
from typing import List, Tuple, Literal, Annotated

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.pydantic_v1 import BaseModel
from pydantic import BaseModel
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
import gradio as gr

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client

proxy_client = get_proxy_client('gen-ai-hub')

import os
import json
from dotenv import load_dotenv, find_dotenv

load_dotenv()


class Config:
    def __init__(self):
        load_dotenv()
        # # 환경 변수에서 경로 및 프로파일 읽어오기
        aicore_home = os.getenv('AICORE_HOME')
        profile = os.getenv('AICORE_PROFILE', 'default')
        config_path = os.path.join(aicore_home, f"config.json")
        from ai_core_sdk.ai_core_v2_client import AICoreV2Client
        # 구성 파일에서 값 읽어오기
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)

        # 클라이언트 초기화
        ai_core_client = AICoreV2Client(
            base_url=config['AICORE_BASE_URL'],
            auth_url=config['AICORE_AUTH_URL'],
            client_id=config['AICORE_CLIENT_ID'],
            client_secret=config['AICORE_CLIENT_SECRET']
        )

        print("AI Core client initialized successfully!")


class UserStoryCriteria(BaseModel):
    weekday_overtime: str
    weekday_night: str
    weekend_overtime: str
    weekend_night: str


class StateSchema(TypedDict):
    messages: Annotated[list, add_messages]
    created_user_story: bool
    waiting_for_approval: bool  # 승인 대기 상태 추가


class Workflow:
    def __init__(self):
        self.prompt_system_task = """Your job is to gather over time hours information from the employees about compensatory leave hours they want to calculate.

        You should obtain the following information from them:

        - weekday_overtime is total Over time of weekdays of a month: 월간 평일 총 연장 근무 시간. 이것은 평일 18시부터 22시 사이에 발생한 총 연장 근무 시간. 시/분 단위로 근무 시간을 수집 한다. (예) 1시간, 1시간 30분, 20분 등. 
        - weekday_night is total Night work time of weekdays of a month : 월간 평일 총 야간 근무 시간. 이것은 평일 22시부터 다음날 6시 사이에 발생한 총 야간 근무 시간. 시/분 단위로 근무 시간을 수집 한다. (예) 1시간, 1시간 30분, 20분 등.
        - weekend_overtime is total Over time of weekends of a month: 월간 주말 총 연장 근무 시간. 이것은 주말 6시부터 22시 사이에 발생한 총 주말 연장 근무 시간. 시/분 단위로 근무 시간을 수집 한다.(예) 1시간, 1시간 30분, 20분 등. 
        - weekend_night is total Night work time of weekends of a month : 월간 주말 총 야간 근무 시간. 이것은 주말 22시부터 다음날 6시 사이에 발생한 총 주말 야간 근무 시간. 시/분 단위로 근무 시간을 수집 한다. (예) 1시간, 1시간 30분, 20분 등.

        If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess. 
        Whenever the user responds to one of the criteria, evaluate if it is detailed enough to be a criterion of compensatory Leave Hours Calculation . If not, ask questions to help the user better detail the criterion.
        Do not overwhelm the user with too many questions at once; ask for the information you need in a way that they do not have to write much in each response. 
        Always remind them that if they do not know how to answer something, you can help them.

        After you are able to discern all the information, call the relevant tool."""

        self.prompt_approval_task = """The user is asking for approval to proceed with the calculation. 
        Check if the user's response indicates approval or rejection.

        If the user approves (예, 승인, 맞습니다, 네, yes, ok, 확인, 계산해주세요, 진행해주세요), respond with a confirmation message.
        If the user rejects or wants to modify (아니오, 수정, 다시, 틀렸습니다, no, 변경), ask them what they would like to modify.

        Be clear and helpful in your response."""

        self.prompt_summarize_input = """Based on the user's input information, provide a clear and concise summary of the overtime hours they want to calculate for compensatory leave.

                Please summarize the following information in Korean:
                - 평일 연장 근무 시간 (Weekday overtime hours)
                - 평일 야간 근무 시간 (Weekday night work hours)
                - 주말 연장 시간 (Weekend overtime hours)
                - 주말 야간 근무 시간 (Weekend night work hours)

                Format your summary as follows:
                === 입력 정보 요약 ===
                평일 연장 근무 시간 : [시간]
                평일 야간 근무 시간: [시간]
                주말 연장 근무 시간: [시간]
                주말 야간 근무 시간: [시간]
    
                위 정보가 정확한지 확인해 주세요.

                ✅ 정보가 정확하다면 '승인' 또는 '예' 또는 '계산해주세요'라고 답변해 주세요.
                ❌ 정보를 수정하고 싶다면 '수정' 또는 '아니오' 또는 '다시'라고 답변해 주세요.

                어떻게 진행하시겠습니까?

                {reqs}"""

        self.prompt_generate_user_story = """Based on the following information, summarize all the working hours and calculate compensatory leave hours. 
        Follow these steps very carefully:
        1. Set fixed Over Time standards:
           Weekday fixed Over Time  = 30 hours
           Weekend fixed Over Time = 10 hours

        2. Calculate weekday compensatory leave:
           a. Excess weekday overtime = MAX(0, weekday_overtime - 30)
           b-1. no exceeding the weekday fixed over time, Weekday compensatory leave  = (Excess weekday overtime × 1.5) + (Weekday night work × 0.5)
           b-2. exceeding the weekday fixed over time, Weekday compensatory leave  = (Excess weekday overtime × 1.5) + (Weekday night work × 2)

        3. Calculate weekend compensatory leave:
           a. Excess weekend overtime = MAX(0, weekend_overtime - 10)
           b-1. If no exceeding the weekend fixed over time, Weekend compensatory leave = (excess weekend overtime × 1.5) + (Weekend night work × 0.5) 
           b-2. If exceeding the weekend fixed over time, Weekend compensatory leave = (excess weekend overtime × 1.5) + (Weekend night work × 2)

        4. Calculate total compensatory leave:
           Total compensatory leave hours = Weekday compensatory leave + Weekend compensatory leave

        5. Round all calculated hours to two decimal places.

        After performing these calculations, provide your answer in the following format:

        === 보상 휴가 시간 계산 결과 ===
        평일 보상 휴가 시간: [calculated hours] 시간
        주말, 휴일 보상 휴가 시간: [calculated hours] 시간
        총 보상 휴가 시간: [calculated hours] 시간
        =============================

        주어진 정보를 바탕으로 보상 휴가 시간을 계산하였습니다.
        {reqs}"""

        self.config = Config()

        self.llm = ChatOpenAI(proxy_model_name='gpt-4o-mini', proxy_client=proxy_client)

        self.llm_with_tool = self.llm.bind_tools([UserStoryCriteria])
        self.memory = MemorySaver()
        self.workflow = self.create_workflow()
        self.graph_memory = self.workflow.compile(checkpointer=self.memory)

    def create_workflow(self):
        workflow = StateGraph(StateSchema)
        workflow.add_node("talk_to_user", self.call_llm)
        workflow.add_node("finalize_dialogue", self.finalize_dialogue)
        workflow.add_node("create_user_story", self.call_model_to_generate_user_story)
        workflow.add_node("handle_approval", self.handle_approval)

        workflow.add_edge(START, "talk_to_user")
        workflow.add_conditional_edges("talk_to_user", self.define_next_action)
        workflow.add_edge("finalize_dialogue", END)  # finalize_dialogue에서 요약 후 대기
        workflow.add_conditional_edges("handle_approval", self.check_approval)
        workflow.add_edge("create_user_story", END)

        return workflow

    def domain_state_tracker(self, messages):
        return [SystemMessage(content=self.prompt_system_task)] + messages

    def approval_state_tracker(self, messages):
        return [SystemMessage(content=self.prompt_approval_task)] + messages

    def call_llm(self, state: StateSchema):
        # 승인 대기 상태인지 확인
        if state.get("waiting_for_approval", False):
            messages = self.approval_state_tracker(state["messages"])
            response = self.llm.invoke(messages)
            return {"messages": [response]}
        else:
            messages = self.domain_state_tracker(state["messages"])
            response = self.llm_with_tool.invoke(messages)
            return {"messages": [response]}

    def finalize_dialogue(self, state: StateSchema):
        """정보 수집 완료 후 요약 출력 및 승인 요청"""
        tool_call = None
        for m in state["messages"]:
            if isinstance(m, AIMessage) and m.tool_calls:
                tool_call = m.tool_calls[0]["args"]
                break

        if tool_call:
            # 요약 및 승인 요청 메시지 생성
            summary_prompt = [SystemMessage(content=self.prompt_summarize_input.format(reqs=tool_call))]
            response = self.llm.invoke(summary_prompt)

            # Tool 응답 추가
            tool_response = ToolMessage(
                content="Summary generated and waiting for approval",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )

            return {
                "messages": [tool_response, response],
                "waiting_for_approval": True
            }
        else:
            return {
                "messages": [AIMessage(content="사용자 입력 정보를 찾을 수 없습니다.")],
                "waiting_for_approval": False
            }

    def handle_approval(self, state: StateSchema):
        """승인 상태 처리"""
        messages = self.approval_state_tracker(state["messages"])
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def check_approval(self, state: StateSchema) -> Literal["create_user_story", END]:
        """사용자 승인 상태 확인"""
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            user_response = last_message.content.lower().strip()

            # 승인 키워드 확인
            approval_keywords = ['승인', '예', '맞습니다', '맞아요', '네', 'yes', 'ok', '확인', '계산해주세요', '진행해주세요']
            rejection_keywords = ['수정', '아니오', '다시', '틀렸습니다', '아니요', 'no', '변경']

            if any(keyword in user_response for keyword in approval_keywords):
                return "create_user_story"
            elif any(keyword in user_response for keyword in rejection_keywords):
                return END

        # 기본값으로 talk_to_user로 이동
        return "talk_to_user"

    def build_prompt_to_generate_user_story(self, messages: list):
        tool_call = None
        for m in messages:
            if isinstance(m, AIMessage) and m.tool_calls:
                tool_call = m.tool_calls[0]["args"]
                break
        return [SystemMessage(content=self.prompt_generate_user_story.format(reqs=tool_call))]

    def call_model_to_generate_user_story(self, state):
        messages = self.build_prompt_to_generate_user_story(state["messages"])
        response = self.llm.invoke(messages)
        return {"messages": [response], "created_user_story": True}

    def define_next_action(self, state) -> Literal["finalize_dialogue", "handle_approval", END]:
        # 승인 대기 상태라면 handle_approval로 이동
        if state.get("waiting_for_approval", False):
            return "handle_approval"

        # 툴 호출이 있다면 finalize_dialogue로 이동
        messages = state["messages"]
        if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
            return "finalize_dialogue"
        else:
            return END


class ChatBot:
    def __init__(self):
        self.workflow = Workflow()
        self.thread_id = str(uuid.uuid4())

    def process_message(self, message: str, history: List[Tuple[str, str]]) -> str:
        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            inputs = {"messages": [HumanMessage(content=message)]}
            result = self.workflow.graph_memory.invoke(inputs, config=config)
            if "messages" in result:
                print(f"스레드 ID: {self.thread_id}")
                for msg in result["messages"]:
                    msg.pretty_print()
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    return last_message.content
            return "응답을 생성하지 못했습니다."
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return "죄송합니다. 응답을 생성하는 동안 오류가 발생했습니다. 다시 시도해 주세요."

    def chat(self, message: str, history: List[Tuple[str, str]]) -> str:
        print(f"Thread ID: {self.thread_id}")
        return self.process_message(message, history)


if __name__ == "__main__":
    chatbot = ChatBot()
    example_questions = [
        "어떻게 시작할 수 있을까?",
        "어떻게 요청하면 될까?",
    ]

    demo = gr.ChatInterface(
        fn=chatbot.chat,
        title="Compensatory Leave Hours Calculator",
        description="직원의 보상 휴가 시간 계산을 도와 줍니다. ",
        examples=example_questions,
        theme=gr.themes.Soft()
    )
    demo.launch()
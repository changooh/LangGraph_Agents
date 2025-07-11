import os
import uuid
from typing import List, Tuple, Literal, Annotated, Optional
import re
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
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

        # azure open ai
        # os.environ["OPENAI_API_VERSION"] = os.getenv('AZURE_OPENAI_VERSION')
        # os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv('AZURE_OPENAI_END_POINT')
        # os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('AZURE_OPENAI_KEY')


class TimeCriteria(BaseModel):
    weekday_overtime: str
    weekday_night: str
    weekend_overtime: str
    weekend_night: str

class StateSchema(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str


# class TimeCalculator(BaseTool):
#     """ input schema"""
#     name = "compensatory_hours_calculator"
#     query: str = Field(..., description="The summary information to calculate compensatory leave hours ")
#     description = "사용자 근무 시간 정보를 바탕으로 보상 휴가 시간을 계산합니다."



class Workflow:
    def __init__(self):
        self.prompt_system_task = """ 당신은 보상 휴가 시간 계산 에이전트입니다. 당신의 역할은 직원들로부터 초과 근무 시간 정보를 수집하고, 이를 요약한 뒤, 보상 휴가 시간을 계산하는 것입니다.
        You should obtain the following information from them:
        - Total over time of weekdays of a month: 월간 평일 총 연장 근무 시간. 이것은 평일 18시부터 22시 사이에 발생한 총 연장 근무 시간. 시/분 단위로 근무 시간을 수집 한다. (예) 1시간, 1시간 30분, 20분 등. 
        - Total night work time of weekdays of a month : 월간 평일 총 야간 근무 시간. 이것은 평일 22시부터 다음날 6시 사이에 발생한 총 야간 근무 시간. 시/분 단위로 근무 시간을 수집 한다. (예) 1시간, 1시간 30분, 20분 등.
        - Total over time of weekends of a month: 월간 주말 총 연장 근무 시간. 이것은 주말 6시부터 22시 사이에 발생한 총 주말 연장 근무 시간. 시/분 단위로 근무 시간을 수집 한다.(예) 1시간, 1시간 30분, 20분 등. 
        - Total night work time of weekends of a month : 월간 주말 총 야간 근무 시간. 이것은 주말 22시부터 다음날 6시 사이에 발생한 총 주말 야간 근무 시간. 시/분 단위로 근무 시간을 수집 한다. (예) 1시간, 1시간 30분, 20분 등.

        If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess. 
        Whenever the user responds to one of the criteria, evaluate if it is detailed enough to be a criterion of compensatory Leave Hours Calculation . If not, ask questions to help the user better detail the criterion.
        Do not overwhelm the user with too many questions at once; ask for the information you need in a way that they do not have to write much in each response. 
        Always remind them that if they do not know how to answer something, you can help them.

        """
        self.backup = """ After you are able to discern all the information, call the relevant tool. """


        self.prompt_generate_summary = """provide a clear and concise summary of the overtime hours they want to calculate for compensatory leave.
        Please summarize the following information in Korean:
        - 평일 연장 근무 시간 (Weekday overtime hours)
        - 평일 야간 근무 시간 (Weekday night work hours)
        - 주말 연장 시간 (Weekend overtime hours)
        - 주말 야간 근무 시간 (Weekend night work hours)
        
        Always ask "보상 휴가 시간을 계산해 드릴까요?" in the end of the response.
        
        output format summary as follows:
        === 입력 정보 요약 ===
        평일 연장 근무 시간 : 
        평일 야간 근무 시간: 
        주말 연장 근무 시간: 
        주말 야간 근무 시간: 

        {reqs}"""

        self.prompt_generate_calculation = """Based on the following information, summarize all the working hours and calculate compensatory leave hours. 
        Follow these steps very carefully:
            
        1. Set fixed Over Time standards:
           Weekday fixed Over Time = 30 hours
           Weekend fixed Over Time = 10 hours
        
        2. Calculate weekday compensatory leave:
           a. Excess weekday overtime = MAX(0, weekday_overtime - 30)
           b-1. no exceeding the weekday fixed over time, Weekday compensatory leave = (Excess weekday overtime × 1.5) + (Weekday night work × 0.5)
           b-2. exceeding the weekday fixed over time, Weekday compensatory leave = (Excess weekday overtime × 1.5) + (Weekday night work × 2)
        
        3. Calculate weekend compensatory leave:
           a. Excess weekend overtime = MAX(0, weekend_overtime - 10)
           b-1. If no exceeding the weekend fixed over time, Weekend compensatory leave = (excess weekend overtime × 1.5) + (Weekend night work × 0.5) 
           b-2. If exceeding the weekend fixed over time, Weekend compensatory leave = (excess weekend overtime × 1.5) + (Weekend night work × 2)
        
        4. Calculate total compensatory leave:
           Total compensatory leave hours = Weekday compensatory leave + Weekend compensatory leave
        
        5. Round all calculated hours to two decimal places.provide a clear and concise summary of the overtime hours they want to calculate for compensatory leave.
        
        output format summary as follows:
        <summary>
        === 보상 휴가 시간 계산 결과 ===
        평일 보상 휴가 시간: 
        주말/휴일 보상 휴가 시간: 
        총 보상 휴가 시간: 
        주말/휴일 근무 시간에 "대체 휴일"을 적용한 경우, 주말/휴일 보상 시간은 위의 계산 결과와 다를 수 있습니다.
        </summary>
        
        {reqs}"""


        self.config = Config()
        self.llm = ChatOpenAI(proxy_model_name='gpt-4o-mini', proxy_client=proxy_client)

        # azure open ai
        # self.llm = AzureChatOpenAI(
        #     azure_deployment=os.getenv('DEPLOYMENT_NAME'),
        #     api_version=os.getenv('AZURE_OPENAI_VERSION'),
        #     temperature=0,
        # )

        self.llm_with_tool = self.llm.bind_tools([TimeCriteria])
        self.memory = MemorySaver()
        self.workflow = self.create_workflow()
        self.graph_memory = self.workflow.compile(checkpointer=self.memory)

    def create_workflow(self):
        workflow = StateGraph(StateSchema)
        workflow.add_node("talk_to_user", self.call_llm)
        workflow.add_node("finalize_dialogue", self.finalize_dialogue)
        workflow.add_node("create_user_summary", self.call_model_to_generate_summary)
        # workflow.add_node("create_calculation", self.call_model_to_generate_calculation)
        workflow.add_edge(START, "talk_to_user")
        workflow.add_conditional_edges("talk_to_user", self.define_next_action)
        workflow.add_edge("finalize_dialogue", "create_user_summary")
        workflow.add_edge("create_user_summary", END)
        # workflow.add_edge("finalize_dialogue", "create_calculation")
        # workflow.add_edge("create_calculation", END)
        return workflow

    def domain_state_tracker(self, messages):
        return [SystemMessage(content=self.prompt_system_task)] + messages

    def call_llm(self, state: StateSchema):
        messages = self.domain_state_tracker(state["messages"])
        response = self.llm_with_tool.invoke(messages)
        return {"messages": [response]}

    def finalize_dialogue(self, state: StateSchema):
        return {
            "messages": [
                ToolMessage(
                    content="Prompt generated!",
                    tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                )
            ]
        }

    def build_prompt_to_generate_summary(self, messages: list):
        tool_call = None
        other_msgs = []
        for m in messages:
            if isinstance(m, AIMessage) and m.tool_calls:
                tool_call = m.tool_calls[0]["args"]
            elif isinstance(m, ToolMessage):
                continue
            elif tool_call is not None:
                other_msgs.append(m)
        return [SystemMessage(content=self.prompt_generate_summary.format(reqs=tool_call))] + other_msgs

    def build_prompt_to_generate_calculation(self, messages: list):
        tool_call = None
        other_msgs = []
        for m in messages:
            if isinstance(m, AIMessage) and m.tool_calls:
                tool_call = m.tool_calls[0]["args"]
            elif isinstance(m, ToolMessage):
                continue
            elif tool_call is not None:
                other_msgs.append(m)
        return [SystemMessage(content=self.prompt_generate_calculation.format(reqs=tool_call))] + other_msgs

    def call_model_to_generate_calculation(self, state):
        messages = self.build_prompt_to_generate_calculation(state["messages"])
        response = self.llm.invoke(messages)

        def extract_summary(text):
            pattern = r'<summary>(.*?)</summary>'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return ""
        summary = extract_summary(response.content)
        response.content=summary

        return {"messages": [response]}

    def call_model_to_generate_summary(self, state):
        messages = self.build_prompt_to_generate_summary(state["messages"])
        response = self.llm.invoke(messages)
        return {"messages": [response], "summary":response.content}

    def define_next_action(self, state) -> Literal["finalize_dialogue", END]:
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
        "너가 무엇을 할 수 있니?",
    ]

    demo = gr.ChatInterface(
        fn=chatbot.chat,
        title="USER STORY AI ASSISTANT",
        description="개발자를 위한 유저 스토리 생성을 도와 드려요. ",
        examples=example_questions,
        theme=gr.themes.Soft()
    )
    demo.launch()

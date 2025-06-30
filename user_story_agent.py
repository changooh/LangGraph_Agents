import os
import uuid
from typing import List, Tuple, Literal, Annotated

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.pydantic_v1 import BaseModel
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
import gradio as gr


class Config:
    def __init__(self):
        load_dotenv()
        os.environ["OPENAI_API_VERSION"] = os.getenv('AZURE_OPENAI_VERSION')
        os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv('AZURE_OPENAI_END_POINT')
        os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('AZURE_OPENAI_KEY')


class UserStoryCriteria(BaseModel):
    objective: str
    success_criteria: str
    plan_of_execution: str


class StateSchema(TypedDict):
    messages: Annotated[list, add_messages]
    created_user_story: bool


class Workflow:
    def __init__(self):
        self.prompt_system_task = """Your job is to gather information from the user about the User Story they need to create.

        You should obtain the following information from them:

        - Objective: the goal of the user story. should be concrete enough to be developed in 2 weeks.
        - Success criteria the success criteria of the user story
        - Plan_of_execution: the plan of execution of the initiative

        If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess. 
        Whenever the user responds to one of the criteria, evaluate if it is detailed enough to be a criterion of a User Story. If not, ask questions to help the user better detail the criterion.
        Do not overwhelm the user with too many questions at once; ask for the information you need in a way that they do not have to write much in each response. 
        Always remind them that if they do not know how to answer something, you can help them.

        After you are able to discern all the information, call the relevant tool."""

        self.prompt_generate_user_story = """Based on the following requirements, write a good user story in korean language:

        {reqs}"""
        self.config = Config()
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv('DEPLOYMENT_NAME'),
            api_version=os.getenv('AZURE_OPENAI_VERSION'),
            temperature=0,
        )
        self.llm_with_tool = self.llm.bind_tools([UserStoryCriteria])
        self.memory = MemorySaver()
        self.workflow = self.create_workflow()
        self.graph_memory = self.workflow.compile(checkpointer=self.memory)

    def create_workflow(self):
        workflow = StateGraph(StateSchema)
        workflow.add_node("talk_to_user", self.call_llm)
        workflow.add_node("finalize_dialogue", self.finalize_dialogue)
        workflow.add_node("create_user_story", self.call_model_to_generate_user_story)
        workflow.add_edge(START, "talk_to_user")
        workflow.add_conditional_edges("talk_to_user", self.define_next_action)
        workflow.add_edge("finalize_dialogue", "create_user_story")
        workflow.add_edge("create_user_story", END)
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

    def build_prompt_to_generate_user_story(self, messages: list):
        tool_call = None
        other_msgs = []
        for m in messages:
            if isinstance(m, AIMessage) and m.tool_calls:
                tool_call = m.tool_calls[0]["args"]
            elif isinstance(m, ToolMessage):
                continue
            elif tool_call is not None:
                other_msgs.append(m)
        return [SystemMessage(content=self.prompt_generate_user_story.format(reqs=tool_call))] + other_msgs

    def call_model_to_generate_user_story(self, state):
        messages = self.build_prompt_to_generate_user_story(state["messages"])
        response = self.llm.invoke(messages)
        return {"messages": [response]}

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
        "유저 스토리가 뭐야?",
        "목표를 설정을 도와줘",
        "어떻게 요청하면 될까?",
        "목표는 사용자가 챗봇 APP을 통해서 휴가 신청을 한다.",
        "예시를 그냥 사용해라"
    ]

    demo = gr.ChatInterface(
        fn=chatbot.chat,
        title="USER STORY AI ASSISTANT",
        description="개발자를 위한 유저 스토리 생성을 도와 드려요. ",
        examples=example_questions,
        theme=gr.themes.Soft()
    )
    demo.launch()

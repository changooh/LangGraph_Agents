
from typing import List, Tuple
import os
import uuid
from langchain_google_community import CalendarToolkit
from langchain_google_community.calendar.utils import (
    build_resource_service,
    get_google_credentials,
)
from langchain_openai import AzureChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
import gradio as gr



class CalendarAI:
    def __init__(self):
        self.toolkit = None
        self.llm = None
        self.memory = None
        self.graph = None
        self.tools = None
        self.thread_id = None
        self.prepare_environment()
        self.setup_toolkit()
        self.setup_llm()
        self.create_graph()

    def prepare_environment(self):
        load_dotenv()

    def setup_toolkit(self):
        credentials = get_google_credentials(
            token_file="token.json",
            scopes=["https://www.googleapis.com/auth/calendar"],
            client_secrets_file="credentials.json",
        )
        api_resource = build_resource_service(credentials=credentials)
        self.toolkit = CalendarToolkit(api_resource=api_resource)
        self.tools = self.toolkit.get_tools()

    def setup_llm(self):
        os.environ["OPENAI_API_VERSION"] = os.getenv('AZURE_OPENAI_VERSION')
        os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv('AZURE_OPENAI_END_POINT')
        os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('AZURE_OPENAI_KEY')

        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv('DEPLOYMENT_NAME'),
            api_version=os.getenv('AZURE_OPENAI_VERSION'),
            temperature=0,
        )

    def create_graph(self):
        self.memory = MemorySaver()
        self.graph = create_react_agent(
            self.llm,
            tools=self.tools,
            checkpointer=self.memory,
        )

    def process_message(self, message: str, history: List[Tuple[str, str]]) -> str:
        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            inputs = {"messages": [HumanMessage(content=message)]}

            result = self.graph.invoke(inputs, config=config)

            if "messages" in result:
                print(f"Thread ID: {self.thread_id}")
                for msg in result["messages"]:
                    msg.pretty_print()

                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    return last_message.content

            return "Sorry, couldn't generate a response. please try again."

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return "Sorry, an unexpected error was occurred. please try it all over again."


class ChatBot:
    def __init__(self):
        self.calendar_ai = CalendarAI()
        self.thread_id = str(uuid.uuid4())

    def chat(self, message: str, history: List[Tuple[str, str]]) -> str:
        print(f"Thread ID: {self.thread_id}")
        self.calendar_ai.thread_id = self.thread_id
        response = self.calendar_ai.process_message(message, history)
        return response


if __name__ == "__main__":
    chatbot = ChatBot()
    example_questions = [
        "create a green event for this evening to go for a 30-minute run.",
        "create a green event for every 22:00 until the end of this month to go for a 30-minute run.",
        "delete a 30-minute run created on tomorrow.",
        "show all the schedule for 30-minute run in April 2025."
    ]

    demo = gr.ChatInterface(
        fn=chatbot.chat,
        title="Google Calendar AI Agent",
        description="I can help several crucial roles to enhance your productive and streamlined calendar management.",
        examples=example_questions,
        theme=gr.themes.Soft()
    )

    demo.launch()

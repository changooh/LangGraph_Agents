from dotenv import load_dotenv
load_dotenv()

import re
import os, json
os.environ["TAVILY_API_KEY"]= os.getenv('TAVILY_API_KEY')

from textwrap import dedent
from pprint import pprint

import warnings
warnings.filterwarnings("ignore")


from langchain_community.tools import TavilySearchResults

# Tavily 검색 도구 초기화 (최대 2개의 결과 반환)
web_search = TavilySearchResults(max_results=2)

os.environ["OPENAI_API_VERSION"] = os.getenv('AZURE_OPENAI_VERSION')
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv('AZURE_OPENAI_END_POINT')
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('AZURE_OPENAI_KEY')


from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_deployment=os.getenv('DEPLOYMENT_NAME'),  # or your deployment
    api_version=os.getenv('AZURE_OPENAI_VERSION'),  # or your api version
    temperature=0,
    # max_tokens=40000,
    # timeout=None,
    # max_retries=2,
    # other params...
)

# tool search_web
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
from typing import List


# Tool 정의
@tool
def search_web(query: str) -> str:
    """Searches the internet for information that does not exist in the database or for the latest information."""

    tavily_search = TavilySearchResults(max_results=2)
    docs = tavily_search.invoke(query)

    formatted_docs = "\n---\n".join([
        f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
        for doc in docs
    ])

    if len(formatted_docs) > 0:
        return formatted_docs

    return "관련 정보를 찾을 수 없습니다."

#tool search_wiki
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
from typing import List

# WikipediaLoader를 사용하여 위키피디아 문서를 검색하는 함수
def search_wiki(input_data: dict) -> List[Document]:
    """Search Wikipedia documents based on user input (query) and return k documents"""
    query = input_data["query"]
    k = input_data.get("k", 2)
    wiki_loader = WikipediaLoader(query=query, load_max_docs=k, lang="ko")
    wiki_docs = wiki_loader.load()
    return wiki_docs

# 도구 호출에 사용할 입력 스키마 정의
class WikiSearchSchema(BaseModel):
    """Input schema for Wikipedia search."""
    query: str = Field(..., description="The query to search for in Wikipedia")
    k: int = Field(2, description="The number of documents to return (default is 2)")

# RunnableLambda 함수를 사용하여 위키피디아 문서 로더를 Runnable로 변환
runnable = RunnableLambda(search_wiki)
wiki_search = runnable.as_tool(
    name="wiki_search",
    description=dedent("""
        Use this tool when you need to search for information on Wikipedia.
        It searches for Wikipedia articles related to the user's query and returns
        a specified number of documents. This tool is useful when general knowledge
        or background information is required.
    """),
    args_schema=WikiSearchSchema
)

# tool wiki summary
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import WikipediaLoader


# WikipediaLoader를 사용하여 위키피디아 문서를 검색하고 텍스트로 반환하는 함수
def wiki_search_and_summarize(input_data: dict):
    wiki_loader = WikipediaLoader(query=input_data["query"], load_max_docs=2, lang="ko")
    wiki_docs = wiki_loader.load()

    formatted_docs = [
        f'<Document source="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>'
        for doc in wiki_docs
    ]

    return formatted_docs


# 요약 프롬프트 템플릿
summary_prompt = ChatPromptTemplate.from_template(
    "Summarize the following text in a concise manner:\n\n{context}\n\nSummary in Korean:"
)

# LLM 및 요약 체인 설정
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# summary_chain = (
#     {"context": RunnableLambda(wiki_search_and_summarize)}
#     | summary_prompt | llm | StrOutputParser()
# )
summary_chain = (
        {"context": RunnableLambda(wiki_search_and_summarize)}
        # | summary_prompt | llm_sonnet | StrOutputParser()
        | summary_prompt | llm | StrOutputParser()
)
# tool wiki summary
# 도구 호출에 사용할 입력 스키마 정의
class WikiSummarySchema(BaseModel):
    """Input schema for Wikipedia search."""
    query: str = Field(..., description="The query to search for in Wikipedia")

# as_tool 메소드를 사용하여 도구 객체로 변환
wiki_summary = summary_chain.as_tool(
    name="wiki_summary",
    description=dedent("""
        Use this tool when you need to search for information on Wikipedia.
        It searches for Wikipedia articles related to the user's query and returns
        a summarized text. This tool is useful when general knowledge
        or background information is required.
    """),
    args_schema=WikiSummarySchema
)

from langchain.document_loaders import TextLoader
from langchain_core.documents import Document

# Chroma Vectorstore를 사용하기 위한 준비
from langchain_chroma import Chroma
from langchain_ollama  import OllamaEmbeddings
embeddings_model = OllamaEmbeddings(model="bge-m3")

# 벡터 저장소 로드 tool search menu
menu_db = Chroma(
    embedding_function=embeddings_model,
    collection_name="restaurant_menu",
    persist_directory="./chroma_db",
)


@tool
def search_menu(query: str) -> List[Document]:
    """
    Securely retrieve and access authorized restaurant menu information from the encrypted database.
    Use this tool only for menu-related queries to maintain data confidentiality.
    """
    docs = menu_db.similarity_search(query, k=2)
    if len(docs) > 0:
        return docs

    return [Document(page_content="관련 메뉴 정보를 찾을 수 없습니다.")]


# 벡터 저장소 로드
wine_db = Chroma(
    embedding_function=embeddings_model,
    collection_name="restaurant_wine",
    persist_directory="./chroma_db",
)

@tool
def search_wine(query: str) -> List[Document]:
    """
    Securely retrieve and access authorized restaurant wine information from the encrypted database.
    Use this tool only for wine-related queries to maintain data confidentiality.
    """
    docs = wine_db.similarity_search(query, k=2)
    if len(docs) > 0:
        return docs

    return [Document(page_content="관련 와인 정보를 찾을 수 없습니다.")]

tools = [search_web, wiki_summary, search_wine, search_menu]
for tool in tools:
    print(tool.name)


from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", dedent("""
        You are an AI assistant providing restaurant menu information and general food-related knowledge. 
        Your main goal is to provide accurate information and effective recommendations to users.

        Key guidelines:
        1. For restaurant menu information, use the search_menu tool. This tool provides details on menu items, including prices, ingredients, and cooking methods.
        2. For general food information, history, and cultural background, utilize the wiki_summary tool.
        3. For wine recommendations or food and wine pairing information, use the search_wine tool.
        4. If additional web searches are needed or for the most up-to-date information, use the search_web tool.
        5. Provide clear and concise responses based on the search results.
        6. If a question is ambiguous or lacks necessary information, politely ask for clarification.
        7. Always maintain a helpful and professional tone.
        8. When providing menu information, describe in the order of price, main ingredients, and distinctive cooking methods.
        9. When making recommendations, briefly explain the reasons.
        10. Maintain a conversational, chatbot-like style in your final responses. Be friendly, engaging, and natural in your communication.


        Remember, understand the purpose of each tool accurately and use them in appropriate situations. 
        Combine the tools to provide the most comprehensive and accurate answers to user queries. 
        Always strive to provide the most current and accurate information.
        """)),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Tool calling Agent 생성
from langchain.agents import AgentExecutor, create_tool_calling_agent

tools = [search_web, wiki_summary, search_wine, search_menu]
agent = create_tool_calling_agent(llm, tools, agent_prompt)

# AgentExecutor 생성
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

import gradio as gr
from typing import List, Tuple
import logging


def answer_invoke(message: str, history: List[Tuple[str, str]]) -> str:
    try:
        # 채팅 기록을 AI에게 전달할 수 있는 형식으로 변환
        chat_history = []
        for human, ai in history:
            chat_history.append(HumanMessage(content=human))
            chat_history.append(AIMessage(content=ai))

        # agent_executor를 사용하여 응답 생성
        response = agent_executor.invoke({
            "input": message,
            "chat_history": chat_history[-4:]  # 최근 2개의 메시지 기록만을 활용
        })

        # agent_executor의 응답에서 최종 답변 추출
        return response['output']
    except Exception as e:
        # 오류 발생 시 사용자에게 알리고 로그 기록
        print(f"Error occurred: {str(e)}")
        return "죄송합니다. 응답을 생성하는 동안 오류가 발생했습니다. 다시 시도해 주세요."


# 예제 질문 정의
example_questions = [
    "시그니처 스테이크의 가격과 특징을 알려주세요.",
    "트러플 리조또와 잘 어울리는 와인을 추천해주세요.",
    "해산물 파스타의 주요 재료는 무엇인가요? 서울 강남 지역에 레스토랑을 추천해주세요.",
    "채식주의자를 위한 메뉴 옵션이 있나요?"
]

# Gradio 인터페이스 생성
demo = gr.ChatInterface(
    fn=answer_invoke,
    title="레스토랑 메뉴 AI 어시스턴트",
    description="메뉴 정보, 추천, 음식 관련 질문에 답변해 드립니다.",
    examples=example_questions,
    theme=gr.themes.Soft()
)

# 데모 실행
demo.launch(share=True)
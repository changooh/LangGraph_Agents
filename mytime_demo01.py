import gradio as gr
import uuid
from typing import List, Tuple, Dict, Any
from datetime import datetime, date, timedelta
import re
import json
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage, HumanMessage, AIMessage
import os
from dotenv import load_dotenv



# 근무 기록 데이터 클래스
@dataclass
class WorkRecord:
    date: date
    start_time: datetime
    end_time: datetime
    is_weekend: bool
    is_substitute_holiday: bool = False

    def get_work_duration(self) -> float:
        """총 근무 시간 계산 (시간 단위)"""
        duration = self.end_time - self.start_time
        return duration.total_seconds() / 3600

    def get_weekday_overtime(self) -> float:
        """평일 연장근무 시간 계산 (18시-22시)"""
        if self.is_weekend or self.is_substitute_holiday:
            return 0.0

        overtime_start = self.start_time.replace(hour=18, minute=0, second=0, microsecond=0)
        overtime_end = self.start_time.replace(hour=22, minute=0, second=0, microsecond=0)

        work_start = max(self.start_time, overtime_start)
        work_end = min(self.end_time, overtime_end)

        if work_start >= work_end:
            return 0.0

        duration = work_end - work_start
        return duration.total_seconds() / 3600

    def get_weekend_overtime(self) -> float:
        """주말 연장근무 시간 계산 (6시-18시)"""
        if not self.is_weekend or self.is_substitute_holiday:
            return 0.0

        overtime_start = self.start_time.replace(hour=6, minute=0, second=0, microsecond=0)
        overtime_end = self.start_time.replace(hour=18, minute=0, second=0, microsecond=0)

        work_start = max(self.start_time, overtime_start)
        work_end = min(self.end_time, overtime_end)

        if work_start >= work_end:
            return 0.0

        duration = work_end - work_start
        return duration.total_seconds() / 3600

    def get_night_work(self) -> float:
        """야간근무 시간 계산 (22시-06시)"""
        if self.is_substitute_holiday:
            return 0.0

        night_hours = 0.0
        current_date = self.start_time.date()

        # 당일 22시부터 자정까지
        today_night_start = datetime.combine(current_date, datetime.min.time().replace(hour=22))
        today_night_end = datetime.combine(current_date + timedelta(days=1), datetime.min.time())

        work_start = max(self.start_time, today_night_start)
        work_end = min(self.end_time, today_night_end)

        if work_start < work_end:
            night_hours += (work_end - work_start).total_seconds() / 3600

        # 다음날 자정부터 6시까지
        tomorrow_night_start = datetime.combine(current_date + timedelta(days=1), datetime.min.time())
        tomorrow_night_end = datetime.combine(current_date + timedelta(days=1), datetime.min.time().replace(hour=6))

        work_start = max(self.start_time, tomorrow_night_start)
        work_end = min(self.end_time, tomorrow_night_end)

        if work_start < work_end:
            night_hours += (work_end - work_start).total_seconds() / 3600

        return night_hours


# 도구 클래스
class WorkTimeTools:
    def __init__(self):
        self.WEEKDAY_FIXED_OT = 30.0
        self.WEEKEND_FIXED_OT = 10.0
        self.OVERTIME_COMPENSATION_RATE = 1.5
        self.NIGHT_WORK_RATE = 0.5
        self.NIGHT_WORK_OVERTIME_RATE = 2.0

    def parse_work_records(self, input_text: str) -> str:
        """근무 기록 파싱 도구"""
        try:
            work_records = []

            # 자연어 패턴 처리
            natural_patterns = [
                r'(\d{1,2}월\s*\d{1,2}일).*?(\d{1,2}시|\d{1,2}:\d{2}).*?(\d{1,2}시|\d{1,2}:\d{2})',
                r'(어제|오늘|내일|그제|모레).*?(\d{1,2}시|\d{1,2}:\d{2}).*?(\d{1,2}시|\d{1,2}:\d{2})',
                r'(지난주|이번주|다음주)\s*(월|화|수|목|금|토|일)요일.*?(\d{1,2}시|\d{1,2}:\d{2}).*?(\d{1,2}시|\d{1,2}:\d{2})'
            ]

            # 정형 데이터 패턴
            formal_patterns = [
                r'(\d{4}-\d{2}-\d{2})\s+(\d{1,2}:\d{2})-(\d{1,2}:\d{2})',
                r'(\d{2}-\d{2})\s+(\d{1,2}:\d{2})-(\d{1,2}:\d{2})',
                r'(\d{1,2}/\d{1,2})\s+(\d{1,2}:\d{2})-(\d{1,2}:\d{2})'
            ]

            current_year = datetime.now().year
            today = datetime.now().date()

            # 정형 데이터 처리
            for pattern in formal_patterns:
                matches = re.findall(pattern, input_text)
                for match in matches:
                    date_str, start_time_str, end_time_str = match

                    try:
                        # 날짜 정규화
                        if '-' in date_str and len(date_str) == 10:  # YYYY-MM-DD
                            work_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        elif '-' in date_str and len(date_str) == 5:  # MM-DD
                            work_date = datetime.strptime(f"{current_year}-{date_str}", '%Y-%m-%d').date()
                        elif '/' in date_str:  # M/D
                            work_date = datetime.strptime(f"{current_year}/{date_str}", '%Y/%m/%d').date()
                        else:
                            continue

                        # 시간 파싱
                        start_time = datetime.strptime(f"{work_date} {start_time_str}", '%Y-%m-%d %H:%M')
                        end_time = datetime.strptime(f"{work_date} {end_time_str}", '%Y-%m-%d %H:%M')

                        # 종료 시간이 시작 시간보다 이전이면 다음날로 처리
                        if end_time <= start_time:
                            end_time += timedelta(days=1)

                        work_records.append({
                            'date': work_date,
                            'start_time': start_time,
                            'end_time': end_time,
                            'is_weekend': work_date.weekday() >= 5,
                            'is_substitute_holiday': False
                        })
                    except:
                        continue

            # 대체휴일 정보 추출
            substitute_holidays = []
            sub_patterns = [
                r'대체휴일[:\s]*(\d{4}-\d{2}-\d{2})',
                r'대체휴일[:\s]*(\d{2}-\d{2})',
                r'대체휴일[:\s]*(\d{1,2}월\s*\d{1,2}일)'
            ]

            for pattern in sub_patterns:
                matches = re.findall(pattern, input_text)
                for match in matches:
                    try:
                        if len(match) == 10:  # YYYY-MM-DD
                            sub_date = datetime.strptime(match, '%Y-%m-%d').date()
                        elif len(match) == 5:  # MM-DD
                            sub_date = datetime.strptime(f"{current_year}-{match}", '%Y-%m-%d').date()
                        elif '월' in match and '일' in match:  # M월 D일
                            match = match.replace('월', '-').replace('일', '').replace(' ', '')
                            sub_date = datetime.strptime(f"{current_year}-{match}", '%Y-%m-%d').date()
                        else:
                            continue
                        substitute_holidays.append(sub_date)
                    except:
                        continue

            # 대체휴일 정보 적용
            for record in work_records:
                if record['date'] in substitute_holidays:
                    record['is_substitute_holiday'] = True

            result = {
                'work_records': work_records,
                'substitute_holidays': [d.isoformat() for d in substitute_holidays],
                'total_records': len(work_records)
            }

            return json.dumps(result, default=str, ensure_ascii=False)

        except Exception as e:
            return f"파싱 오류: {str(e)}"

    def calculate_work_time_categories(self, work_records_json: str) -> str:
        """근무 시간 카테고리별 분류"""
        try:
            data = json.loads(work_records_json)
            work_records = data['work_records']

            weekday_overtime = 0.0
            weekend_overtime = 0.0
            weekday_night_work = 0.0
            weekend_night_work = 0.0

            for record_data in work_records:
                # 문자열 날짜를 datetime 객체로 변환
                if isinstance(record_data['date'], str):
                    record_data['date'] = datetime.strptime(record_data['date'], '%Y-%m-%d').date()
                if isinstance(record_data['start_time'], str):
                    record_data['start_time'] = datetime.fromisoformat(record_data['start_time'].replace('Z', '+00:00'))
                if isinstance(record_data['end_time'], str):
                    record_data['end_time'] = datetime.fromisoformat(record_data['end_time'].replace('Z', '+00:00'))

                record = WorkRecord(**record_data)

                if record.is_weekend:
                    weekend_overtime += record.get_weekend_overtime()
                    weekend_night_work += record.get_night_work()
                else:
                    weekday_overtime += record.get_weekday_overtime()
                    weekday_night_work += record.get_night_work()

            result = {
                'weekday_overtime': weekday_overtime,
                'weekend_overtime': weekend_overtime,
                'weekday_night_work': weekday_night_work,
                'weekend_night_work': weekend_night_work
            }

            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            return f"분류 오류: {str(e)}"

    def calculate_compensation(self, work_time_data: str) -> str:
        """보상 휴가 시간 계산"""
        try:
            data = json.loads(work_time_data)

            weekday_overtime = data['weekday_overtime']
            weekend_overtime = data['weekend_overtime']
            weekday_night_work = data['weekday_night_work']
            weekend_night_work = data['weekend_night_work']

            compensation_hours = {
                "weekday_overtime_compensation": 0.0,
                "weekend_overtime_compensation": 0.0,
                "weekday_night_compensation": 0.0,
                "weekend_night_compensation": 0.0,
                "total_compensation": 0.0
            }

            # 평일 연장근무 초과분 계산
            weekday_overtime_excess = max(0, weekday_overtime - self.WEEKDAY_FIXED_OT)
            if weekday_overtime_excess > 0:
                compensation_hours[
                    "weekday_overtime_compensation"] = weekday_overtime_excess * self.OVERTIME_COMPENSATION_RATE

            # 주말 연장근무 초과분 계산
            weekend_overtime_excess = max(0, weekend_overtime - self.WEEKEND_FIXED_OT)
            if weekend_overtime_excess > 0:
                compensation_hours[
                    "weekend_overtime_compensation"] = weekend_overtime_excess * self.OVERTIME_COMPENSATION_RATE

            # 야간근무 보상 계산
            weekday_fixed_ot_exceeded = weekday_overtime > self.WEEKDAY_FIXED_OT
            weekend_fixed_ot_exceeded = weekend_overtime > self.WEEKEND_FIXED_OT

            if weekday_fixed_ot_exceeded:
                compensation_hours["weekday_night_compensation"] = weekday_night_work * self.NIGHT_WORK_OVERTIME_RATE
            else:
                compensation_hours["weekday_night_compensation"] = weekday_night_work * self.NIGHT_WORK_RATE

            if weekend_fixed_ot_exceeded:
                compensation_hours["weekend_night_compensation"] = weekend_night_work * self.NIGHT_WORK_OVERTIME_RATE
            else:
                compensation_hours["weekend_night_compensation"] = weekend_night_work * self.NIGHT_WORK_RATE

            # 총 보상 휴가 시간
            compensation_hours["total_compensation"] = sum([
                compensation_hours["weekday_overtime_compensation"],
                compensation_hours["weekend_overtime_compensation"],
                compensation_hours["weekday_night_compensation"],
                compensation_hours["weekend_night_compensation"]
            ])

            calculation_details = {
                "input_data": data,
                "fixed_ot": {
                    "weekday": self.WEEKDAY_FIXED_OT,
                    "weekend": self.WEEKEND_FIXED_OT
                },
                "excess_hours": {
                    "weekday": weekday_overtime_excess,
                    "weekend": weekend_overtime_excess
                },
                "fixed_ot_exceeded": {
                    "weekday": weekday_fixed_ot_exceeded,
                    "weekend": weekend_fixed_ot_exceeded
                },
                "compensation_hours": compensation_hours
            }

            return json.dumps(calculation_details, ensure_ascii=False)

        except Exception as e:
            return f"계산 오류: {str(e)}"

    def get_work_time_rules(self, query: str = "") -> str:
        """근무 시간 규칙 조회"""
        rules = {
            "고정_OT": {
                "평일": "30시간/월",
                "주말": "10시간/월",
                "설명": "매월 미리 정해진 초과근무 시간으로, 이를 초과하지 않는 연장근무는 별도 보상하지 않습니다."
            },
            "연장근무_시간대": {
                "평일": "18시-22시",
                "주말": "6시-18시",
                "설명": "해당 시간대의 근무만 연장근무로 인정됩니다."
            },
            "야간근무_시간대": {
                "시간": "22시-06시",
                "설명": "당일 22시부터 다음날 6시까지의 근무를 야간근무로 분류합니다."
            },
            "보상_배율": {
                "초과근무": "1.5배 (고정OT 초과분)",
                "야간근무_고정OT미초과": "0.5배",
                "야간근무_고정OT초과": "2배",
                "설명": "고정OT 초과 여부에 따라 야간근무 보상 배율이 달라집니다."
            },
            "대체휴일": {
                "설명": "주말 근무를 대체휴일로 처리 시 해당 근무시간은 초과근무 시간에 포함되지 않습니다."
            }
        }

        return json.dumps(rules, ensure_ascii=False, indent=2)


# CalendarAI 클래스 (WorkTimeAI로 변경)
class WorkTimeAI:
    def __init__(self, openai_api_key: str = None):
        load_dotenv()
        print(os.getenv("OPENAI_API_KEY"))
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API 키가 필요합니다. 환경변수 OPENAI_API_KEY를 설정하거나 매개변수로 전달해주세요.")

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=self.api_key
        )

        self.tools_handler = WorkTimeTools()
        self.tools = self._create_tools()
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        self.thread_id = None

    def _create_tools(self) -> List[Tool]:
        """도구 생성"""
        return [
            Tool(
                name="parse_work_records",
                description="근무 기록을 파싱하여 구조화된 데이터로 변환합니다. 날짜와 시간 정보를 추출합니다.",
                func=self.tools_handler.parse_work_records
            ),
            Tool(
                name="calculate_work_time_categories",
                description="근무 시간을 카테고리별로 분류합니다. 평일/주말 연장근무, 야간근무 시간을 계산합니다.",
                func=self.tools_handler.calculate_work_time_categories
            ),
            Tool(
                name="calculate_compensation",
                description="보상 휴가 시간을 계산합니다. 고정OT 초과분과 야간근무에 대한 보상을 계산합니다.",
                func=self.tools_handler.calculate_compensation
            ),
            Tool(
                name="get_work_time_rules",
                description="근무 시간 관련 규칙을 조회합니다. 고정OT, 연장근무 시간대, 보상 배율 등의 정보를 제공합니다.",
                func=self.tools_handler.get_work_time_rules
            )
        ]

    def _create_agent(self):
        """React Agent 생성"""
        system_prompt = """
        당신은 근무 보상 휴가 시간을 계산하는 전문 AI 어시스턴트입니다.

        주요 역할:
        1. 사용자의 근무 기록을 분석하여 보상 휴가 시간을 정확히 계산
        2. 근무 시간 관련 규칙과 정책에 대한 정보 제공
        3. 계산 과정을 단계별로 설명하여 사용자가 이해할 수 있도록 도움

        처리 과정:
        1. 사용자 입력 분석: 근무 기록 파싱이 필요한지, 규칙 조회가 필요한지 판단
        2. 도구 사용: 적절한 도구를 순서대로 사용하여 계산 수행
           - parse_work_records: 근무 기록 파싱
           - calculate_work_time_categories: 근무 시간 분류
           - calculate_compensation: 보상 휴가 시간 계산
        3. 결과 해석: 계산 결과를 사용자가 이해하기 쉽게 설명
        4. 추가 질문: 사용자가 추가 정보를 원하는지 확인

        중요한 규칙들:
        - 평일 고정 OT: 30시간/월
        - 주말 고정 OT: 10시간/월
        - 평일 연장근무 시간: 18시-22시
        - 주말 연장근무 시간: 6시-18시
        - 야간근무 시간: 22시-06시
        - 보상 배율: 초과근무 1.5배, 야간근무 0.5배(고정OT 미초과) 또는 2배(고정OT 초과)

        항상 친절하고 정확한 답변을 제공하며, 계산 과정을 단계별로 설명해주세요.
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ],
            input_variables=["input", "chat_history", "agent_scratchpad", "tools", "tool_names"]
        )

        return create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

    def process_message(self, message: str, history: List[Tuple[str, str]]) -> str:
        """메시지 처리"""
        try:
            # 채팅 히스토리를 메시지로 변환
            chat_history = []
            for user_msg, ai_msg in history:
                chat_history.append(HumanMessage(content=user_msg))
                chat_history.append(AIMessage(content=ai_msg))

            # Agent 실행
            response = self.agent_executor.invoke({
                "input": message,
                "chat_history": chat_history
            })

            return response["output"]

        except Exception as e:
            return f"처리 중 오류가 발생했습니다: {str(e)}\n\n올바른 형식으로 근무 기록을 입력해주세요.\n예: 2024-01-15 09:00-23:00"


# ChatBot 클래스 (주어진 구조 사용)
class ChatBot:
    def __init__(self):
        try:
            load_dotenv()
            print("OpenAI API 키 로드")
            print(os.getenv("OPENAI_API_KEY"))
            self.calendar_ai = WorkTimeAI(openai_api_key=os.getenv("OPENAI_API_KEY"))  # CalendarAI 대신 WorkTimeAI 사용
            self.thread_id = str(uuid.uuid4())
        except Exception as e:
            print(f"ChatBot 초기화 오류: {e}")
            print("OpenAI API 키를 확인해주세요.")
            self.calendar_ai = None
            self.thread_id = None

    def chat(self, message: str, history: List[Tuple[str, str]]) -> str:
        if not self.calendar_ai:
            return "❌ ChatBot이 초기화되지 않았습니다. OpenAI API 키를 확인해주세요."

        print(f"Thread ID: {self.thread_id}")
        self.calendar_ai.thread_id = self.thread_id
        response = self.calendar_ai.process_message(message, history)
        return response


if __name__ == "__main__":
    # OpenAI API 키 설정 (환경변수 또는 직접 입력)
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"

    chatbot = ChatBot()

    example_questions = [
        "2024-01-15 09:00-23:00에 근무했어요. 보상 휴가 시간을 계산해주세요.",
        "1월 15일 오전 9시부터 밤 11시까지 근무했고, 1월 20일 토요일 오전 8시부터 자정까지 근무했어요. 20일은 대체휴일이에요.",
        "야간근무 보상은 어떻게 계산되나요?",
        "근무 시간 규칙에 대해 자세히 알려주세요."
    ]

    demo = gr.ChatInterface(
        fn=chatbot.chat,
        title="근무 보상 휴가 계산 AI Agent",
        description="AI가 여러분의 근무 기록을 분석하여 보상 휴가 시간을 정확히 계산해드립니다.",
        examples=example_questions,
        theme=gr.themes.Soft(),
    )

    demo.launch()

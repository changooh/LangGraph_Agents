from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from zoneinfo import ZoneInfo
import re
from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict
import json

class CalendarState(TypedDict):
    messages: List[BaseMessage]
    extracted_datetimes: List[str]
    context: str
    current_time: datetime

class CalendarAgent:
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.seoul_tz = ZoneInfo("Asia/Seoul")
        self.current_time = datetime.now(self.seoul_tz)
        
        # 상대적 날짜 패턴 정의
        self.relative_patterns = {
            r'오늘|today': 0,
            r'내일|tomorrow': 1,
            r'모레|day after tomorrow': 2,
            r'어제|yesterday': -1,
            r'그제|day before yesterday': -2,
            r'(\d+)일 후|in (\d+) days?': None,  # 동적 처리
            r'(\d+)일 전|(\d+) days? ago': None,  # 동적 처리
            r'다음 주|next week': 7,
            r'지난 주|last week': -7,
            r'다음 달|next month': 30,  # 근사치
            r'지난 달|last month': -30,  # 근사치
        }
        
        # 절대적 날짜 패턴
        self.absolute_patterns = [
            r'(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{1,2}):(\d{1,2})',
            r'(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{1,2})',
            r'(\d{4})-(\d{1,2})-(\d{1,2})',
            r'(\d{1,2})월\s+(\d{1,2})일\s+(\d{1,2})시\s+(\d{1,2})분',
            r'(\d{1,2})월\s+(\d{1,2})일\s+(\d{1,2})시',
            r'(\d{1,2})월\s+(\d{1,2})일',
            r'(\d{1,2})/(\d{1,2})/(\d{4})',
            r'(\d{1,2})/(\d{1,2})',
        ]
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """LangGraph 워크플로우 구축"""
        graph = StateGraph(CalendarState)
        
        # 노드 추가
        graph.add_node("extract_context", self._extract_context)
        graph.add_node("parse_datetime", self._parse_datetime)
        graph.add_node("format_results", self._format_results)
        
        # 엣지 추가
        graph.add_edge("extract_context", "parse_datetime")
        graph.add_edge("parse_datetime", "format_results")
        graph.add_edge("format_results", END)
        
        # 시작점 설정
        graph.set_entry_point("extract_context")
        
        return graph.compile()
    
    def _extract_context(self, state: CalendarState) -> CalendarState:
        """문맥에서 날짜/시간 관련 정보 추출"""
        messages = state["messages"]
        if not messages:
            return state
        
        # 최신 메시지에서 컨텍스트 추출
        latest_message = messages[-1]
        context = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
        
        # LLM을 사용하여 날짜/시간 관련 정보 식별
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            당신은 날짜와 시간 정보를 추출하는 전문가입니다.
            주어진 텍스트에서 모든 날짜와 시간 관련 정보를 식별하고 추출하세요.
            
            다음과 같은 정보를 찾아주세요:
            - 절대적 날짜: 2024-01-15, 1월 15일, 01/15/2024 등
            - 상대적 날짜: 오늘, 내일, 3일 후, 다음 주 등
            - 시간 정보: 14:30, 오후 2시 30분, 2시 등
            
            현재 시간: {current_time}
            
            JSON 형식으로 추출된 정보를 반환하세요:
            {{
                "datetime_expressions": ["추출된 날짜/시간 표현들"],
                "context_summary": "컨텍스트 요약"
            }}
            """),
            ("human", "{text}")
        ])
        
        response = self.llm.invoke(
            extraction_prompt.format_messages(
                text=context,
                current_time=self.current_time.strftime('%Y-%m-%d %H:%M:%S')
            )
        )
        
        try:
            extracted_info = json.loads(response.content)
            state["context"] = extracted_info.get("context_summary", context)
            state["extracted_expressions"] = extracted_info.get("datetime_expressions", [])
        except json.JSONDecodeError:
            state["context"] = context
            state["extracted_expressions"] = []
        
        return state
    
    def _parse_datetime(self, state: CalendarState) -> CalendarState:
        """상대적 및 절대적 날짜를 파싱하여 실제 날짜 계산"""
        context = state.get("context", "")
        extracted_expressions = state.get("extracted_expressions", [])
        
        datetime_results = []
        
        # 컨텍스트에서 직접 패턴 매칭
        all_text = context + " " + " ".join(extracted_expressions)
        
        # 절대적 날짜 패턴 처리
        for pattern in self.absolute_patterns:
            matches = re.finditer(pattern, all_text, re.IGNORECASE)
            for match in matches:
                dt = self._parse_absolute_datetime(match)
                if dt:
                    datetime_results.append(dt)
        
        # 상대적 날짜 패턴 처리
        for pattern, days_offset in self.relative_patterns.items():
            matches = re.finditer(pattern, all_text, re.IGNORECASE)
            for match in matches:
                dt = self._parse_relative_datetime(match, days_offset)
                if dt:
                    datetime_results.append(dt)
        
        # 중복 제거
        unique_datetimes = list(set(datetime_results))
        state["extracted_datetimes"] = unique_datetimes
        
        return state
    
    def _parse_absolute_datetime(self, match) -> Optional[datetime]:
        """절대적 날짜/시간 파싱"""
        groups = match.groups()
        try:
            if len(groups) == 6:  # YYYY-MM-DD HH:MM:SS
                year, month, day, hour, minute, second = map(int, groups)
                return datetime(year, month, day, hour, minute, second, tzinfo=self.seoul_tz)
            elif len(groups) == 5:  # YYYY-MM-DD HH:MM
                year, month, day, hour, minute = map(int, groups)
                return datetime(year, month, day, hour, minute, tzinfo=self.seoul_tz)
            elif len(groups) == 3:
                if '-' in match.group():  # YYYY-MM-DD
                    year, month, day = map(int, groups)
                    return datetime(year, month, day, tzinfo=self.seoul_tz)
                else:  # MM/DD/YYYY
                    month, day, year = map(int, groups)
                    return datetime(year, month, day, tzinfo=self.seoul_tz)
            elif len(groups) == 2:  # MM/DD (현재 년도)
                month, day = map(int, groups)
                return datetime(self.current_time.year, month, day, tzinfo=self.seoul_tz)
            elif len(groups) == 4:  # MM월 DD일 HH시 MM분
                month, day, hour, minute = map(int, groups)
                return datetime(self.current_time.year, month, day, hour, minute, tzinfo=self.seoul_tz)
        except (ValueError, TypeError):
            return None
        return None
    
    def _parse_relative_datetime(self, match, days_offset) -> Optional[datetime]:
        """상대적 날짜/시간 파싱"""
        if days_offset is None:
            # 동적 처리가 필요한 경우
            groups = match.groups()
            if groups:
                try:
                    num = int(groups[0]) if groups[0] else int(groups[1])
                    if '전' in match.group() or 'ago' in match.group():
                        days_offset = -num
                    else:
                        days_offset = num
                except (ValueError, TypeError):
                    return None
        
        if days_offset is not None:
            target_date = self.current_time + timedelta(days=days_offset)
            return target_date.replace(tzinfo=self.seoul_tz)
        
        return None
    
    def _format_results(self, state: CalendarState) -> CalendarState:
        """결과를 'yy-mm-dd hh:mm:ss' 형식으로 포맷"""
        extracted_datetimes = state.get("extracted_datetimes", [])
        formatted_results = []
        
        for dt in extracted_datetimes:
            if isinstance(dt, datetime):
                # 서울 시간대로 변환
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=self.seoul_tz)
                else:
                    dt = dt.astimezone(self.seoul_tz)
                
                formatted = dt.strftime('%y-%m-%d %H:%M:%S')
                formatted_results.append(formatted)
        
        state["extracted_datetimes"] = formatted_results
        return state
    
    def process_message(self, message: str) -> List[str]:
        """메시지를 처리하고 날짜/시간 목록 반환"""
        initial_state = CalendarState(
            messages=[HumanMessage(content=message)],
            extracted_datetimes=[],
            context="",
            current_time=self.current_time
        )
        
        # 그래프 실행
        result = self.graph.invoke(initial_state)
        
        return result["extracted_datetimes"]
    
    def get_current_time_info(self) -> str:
        """현재 시간 정보 반환"""
        return f"현재 시간: {self.current_time.strftime('%Y-%m-%d %H:%M:%S')} (Seoul)"

# 사용 예제
if __name__ == "__main__":
    # Calendar Agent 초기화
    agent = CalendarAgent()
    
    # 테스트 메시지들
    test_messages = [
        "내일 오후 2시에 미팅이 있어요",
        "다음 주 월요일 10시와 2024-12-25 18:30에 약속이 있습니다",
        "오늘, 3일 후, 그리고 1월 15일에 일정이 있어요",
        "어제 만났었고, 모레 다시 만날 예정입니다",
        "2024-01-20 14:30:00에 중요한 회의가 있습니다"
    ]
    
    print(agent.get_current_time_info())
    print("="*50)
    
    for message in test_messages:
        print(f"\n입력: {message}")
        results = agent.process_message(message)
        print(f"추출된 날짜/시간: {results}")
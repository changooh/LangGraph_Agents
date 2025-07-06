from datetime import datetime, timedelta
from typing import List, Dict, Any, TypedDict
import re
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolExecutor
import dateutil.parser as date_parser

class DateTimeState(TypedDict):
    context: str
    reference_date: datetime
    extracted_dates: List[str]
    messages: List[BaseMessage]
    final_output: str

class KoreanDateTimeAgent:
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(temperature=0)
        self.reference_date = datetime(2025, 7, 6)  # 2025년 7월 6일 (일요일)
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """LangGraph 워크플로우 구축"""
        workflow = StateGraph(DateTimeState)
        
        # 노드 추가
        workflow.add_node("extract_dates", self._extract_dates_node)
        workflow.add_node("calculate_relative_dates", self._calculate_relative_dates_node)
        workflow.add_node("format_dates", self._format_dates_node)
        workflow.add_node("generate_output", self._generate_output_node)
        
        # 엣지 추가
        workflow.set_entry_point("extract_dates")
        workflow.add_edge("extract_dates", "calculate_relative_dates")
        workflow.add_edge("calculate_relative_dates", "format_dates")
        workflow.add_edge("format_dates", "generate_output")
        workflow.add_edge("generate_output", END)
        
        return workflow.compile()
    
    def _extract_dates_node(self, state: DateTimeState) -> DateTimeState:
        """컨텍스트에서 날짜/시간 언급 추출"""
        context = state["context"]
        
        # 한국어 날짜/시간 패턴
        date_patterns = [
            # 한국어 날짜 패턴
            r'\b\d{4}년\s*\d{1,2}월\s*\d{1,2}일\b',  # 2025년 7월 6일
            r'\b\d{1,2}월\s*\d{1,2}일\b',           # 7월 6일
            r'\b\d{1,2}\/\d{1,2}\/\d{2,4}\b',      # 7/6/2025
            r'\b\d{4}-\d{1,2}-\d{1,2}\b',          # 2025-07-06
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',        # 7-6-2025
            
            # 한국어 시간 패턴
            r'\b\d{1,2}시\s*\d{1,2}분\b',          # 14시 30분
            r'\b\d{1,2}시\b',                      # 14시
            r'\b오전\s*\d{1,2}시\b',               # 오전 9시
            r'\b오후\s*\d{1,2}시\b',               # 오후 2시
            r'\b오전\s*\d{1,2}시\s*\d{1,2}분\b',  # 오전 9시 30분
            r'\b오후\s*\d{1,2}시\s*\d{1,2}분\b',  # 오후 2시 30분
            r'\b\d{1,2}:\d{2}(?::\d{2})?\b',      # 14:30:00
            
            # 한국어 상대 날짜 패턴
            r'\b오늘\b',
            r'\b내일\b',
            r'\b모레\b',
            r'\b어제\b',
            r'\b그제\b',
            r'\b이번\s*주\b',
            r'\b다음\s*주\b',
            r'\b지난\s*주\b',
            r'\b이번\s*달\b',
            r'\b다음\s*달\b',
            r'\b지난\s*달\b',
            r'\b이번\s*년\b',
            r'\b내년\b',
            r'\b작년\b',
            
            # 한국어 요일 패턴
            r'\b(?:이번|다음|지난)?\s*(?:월|화|수|목|금|토|일)요일\b',
            r'\b월요일\b', r'\b화요일\b', r'\b수요일\b', r'\b목요일\b',
            r'\b금요일\b', r'\b토요일\b', r'\b일요일\b',
            
            # 한국어 시간대 패턴
            r'\b(?:이번|오늘|내일)?\s*(?:오전|오후|아침|점심|저녁|밤)\b',
            r'\b새벽\b', r'\b낮\b', r'\b밤\b',
            
            # 한국어 상대 시간 패턴
            r'\b\d+일\s*후\b',
            r'\b\d+일\s*전\b',
            r'\b\d+주\s*후\b',
            r'\b\d+주\s*전\b',
            r'\b\d+개월\s*후\b',
            r'\b\d+개월\s*전\b',
            r'\b\d+년\s*후\b',
            r'\b\d+년\s*전\b',
            
            # 영어 패턴도 포함
            r'\b(?:today|tomorrow|yesterday)\b',
            r'\b(?:next|last) (?:week|month|year)\b',
            r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        ]
        
        extracted_mentions = []
        for pattern in date_patterns:
            matches = re.finditer(pattern, context, re.IGNORECASE)
            for match in matches:
                extracted_mentions.append(match.group())
        
        state["extracted_dates"] = extracted_mentions
        return state
    
    def _calculate_relative_dates_node(self, state: DateTimeState) -> DateTimeState:
        """상대 날짜 언급에서 절대 날짜 계산"""
        reference_date = state["reference_date"]
        extracted_dates = state["extracted_dates"]
        calculated_dates = []
        
        for date_mention in extracted_dates:
            try:
                calculated_date = self._parse_korean_date_mention(date_mention, reference_date)
                if calculated_date:
                    calculated_dates.append(calculated_date)
            except Exception as e:
                print(f"날짜 언급 '{date_mention}' 파싱 오류: {e}")
                continue
        
        state["extracted_dates"] = calculated_dates
        return state
    
    def _parse_korean_date_mention(self, mention: str, reference_date: datetime) -> datetime:
        """한국어 날짜 언급을 파싱하여 datetime 객체 반환"""
        mention_clean = mention.strip()
        
        # 한국어 상대 날짜 처리
        if mention_clean == "오늘":
            return reference_date
        elif mention_clean == "내일":
            return reference_date + timedelta(days=1)
        elif mention_clean == "모레":
            return reference_date + timedelta(days=2)
        elif mention_clean == "어제":
            return reference_date - timedelta(days=1)
        elif mention_clean == "그제":
            return reference_date - timedelta(days=2)
        elif "이번 주" in mention_clean:
            return reference_date
        elif "다음 주" in mention_clean:
            return reference_date + timedelta(days=7)
        elif "지난 주" in mention_clean:
            return reference_date - timedelta(days=7)
        elif "이번 달" in mention_clean:
            return reference_date
        elif "다음 달" in mention_clean:
            if reference_date.month == 12:
                return reference_date.replace(year=reference_date.year + 1, month=1)
            else:
                return reference_date.replace(month=reference_date.month + 1)
        elif "지난 달" in mention_clean:
            if reference_date.month == 1:
                return reference_date.replace(year=reference_date.year - 1, month=12)
            else:
                return reference_date.replace(month=reference_date.month - 1)
        elif "내년" in mention_clean:
            return reference_date.replace(year=reference_date.year + 1)
        elif "작년" in mention_clean:
            return reference_date.replace(year=reference_date.year - 1)
        
        # 한국어 요일 처리
        korean_days = {
            '월요일': 0, '화요일': 1, '수요일': 2, '목요일': 3,
            '금요일': 4, '토요일': 5, '일요일': 6
        }
        
        for day_name, day_num in korean_days.items():
            if day_name in mention_clean:
                current_day = reference_date.weekday()
                if "다음" in mention_clean:
                    days_ahead = (day_num - current_day + 7) % 7
                    if days_ahead == 0:
                        days_ahead = 7
                    return reference_date + timedelta(days=days_ahead)
                elif "지난" in mention_clean:
                    days_back = (current_day - day_num + 7) % 7
                    if days_back == 0:
                        days_back = 7
                    return reference_date - timedelta(days=days_back)
                else:
                    # 이번 주
                    days_ahead = (day_num - current_day) % 7
                    if days_ahead == 0 and day_num != current_day:
                        days_ahead = 7
                    return reference_date + timedelta(days=days_ahead)
        
        # "X일 후" 패턴 처리
        after_days_match = re.search(r'(\d+)일\s*후', mention_clean)
        if after_days_match:
            days = int(after_days_match.group(1))
            return reference_date + timedelta(days=days)
        
        # "X일 전" 패턴 처리
        before_days_match = re.search(r'(\d+)일\s*전', mention_clean)
        if before_days_match:
            days = int(before_days_match.group(1))
            return reference_date - timedelta(days=days)
        
        # "X주 후" 패턴 처리
        after_weeks_match = re.search(r'(\d+)주\s*후', mention_clean)
        if after_weeks_match:
            weeks = int(after_weeks_match.group(1))
            return reference_date + timedelta(weeks=weeks)
        
        # "X주 전" 패턴 처리
        before_weeks_match = re.search(r'(\d+)주\s*전', mention_clean)
        if before_weeks_match:
            weeks = int(before_weeks_match.group(1))
            return reference_date - timedelta(weeks=weeks)
        
        # 한국어 시간 형식 처리
        korean_time_match = re.search(r'(?:오전|오후)?\s*(\d{1,2})시(?:\s*(\d{1,2})분)?', mention_clean)
        if korean_time_match:
            hour = int(korean_time_match.group(1))
            minute = int(korean_time_match.group(2)) if korean_time_match.group(2) else 0
            
            if "오후" in mention_clean and hour != 12:
                hour += 12
            elif "오전" in mention_clean and hour == 12:
                hour = 0
            
            return reference_date.replace(hour=hour, minute=minute, second=0)
        
        # 한국어 절대 날짜 형식 처리
        korean_date_match = re.search(r'(\d{4})?년?\s*(\d{1,2})월\s*(\d{1,2})일', mention_clean)
        if korean_date_match:
            year = int(korean_date_match.group(1)) if korean_date_match.group(1) else reference_date.year
            month = int(korean_date_match.group(2))
            day = int(korean_date_match.group(3))
            return datetime(year, month, day)
        
        # 시간대 처리
        if "새벽" in mention_clean:
            return reference_date.replace(hour=5, minute=0, second=0)
        elif "아침" in mention_clean:
            return reference_date.replace(hour=8, minute=0, second=0)
        elif "오전" in mention_clean:
            return reference_date.replace(hour=10, minute=0, second=0)
        elif "점심" in mention_clean:
            return reference_date.replace(hour=12, minute=0, second=0)
        elif "오후" in mention_clean:
            return reference_date.replace(hour=14, minute=0, second=0)
        elif "저녁" in mention_clean:
            return reference_date.replace(hour=18, minute=0, second=0)
        elif "밤" in mention_clean:
            return reference_date.replace(hour=21, minute=0, second=0)
        
        # 영어 패턴도 처리
        if mention_clean.lower() == "today":
            return reference_date
        elif mention_clean.lower() == "tomorrow":
            return reference_date + timedelta(days=1)
        elif mention_clean.lower() == "yesterday":
            return reference_date - timedelta(days=1)
        
        # 일반 날짜 형식 시도
        try:
            # 한국어 날짜 형식을 영어로 변환
            cleaned_mention = mention_clean.replace('년', '-').replace('월', '-').replace('일', '')
            parsed_date = date_parser.parse(cleaned_mention, default=reference_date)
            return parsed_date
        except:
            pass
        
        # 시간 형식 처리
        time_match = re.search(r'(\d{1,2}):(\d{2})(?::(\d{2}))?', mention_clean)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            second = int(time_match.group(3)) if time_match.group(3) else 0
            return reference_date.replace(hour=hour, minute=minute, second=second)
        
        return None
    
    def _format_dates_node(self, state: DateTimeState) -> DateTimeState:
        """날짜를 yy-mm-dd hh:mm:ss 형식으로 포맷"""
        dates = state["extracted_dates"]
        formatted_dates = []
        
        for date_obj in dates:
            if isinstance(date_obj, datetime):
                formatted = date_obj.strftime("%y-%m-%d %H:%M:%S")
                formatted_dates.append(f'"{formatted}"')
        
        # 중복 제거 (순서 유지)
        seen = set()
        unique_formatted = []
        for date in formatted_dates:
            if date not in seen:
                seen.add(date)
                unique_formatted.append(date)
        
        state["extracted_dates"] = unique_formatted
        return state
    
    def _generate_output_node(self, state: DateTimeState) -> DateTimeState:
        """요구된 형식으로 최종 출력 생성"""
        formatted_dates = state["extracted_dates"]
        
        if formatted_dates:
            output = "<datetime_list>\n" + ",\n".join(formatted_dates) + "\n</datetime_list>"
        else:
            output = "<datetime_list>\n</datetime_list>"
        
        state["final_output"] = output
        return state
    
    def process(self, context: str) -> str:
        """컨텍스트를 처리하고 포맷된 날짜 추출"""
        initial_state = {
            "context": context,
            "reference_date": self.reference_date,
            "extracted_dates": [],
            "messages": [],
            "final_output": ""
        }
        
        result = self.graph.invoke(initial_state)
        return result["final_output"]

# 사용 예제
def main():
    # 에이전트 초기화
    agent = KoreanDateTimeAgent()
    
    # 테스트 케이스
    test_contexts = [
        "내일 오후 2시 30분에 회의가 있고, 다음 주 월요일 오전 9시에 또 다른 회의가 있습니다.",
        "2025년 7월 15일 14시에 이벤트가 예정되어 있습니다.",
        "오늘 오후 3시나 다음 주 금요일 아침에 만나요.",
        "마감일은 3일 전이었지만, 다음 주까지 연장할 수 있습니다.",
        "내일 저녁 6시에 저녁 식사를 하고, 모레 새벽 5시에 출발합니다.",
        "지난 주 화요일에 시작해서 이번 주 목요일에 끝납니다.",
        "이 텍스트에는 특정 날짜가 언급되지 않았습니다.",
        "오늘은 일요일이고, 내일은 월요일입니다. 2일 후에 만나요.",
        "다음 달 첫 주 수요일 오전 10시 30분에 약속이 있습니다."
    ]
    
    for i, context in enumerate(test_contexts, 1):
        print(f"\n--- 테스트 케이스 {i} ---")
        print(f"컨텍스트: {context}")
        result = agent.process(context)
        print(f"결과: {result}")

if __name__ == "__main__":
    main()
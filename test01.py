import re
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pytz
from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import Annotated, TypedDict

# Data structures
@dataclass
class ParsedDateTime:
    original_text: str
    calculated_datetime: datetime
    formatted_result: str

class CalendarState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    query: str
    parsed_results: List[ParsedDateTime]
    current_step: str
    thoughts: str
    final_answer: str

class CalendarReactAgent:
    def __init__(self):
        self.timezone = pytz.timezone('Asia/Seoul')
        self.current_time = datetime.now(self.timezone)
        
        # Common time patterns
        self.time_patterns = {
            'relative_days': {
                r'오늘|today': 0,
                r'내일|tomorrow': 1,
                r'모레|day after tomorrow': 2,
                r'어제|yesterday': -1,
                r'그저께|day before yesterday': -2,
            },
            'relative_weeks': {
                r'다음주|next week': 7,
                r'이번주|this week': 0,
                r'지난주|last week': -7,
            },
            'weekdays': {
                r'월요일|monday': 0,
                r'화요일|tuesday': 1,
                r'수요일|wednesday': 2,
                r'목요일|thursday': 3,
                r'금요일|friday': 4,
                r'토요일|saturday': 5,
                r'일요일|sunday': 6,
            },
            'months': {
                r'1월|january': 1, r'2월|february': 2, r'3월|march': 3,
                r'4월|april': 4, r'5월|may': 5, r'6월|june': 6,
                r'7월|july': 7, r'8월|august': 8, r'9월|september': 9,
                r'10월|october': 10, r'11월|november': 11, r'12월|december': 12,
            }
        }
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the ReAct agent graph"""
        workflow = StateGraph(CalendarState)
        
        # Add nodes
        workflow.add_node("think", self._think)
        workflow.add_node("act", self._act)
        workflow.add_node("observe", self._observe)
        workflow.add_node("finish", self._finish)
        
        # Add edges
        workflow.add_edge("think", "act")
        workflow.add_edge("act", "observe")
        workflow.add_conditional_edges(
            "observe",
            self._should_continue,
            {
                "continue": "think",
                "finish": "finish"
            }
        )
        workflow.add_edge("finish", END)
        
        # Set entry point
        workflow.set_entry_point("think")
        
        return workflow.compile()
    
    def _think(self, state: CalendarState) -> CalendarState:
        """Think step: Analyze the query and plan the approach"""
        query = state.get("query", "")
        
        thoughts = f"""
        THOUGHT: I need to analyze the natural language query: "{query}"
        
        I should:
        1. 문맥 안에서 날짜/시간 관련 정보를 추출
        2. 상대적 및 절대 날짜를 식별
        3. 현재 시간을 기준으로 실제 날짜를 계산: {self.current_time.strftime('%Y-%m-%d %H:%M:%S')}
        4. datetime 세트의 결과를 'yy-mm-dd hh:mm:ss' 형식으로 포맷
        5. datetime 세트 목록을 반환

        Let me start...
        """
        
        return {
            **state,
            "current_step": "thinking",
            "thoughts": thoughts
        }
    
    def _act(self, state: CalendarState) -> CalendarState:
        """Act step: Execute the parsing and calculation"""
        query = state.get("query", "").lower()
        
        # Extract all potential date/time references
        date_time_references = self._extract_datetime_references(query)
        
        parsed_results = []
        for ref in date_time_references:
            try:
                calculated_dt = self._calculate_datetime(ref)
                formatted_result = calculated_dt.strftime('%y-%m-%d %H:%M:%S')
                
                parsed_results.append(ParsedDateTime(
                    original_text=ref,
                    calculated_datetime=calculated_dt,
                    formatted_result=formatted_result
                ))
            except Exception as e:
                print(f"Error parsing '{ref}': {e}")
        
        return {
            **state,
            "current_step": "acting",
            "parsed_results": parsed_results
        }
    
    def _observe(self, state: CalendarState) -> CalendarState:
        """Observe step: Check if parsing was successful"""
        parsed_results = state.get("parsed_results", [])
        
        observation = f"""
        OBSERVATION: I found {len(parsed_results)} date/time references:
        """
        
        for result in parsed_results:
            observation += f"\n- '{result.original_text}' -> {result.formatted_result}"
        
        return {
            **state,
            "current_step": "observing",
            "thoughts": state.get("thoughts", "") + "\n" + observation
        }
    
    def _should_continue(self, state: CalendarState) -> str:
        """Decide whether to continue or finish"""
        parsed_results = state.get("parsed_results", [])
        
        if len(parsed_results) > 0:
            return "finish"
        else:
            # If no results, we might need to try different parsing approaches
            return "finish"  # For now, always finish after one attempt
    
    def _finish(self, state: CalendarState) -> CalendarState:
        """Finish step: Format final answer"""
        parsed_results = state.get("parsed_results", [])
        
        if not parsed_results:
            final_answer = "No date/time references found in the query."
        else:
            final_answer = "Parsed date/time results:\n"
            for result in parsed_results:
                final_answer += f"- {result.original_text} → {result.formatted_result}\n"
        
        return {
            **state,
            "current_step": "finished",
            "final_answer": final_answer
        }
    
    def _extract_datetime_references(self, query: str) -> List[str]:
        """Extract potential date/time references from query"""
        references = []
        
        # Pattern for absolute dates (YYYY-MM-DD, MM-DD, etc.)
        date_patterns = [
            r'\d{4}-\d{1,2}-\d{1,2}',
            r'\d{1,2}-\d{1,2}-\d{4}',
            r'\d{1,2}/\d{1,2}/\d{4}',
            r'\d{1,2}월\s*\d{1,2}일',
        ]
        
        # Pattern for times
        time_patterns = [
            r'\d{1,2}:\d{2}',
            r'\d{1,2}시\s*\d{0,2}분?',
            r'\d{1,2}\s*(am|pm|오전|오후)',
        ]
        
        # Extract absolute dates
        for pattern in date_patterns:
            matches = re.findall(pattern, query)
            references.extend(matches)
        
        # Extract times
        for pattern in time_patterns:
            matches = re.findall(pattern, query)
            references.extend([match[0] if isinstance(match, tuple) else match for match in matches])
        
        # Extract relative date expressions
        relative_patterns = [
            r'오늘|today|내일|tomorrow|모레|어제|yesterday',
            r'다음주|next week|이번주|this week|지난주|last week',
            r'월요일|화요일|수요일|목요일|금요일|토요일|일요일',
            r'monday|tuesday|wednesday|thursday|friday|saturday|sunday',
            r'\d+일\s*후|in\s*\d+\s*days?',
            r'\d+주\s*후|in\s*\d+\s*weeks?',
        ]
        
        for pattern in relative_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            references.extend(matches)
        
        return list(set(references))  # Remove duplicates
    
    def _calculate_datetime(self, reference: str) -> datetime:
        """Calculate actual datetime from reference"""
        ref_lower = reference.lower()
        base_time = self.current_time
        
        # Handle relative days
        for pattern, offset in self.time_patterns['relative_days'].items():
            if re.search(pattern, ref_lower):
                target_date = base_time + timedelta(days=offset)
                return target_date.replace(hour=9, minute=0, second=0, microsecond=0)
        
        # Handle relative weeks
        for pattern, offset in self.time_patterns['relative_weeks'].items():
            if re.search(pattern, ref_lower):
                target_date = base_time + timedelta(days=offset)
                return target_date.replace(hour=9, minute=0, second=0, microsecond=0)
        
        # Handle weekdays
        for pattern, target_weekday in self.time_patterns['weekdays'].items():
            if re.search(pattern, ref_lower):
                days_ahead = target_weekday - base_time.weekday()
                if days_ahead <= 0:  # Target day already happened this week
                    days_ahead += 7
                target_date = base_time + timedelta(days=days_ahead)
                return target_date.replace(hour=9, minute=0, second=0, microsecond=0)
        
        # Handle absolute dates
        if re.match(r'\d{4}-\d{1,2}-\d{1,2}', reference):
            parts = reference.split('-')
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            return datetime(year, month, day, 9, 0, 0, tzinfo=self.timezone)
        
        # Handle times
        if re.match(r'\d{1,2}:\d{2}', reference):
            time_parts = reference.split(':')
            hour, minute = int(time_parts[0]), int(time_parts[1])
            return base_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # Handle Korean time format
        if '시' in reference:
            hour_match = re.search(r'(\d{1,2})시', reference)
            minute_match = re.search(r'(\d{1,2})분', reference)
            
            hour = int(hour_match.group(1)) if hour_match else 0
            minute = int(minute_match.group(1)) if minute_match else 0
            
            return base_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # Handle "N days later" pattern
        days_later_match = re.search(r'(\d+)일\s*후', reference)
        if days_later_match:
            days = int(days_later_match.group(1))
            return base_time + timedelta(days=days)
        
        # Default: return current time
        return base_time
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query and return parsed results"""
        initial_state = CalendarState(
            messages=[],
            query=query,
            parsed_results=[],
            current_step="",
            thoughts="",
            final_answer=""
        )
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        return {
            "query": query,
            "thoughts": result.get("thoughts", ""),
            "final_answer": result.get("final_answer", ""),
            "parsed_results": [
                {
                    "original_text": r.original_text,
                    "formatted_result": r.formatted_result,
                    "calculated_datetime": r.calculated_datetime.isoformat()
                }
                for r in result.get("parsed_results", [])
            ]
        }

# Example usage and testing
def main():
    # Create the calendar agent
    agent = CalendarReactAgent()
    
    # Test queries
    test_queries = [
        "내일 오후 3시에 미팅이 있어",  
        "다음주 월요일 9시부터 회의",
        "오늘과 내일, 그리고 모레 일정 확인",
        "2024-03-15 14:30에 약속",
        "이번주 금요일 오전 10시",
        "3일 후 오후 2시",
        "Schedule a meeting tomorrow at 3 PM and next Friday at 10 AM"
    ]
    
    print("=== Calendar Agent Test Results ===\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        result = agent.process_query(query)
        
        print("Thoughts:")
        print(result["thoughts"])
        print("\nFinal Answer:")
        print(result["final_answer"])
        
        if result["parsed_results"]:
            print("Detailed Results:")
            for parsed in result["parsed_results"]:
                print(f"  - {parsed['original_text']} → {parsed['formatted_result']}")
        
        print("-" * 50)

if __name__ == "__main__":
    main()
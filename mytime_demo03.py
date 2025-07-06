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

class DateTimeAgent:
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(temperature=0)
        self.reference_date = datetime(2025, 7, 6)  # July 6, 2025 (Sunday)
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(DateTimeState)
        
        # Add nodes
        workflow.add_node("extract_dates", self._extract_dates_node)
        workflow.add_node("calculate_relative_dates", self._calculate_relative_dates_node)
        workflow.add_node("format_dates", self._format_dates_node)
        workflow.add_node("generate_output", self._generate_output_node)
        
        # Add edges
        workflow.set_entry_point("extract_dates")
        workflow.add_edge("extract_dates", "calculate_relative_dates")
        workflow.add_edge("calculate_relative_dates", "format_dates")
        workflow.add_edge("format_dates", "generate_output")
        workflow.add_edge("generate_output", END)
        
        return workflow.compile()
    
    def _extract_dates_node(self, state: DateTimeState) -> DateTimeState:
        """Extract potential date/time mentions from context"""
        context = state["context"]
        
        # Common date/time patterns
        date_patterns = [
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',     # YYYY-MM-DD
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month DD, YYYY
            r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b',  # DD Month YYYY
            r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',  # Time patterns
            r'\b(?:today|tomorrow|yesterday)\b',
            r'\b(?:next|last) (?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(?:this|next) (?:morning|afternoon|evening|night)\b',
            r'\bin \d+ days?\b',
            r'\b\d+ days? ago\b',
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
        """Calculate absolute dates from relative mentions"""
        reference_date = state["reference_date"]
        extracted_dates = state["extracted_dates"]
        calculated_dates = []
        
        for date_mention in extracted_dates:
            try:
                calculated_date = self._parse_date_mention(date_mention, reference_date)
                if calculated_date:
                    calculated_dates.append(calculated_date)
            except Exception as e:
                print(f"Error parsing date mention '{date_mention}': {e}")
                continue
        
        state["extracted_dates"] = calculated_dates
        return state
    
    def _parse_date_mention(self, mention: str, reference_date: datetime) -> datetime:
        """Parse a date mention and return a datetime object"""
        mention_lower = mention.lower().strip()
        
        # Handle relative dates
        if mention_lower == "today":
            return reference_date
        elif mention_lower == "tomorrow":
            return reference_date + timedelta(days=1)
        elif mention_lower == "yesterday":
            return reference_date - timedelta(days=1)
        elif "next week" in mention_lower:
            days_ahead = 7 - reference_date.weekday()
            return reference_date + timedelta(days=days_ahead)
        elif "last week" in mention_lower:
            days_back = reference_date.weekday() + 7
            return reference_date - timedelta(days=days_back)
        elif "next month" in mention_lower:
            if reference_date.month == 12:
                return reference_date.replace(year=reference_date.year + 1, month=1)
            else:
                return reference_date.replace(month=reference_date.month + 1)
        
        # Handle specific days of week
        days_of_week = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        for day_name, day_num in days_of_week.items():
            if day_name in mention_lower:
                current_day = reference_date.weekday()
                if "next" in mention_lower:
                    days_ahead = (day_num - current_day + 7) % 7
                    if days_ahead == 0:
                        days_ahead = 7
                    return reference_date + timedelta(days=days_ahead)
                elif "last" in mention_lower:
                    days_back = (current_day - day_num + 7) % 7
                    if days_back == 0:
                        days_back = 7
                    return reference_date - timedelta(days=days_back)
                else:
                    # This week
                    days_ahead = (day_num - current_day) % 7
                    return reference_date + timedelta(days=days_ahead)
        
        # Handle "in X days" pattern
        in_days_match = re.match(r'in (\d+) days?', mention_lower)
        if in_days_match:
            days = int(in_days_match.group(1))
            return reference_date + timedelta(days=days)
        
        # Handle "X days ago" pattern
        ago_match = re.match(r'(\d+) days? ago', mention_lower)
        if ago_match:
            days = int(ago_match.group(1))
            return reference_date - timedelta(days=days)
        
        # Handle time-only mentions
        time_match = re.match(r'(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM|am|pm)?', mention)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            second = int(time_match.group(3)) if time_match.group(3) else 0
            ampm = time_match.group(4)
            
            if ampm and ampm.lower() == 'pm' and hour != 12:
                hour += 12
            elif ampm and ampm.lower() == 'am' and hour == 12:
                hour = 0
            
            return reference_date.replace(hour=hour, minute=minute, second=second)
        
        # Try to parse as absolute date
        try:
            parsed_date = date_parser.parse(mention, default=reference_date)
            return parsed_date
        except:
            return None
    
    def _format_dates_node(self, state: DateTimeState) -> DateTimeState:
        """Format dates to yy-mm-dd hh:mm:ss format"""
        dates = state["extracted_dates"]
        formatted_dates = []
        
        for date_obj in dates:
            if isinstance(date_obj, datetime):
                formatted = date_obj.strftime("%y-%m-%d %H:%M:%S")
                formatted_dates.append(f'"{formatted}"')
        
        # Remove duplicates while preserving order
        seen = set()
        unique_formatted = []
        for date in formatted_dates:
            if date not in seen:
                seen.add(date)
                unique_formatted.append(date)
        
        state["extracted_dates"] = unique_formatted
        return state
    
    def _generate_output_node(self, state: DateTimeState) -> DateTimeState:
        """Generate final output in required format"""
        formatted_dates = state["extracted_dates"]
        
        if formatted_dates:
            output = "<datetime_list>\n" + ",\n".join(formatted_dates) + "\n</datetime_list>"
        else:
            output = "<datetime_list>\n</datetime_list>"
        
        state["final_output"] = output
        return state
    
    def process(self, context: str) -> str:
        """Process the context and extract formatted dates"""
        initial_state = {
            "context": context,
            "reference_date": self.reference_date,
            "extracted_dates": [],
            "messages": [],
            "final_output": ""
        }
        
        result = self.graph.invoke(initial_state)
        return result["final_output"]

# Usage example
def main():
    # Initialize the agent
    agent = DateTimeAgent()
    
    # Test cases
    test_contexts = [
        "지난 수요일에는 밤10시부터 새벽 1시까지 회의가 있었습니다.",
        "다음 주 월요일 오전 9시에 회의가 있습니다.",
        "오늘 오후 3시에 만나자고 했는데, 내일은 오후 2시 30분에 회의가 있습니다.",
    ]
    
    for i, context in enumerate(test_contexts, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Context: {context}")
        result = agent.process(context)
        print(f"Result: {result}")

if __name__ == "__main__":
    main()
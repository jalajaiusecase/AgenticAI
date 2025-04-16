
import chainlit as cl
import uuid
from typing import cast, List
from pydantic import BaseModel

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_core.memory import ListMemory, MemoryContent
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from pathlib import Path
import asyncio
import random

import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

## GLOBAL SETTINGS
BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "llama-3.3-70b-versatile"
API_KEY = "Ygsk_lELhhaYeEraZOdOhW4rBWGdyb3FYs8QEgZCiiPlGqWYPvL4uSLhK"

# Initialize the model client
model_client = OpenAIChatCompletionClient(
    model=MODEL_NAME,
    base_url=BASE_URL,
    api_key=API_KEY,
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": ModelFamily.ANY,
    },
)

# Context classes for agents
class AirlineAgentContext(BaseModel):
    passenger_name: str | None = None
    confirmation_number: str | None = None
    seat_number: str | None = None
    flight_number: str | None = None

# MCP Tools
async def webfetch_mcp_tool(task: str) -> str:
    """Fetch web url content and summarize it"""
    fetch_mcp_server = StdioServerParams(command="uvx", args=["mcp-server-fetch"])
    tools = await mcp_server_tools(fetch_mcp_server)
    agent = AssistantAgent(name="fetcher", model_client=model_client, tools=tools, reflect_on_tool_use=True)
    result = await agent.run(task=f"{task}")
    return result.messages[-1].content

async def files_mcp_tool(task: str, work_dir: str = "./mcp_working") -> str:
    """Provides direct access to local file systems"""
    path = Path(work_dir)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    
    allow_dirs = work_dir
    server_params = StdioServerParams(
        command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", allow_dirs]
    )
    tools = await mcp_server_tools(server_params)
    agent = AssistantAgent(
        name="file_manager",
        model_client=model_client, 
        tools=tools,
    )
    result = await agent.run(task=task, cancellation_token=CancellationToken())
    return result.messages[-1].content

# User Memory Management
_user_preference = None
_past_experience = None

async def load_user_preferences():
    """Load user preferences and past experiences from memory"""
    global _user_preference
    global _past_experience    
    
    _user_preference = ListMemory("user_preference")
    user_preferences = """ 
    Food Preferences:
    - User likes pizza.
    - User dislikes cheese.
    - User dislikes spicy pepper.
    - Food preference types: [window, vegetarian or vegan]
    - Food allergies: ["peanuts", "shellfish"]

    Seat Preferences:
    - Preferred seat types are window, aisle and emergency exit.
    - seat_preferences: ["first_class", "business_class", "economy_class"]
    
    Airline Preferences:
    - Preferred airline: "Air Canada"
    - Allowed airlines: ["Air Canada", "Qantas", "Delta"]

    Flight Time Preferences:
    - Preferred departure time: "08:00 AM"
    - Preferred arrival time: "05:00 PM"
    """    
    await _user_preference.add(
        MemoryContent(
            content=user_preferences,
            mime_type="text/markdown",
            metadata={"category": "preferences", "type": "units"},
        )
    )

    _past_experience = ListMemory("past_experience")
    user_past_experience = """ 
    Recent Flight experiences:
    - Air Canada morning flight from toronto to new york(2025-03-07)
        - Rating: 4/5
        - Liked: window seat, pizza
        - Disliked: tight seat        
        
    - American Airline night flight from new york to toronto(2025-03-08)
        - Rating: 4/5
        - Liked: window seat, coffee
        - Disliked: smaller seat size        
    """
    
    await _past_experience.add(
        MemoryContent(
            content=user_past_experience,
            mime_type="text/markdown",
            metadata={"category": "preferences", "type": "units"},
        )
    )
    
    return _user_preference, _past_experience

# FAQ Tool
async def faq_lookup_tool(question: str) -> str:
    """Lookup frequently asked questions about bag, seats and wifi."""
    if "bag" in question.lower() or "baggage" in question.lower() or "luggage" in question.lower():
        return (
            "You are allowed to bring one carry-on bag and one personal item on the plane. "
            "Carry-on must be under 50 pounds and 22 inches x 14 inches x 9 inches. "
            "You can check up to 2 bags with most tickets, with additional fees for extra baggage."
        )
    elif "seat" in question.lower() or "plane" in question.lower() or "aircraft" in question.lower():
        return (
            "Our standard planes have 150 seats. "
            "There are 24 business class seats and 126 economy seats. "
            "Exit rows are rows 10 and 24. "
            "Rows 11-15 are Economy Plus, with extra legroom. "
            "Window and aisle seats can be selected during booking."
        )
    elif "wifi" in question.lower() or "internet" in question.lower():
        return "We have high-speed WiFi available on all flights. Connect to 'SkyConnect-WiFi' once onboard. Basic browsing is free, and premium streaming packages are available for purchase."
    elif "food" in question.lower() or "meal" in question.lower() or "eat" in question.lower():
        return "We offer complimentary snacks on all flights. Flights over 3 hours include a meal service. Business class includes full meal service with beverages. Special meals can be requested 48 hours before departure."
    elif "cancel" in question.lower() or "refund" in question.lower():
        return "Cancellations made more than 24 hours before departure are eligible for a full refund or flight credit. Changes within 24 hours may incur a fee depending on your ticket type."
    return "I don't have specific information about that question. Would you like me to connect you with our flight booking specialist for more details?"

# Seat Booking Tool
async def update_seat_tool(confirmation_number: str, new_seat: str, flight_number: str) -> str:
    """Update the seat for a given confirmation number."""
    valid_seats = ["A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2"]
    
    if new_seat not in valid_seats:
        return f"Sorry, seat {new_seat} is not available. Available seats are: {', '.join(valid_seats)}"
    
    return f"Successfully updated seat to {new_seat} for confirmation number {confirmation_number} on flight {flight_number}"

# Create the memory agent
async def user_memory_tool(query: str) -> str:
    """Lookup user preferences about flight, airline, flight time, foods and past experience"""
    global _memory_agent
    
    if not _user_preference or not _past_experience:
        await load_user_preferences()
        
    _memory_agent = AssistantAgent(
        name="memory_assistant",
        model_client=model_client,
        memory=[_user_preference, _past_experience],
        system_message="You are a helpful preference and memory assistant.",
        reflect_on_tool_use=True,
    )
    
    response = await _memory_agent.on_messages(
        [TextMessage(content=query, source="user")], CancellationToken()
    )
    
    return response.chat_message.content

# Initialize Agents
async def initialize_agents():
    """Initialize the booking and FAQ agents"""
    faq_agent = AssistantAgent(
        name="faq_agent",
        model_client=model_client,
        system_message="""You are a helpful airline FAQ agent. Answer user questions about baggage policies, 
        seating arrangements, flight amenities, and other common flight-related questions. Use the FAQ lookup tool 
        when appropriate rather than relying on your own knowledge.""",
        tools=[faq_lookup_tool]
    )
    
    booking_agent = AssistantAgent(
        name="booking_agent",
        model_client=model_client,
        system_message="""You are a helpful flight booking agent. Assist users with booking flights,
        selecting seats, and managing their reservations. Collect necessary information such as departure/arrival 
        locations, dates, preferences, and help with seat selection.""",
        tools=[update_seat_tool, user_memory_tool, webfetch_mcp_tool, files_mcp_tool]
    )
    
    return faq_agent, booking_agent

# Chainlit Chat UI Setup
@cl.set_chat_profiles
async def chat_profile(current_user: cl.User):
    return [
        cl.ChatProfile(
            name="Flight Booking Assistant",
            icon="https://picsum.photos/250",
            markdown_description="Your personal flight booking assistant powered by Groq and Autogen",
            starters = [
                cl.Starter(
                    label="Greetings",
                    message="Hello! I'd like to book a flight.",
                ),
                cl.Starter(
                    label="Baggage Policy",
                    message="What's your baggage policy for international flights?",
                ),
                cl.Starter(
                    label="Seat Selection",
                    message="I'd like to book a window seat for my upcoming flight.",
                ),
                cl.Starter(
                    label="Airline Recommendations",
                    message="Can you recommend a good airline for domestic US travel?",
                ),
                cl.Starter(
                    label="Food Options",
                    message="What food options do you have on long-haul flights?",
                ),
            ]
        )
    ]

@cl.on_chat_start
async def start_chat() -> None:
    # Create the TaskList
    task_list = cl.TaskList(name="Booking Tasks")
    task_list.status = "Initializing..."
    await task_list.send()
    
    # Load user preferences
    await load_user_preferences()
    
    # Initialize agents
    faq_agent, booking_agent = await initialize_agents()
    
    # Store in session
    cl.user_session.set("message_history", [{"role": "system", "content": "You are a helpful flight booking assistant"}])
    cl.user_session.set("faq_agent", faq_agent)
    cl.user_session.set("booking_agent", booking_agent)
    cl.user_session.set("current_agent", booking_agent)  # Default to booking agent
    cl.user_session.set("task_list", task_list)
    cl.user_session.set("context", {
        "passenger_name": None,
        "confirmation_number": None,
        "flight_number": None,
        "seat_number": None
    })
    
    task_list.status = "Ready"
    await task_list.send()
    
    await cl.Message(content="Hello! I'm your flight booking assistant. How can I help you today?").send()

@cl.on_message
async def chat(message: cl.Message) -> None:
    # Get session data
    task_list = cast(cl.TaskList, cl.user_session.get("task_list"))
    booking_agent = cl.user_session.get("booking_agent")
    faq_agent = cl.user_session.get("faq_agent")
    context = cl.user_session.get("context")
    
    # Update message history
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})
    cl.user_session.set("message_history", message_history)
    
    # Determine which agent to use based on message content
    current_query = message.content.lower()
    current_agent = booking_agent
    
    if any(keyword in current_query for keyword in ["baggage", "luggage", "bag", "wifi", "food", "meal", "seat policy"]):
        current_agent = faq_agent
        task = cl.Task(title="Processing FAQ request", status=cl.TaskStatus.RUNNING)
    else:
        current_agent = booking_agent
        task = cl.Task(title="Processing booking request", status=cl.TaskStatus.RUNNING)
    
    await task_list.add_task(task)
    
    # Create status message
    status_msg = cl.Message(content="Processing your request...")
    await status_msg.send()
    
    try:
        # Process with the appropriate agent
        if current_agent == faq_agent:
            # For FAQ queries
            response = await faq_lookup_tool(message.content)
            await cl.Message(content=f"faq_agent: {response}").send()
        else:
            # For booking related queries
            # Check if it's a web lookup request
            if "recommend" in current_query or "compare" in current_query or "review" in current_query:
                task.status = cl.TaskStatus.RUNNING
                task.title = "Fetching web information"
                await task_list.send()
                
                # If it's a web search query
                web_result = await webfetch_mcp_tool(message.content)
                await cl.Message(content=f"booking_agent: {web_result}").send()
            
            # Check if it's a seat booking request
            elif "book" in current_query and "seat" in current_query:
                # Generate a confirmation number if not exists
                if not context["confirmation_number"]:
                    context["confirmation_number"] = f"CONF-{uuid.uuid4().hex[:6].upper()}"
                
                # Generate a flight number if not exists
                if not context["flight_number"]:
                    context["flight_number"] = f"FL-{random.randint(1000, 9999)}"
                
                # Suggest available seats
                available_seats = ["A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2"]
                user_preferences = await user_memory_tool("What are the user's seat preferences?")
                
                seat_suggestion = "Based on your preferences, I recommend a window seat."
                if "window" in user_preferences.lower():
                    seat_suggestion = "Based on your preference for window seats, I recommend seat A1 or A2."
                elif "aisle" in user_preferences.lower():
                    seat_suggestion = "Based on your preference for aisle seats, I recommend seat C1 or C2."
                
                response = (
                    f"I'm ready to book your seat on flight {context['flight_number']}.\n"
                    f"Your confirmation number is {context['confirmation_number']}.\n"
                    f"Available seats: {', '.join(available_seats)}\n\n"
                    f"{seat_suggestion}\n\n"
                    f"Which seat would you like to book?"
                )
                
                await cl.Message(content=f"Booking Agent: {response}").send()
                cl.user_session.set("context", context)
            
            # Process seat selection
            elif any(seat in current_query for seat in ["a1", "a2", "b1", "b2", "c1", "c2", "d1", "d2"]):
                # Extract the seat from the message
                for seat in ["A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2"]:
                    if seat.lower() in current_query:
                        selected_seat = seat
                        break
                else:
                    selected_seat = "A1"  # Default if no match
                
                # Update the seat
                booking_response = await update_seat_tool(
                    context["confirmation_number"], 
                    selected_seat,
                    context["flight_number"]
                )
                
                # Update context
                context["seat_number"] = selected_seat
                cl.user_session.set("context", context)
                
                # Response with booking confirmation
                response = (
                    f"{booking_response}\n\n"
                    f"Your booking details:\n"
                    f"- Flight: {context['flight_number']}\n"
                    f"- Confirmation: {context['confirmation_number']}\n"
                    f"- Seat: {context['seat_number']}\n\n"
                    f"Is there anything else you need help with today?"
                )
                
                await cl.Message(content=f"Booking Agent: {response}").send()
            else:
                # For general booking queries, check if preferences should be used
                if any(word in current_query for word in ["prefer", "like", "recommend", "suggest"]):
                    preferences = await user_memory_tool("What are the user's preferences?")
                    response = (
                        f"Based on your preferences, I can help with personalized recommendations:\n\n"
                        f"{preferences}\n\n"
                        f"How would you like to proceed with your booking?"
                    )
                else:
                    response = (
                        "I can help you book a flight. To get started, I'll need the following information:\n\n"
                        "1. Departure city and arrival city\n"
                        "2. Travel dates\n"
                        "3. Number of passengers\n"
                        "4. Preferred class (economy, business, first)\n\n"
                        "Please provide these details so I can find the best options for you."
                    )
                
                await cl.Message(content=f"Booking Agent: {response}").send()
        
        # Update task status
        task.status = cl.TaskStatus.DONE
        await task_list.send()
        
    except Exception as e:
        task.status = cl.TaskStatus.FAILED
        await task_list.send()
        await cl.Message(content=f"I encountered an error: {str(e)}. Please try again.").send()
    
    # Remove the status message
    await status_msg.remove()

@cl.on_stop
def on_stop():
    print("The user wants to stop the task!")

if __name__ == "__main__":
    # Initialize the memory before starting the app
    asyncio.run(load_user_preferences())
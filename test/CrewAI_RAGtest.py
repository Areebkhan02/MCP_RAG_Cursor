from crewai import Crew, Agent, Task, LLM
from crewai_tools import RagTool
import os

# Initialize Gemini model with proper error handling
os.environ["GOOGLE_API_KEY"]  # Replace with your actual key

# Add verbose logging
import logging
logging.basicConfig(level=logging.INFO)

llm = LLM(
    api_key=os.environ["GOOGLE_API_KEY"],
    model="gemini/gemini-2.0-flash",
)

# Create a RAG tool with correct configuration settings
config = {
    "llm": {
        "provider": "google",
        "config": {
            "model": "models/gemini-2.0-flash",
            "temperature": 0.1
        }
    },
    "embedder": {
        "provider": "google",
        "config": {
            "model": "models/embedding-001"
        }
    }
}

rag_tool = RagTool(config=config)

# Add content from the web page with error handling
try:
    print("Adding documentation to knowledge base...")
    rag_tool.add(
        source="https://docs.crewai.com/concepts/memory/",  # More specific URL about memory
        data_type="web_page"
    )
    print("Documentation added successfully.")
except Exception as e:
    print(f"Error adding documentation: {e}")

# Define an agent
def knowledge_expert() -> Agent:
    return Agent(
        role='Knowledge Expert',
        goal='Answer questions accurately using the provided knowledge base',
        backstory='I am an AI expert with deep knowledge of the CrewAI documentation and ability to provide accurate information.',
        allow_delegation=False,
        tools=[rag_tool],
        llm=llm,
        verbose=True  # Enable verbose mode for debugging
    )

# Create a task for the agent
def get_user_query():
    return input("Please enter your question about CrewAI: ")

# Create an instance of the agent
expert = knowledge_expert()

while True:
    user_query = get_user_query()
    task = Task(
        description=f"Answer this question about CrewAI: {user_query}",
        expected_output="Detailed answer based on the CrewAI documentation",
        agent=expert
    )
    
    # Create a crew with the agent instance
    crew = Crew(
        agents=[expert],
        tasks=[task],
        verbose=True  # Enable verbose mode for debugging
    )
    
    # Run the crew
    result = crew.kickoff()
    print(result)
    
    if input("\nDo you want to ask another question? (yes/no): ").lower() != 'yes':
        break
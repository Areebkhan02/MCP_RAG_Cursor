from crewai import Crew, Agent, Task, LLM
from crewai_tools import RagTool
import os
import json
import asyncio
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv

load_dotenv()

# Keep the API key setup, but ideally get this from environment variables
# in a production environment rather than hardcoding it
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class URLMonitor(FileSystemEventHandler):
    def __init__(self, rag_tool):
        self.rag_tool = rag_tool
        self.url_cache = {}
        
    def load_urls(self, filename):
        try:
            with open(filename, 'r') as f:
                content = f.read().strip()
                # Try parsing as JSON first
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # If not JSON, try parsing as plain text with URLs
                    urls = []
                    for line in content.split('\n'):
                        # Remove quotes and whitespace
                        line = line.strip().strip('"').strip("'").strip()
                        if line:  # Only add non-empty lines
                            urls.append(line)
                    return urls
        except Exception as e:
            print(f"Error reading file {filename}: {str(e)}")
            return []
            
    def update_knowledge_base(self, filename):
        print(f"\n🔄 Updating knowledge base from {filename}")
        new_urls = self.load_urls(filename)
        print(f"🔍 Found URLs: {new_urls}")
        
        # If this file was processed before, find new/removed URLs
        if filename in self.url_cache:
            old_urls = self.url_cache[filename]
            urls_to_add = set(new_urls) - set(old_urls)
            print(f"📌 New URLs to add: {urls_to_add}")
        else:
            urls_to_add = new_urls
            
        # Update the knowledge base
        for url in urls_to_add:
            try:
                print(f"📥 Adding content from: {url}")
                self.rag_tool.add(
                    source=url,
                    data_type="web_page"
                )
                print(f"✅ Successfully added: {url}")
            except Exception as e:
                print(f"❌ Error adding {url}: {str(e)}")
                
        # Update cache
        self.url_cache[filename] = new_urls
        
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.txt'):
            self.update_knowledge_base(event.src_path)

class CrewAI_RAG:
    def __init__(self):
        self.rag_tool = None
        self.llm = None
        self.observer = None
        self.url_monitor = None
        self.setup_rag_system()
        self.setup_url_monitor()
        
    def setup_rag_system(self):
        # Initialize LLM and RAG tool
        self.llm = LLM(
            api_key=GOOGLE_API_KEY,
            model="gemini/gemini-2.0-flash",
        )
        
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
        
        self.rag_tool = RagTool(config=config)
    
    def setup_url_monitor(self):
        # Setup directory monitoring
        self.url_monitor = URLMonitor(self.rag_tool)
        input_dir = "input_urls"
        os.makedirs(input_dir, exist_ok=True)
        
        # Initial load of all URL files
        url_file = os.path.join(input_dir, "input_urls.txt")
        if os.path.exists(url_file):
            self.url_monitor.update_knowledge_base(url_file)
        
        # Setup directory observer
        self.observer = Observer()
        self.observer.schedule(self.url_monitor, input_dir, recursive=False)
        self.observer.start()
        print(f"🔍 Monitoring {input_dir} directory for changes...")
        
    def knowledge_expert(self) -> Agent:
        return Agent(
            role='Knowledge Expert',
            goal='Answer questions accurately using the provided knowledge base',
            backstory='I am an AI expert with deep knowledge of the provided content.',
            allow_delegation=False,
            tools=[self.rag_tool],
            llm=self.llm,
            verbose=True
        )
    
    async def query(self, query_text):
        """Query the RAG system with the given text."""
        print(f"Processing query: {query_text}")
        
        # Create agent and task
        expert = self.knowledge_expert()
        
        task = Task(
            description=f"Answer this question: {query_text}",
            expected_output="Detailed answer based on the knowledge base",
            agent=expert
        )
        
        crew = Crew(
            agents=[expert],
            tasks=[task],
            verbose=True
        )
        
        # Run the query in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, crew.kickoff)
        
        print(f"Query complete: {query_text}")
        return result
    
    def stop(self):
        """Stop the file observer."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            print("File monitoring stopped.")

# Keep the main function for standalone usage
def main():
    rag = CrewAI_RAG()
    
    try:
        print("\n🤖 RAG System Ready! Ask your questions (or type 'exit' to quit)")
        
        while True:
            user_query = input("\n❓ Your question: ")
            if user_query.lower() == 'exit':
                break
                
            result = asyncio.run(rag.query(user_query))
            print("\n📝 Answer:", result)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        rag.stop()

if __name__ == "__main__":
    main()
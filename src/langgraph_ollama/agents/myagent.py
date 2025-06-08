from datetime import datetime
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph


class MyAgent:
    def __init__(self, llm: ChatOllama):
        self.llm = llm
        self.agent = None

    async def create_agent(self) -> CompiledStateGraph:
        """
        Create the agent with the specified LLM and tools.

        :return: An instance of the agent.
        """
        client = MultiServerMCPClient(
            {
                "gradio": {
                    "url": "https://opertifelipe-mcp-geo-info.hf.space/gradio_api/mcp/sse",
                    "transport": "sse",
                }
            }
        )
        tools_mcp = await client.get_tools()

        def get_datetime() -> str:
            """This function returns the current date and time."""
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        tools = tools_mcp + [get_datetime]

        self.agent = create_react_agent(model=self.llm, tools=tools)
        return self.agent

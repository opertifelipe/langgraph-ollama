from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import Literal
from langgraph_ollama.agents.myagent import MyAgent
from langchain_core.prompts import ChatPromptTemplate


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    intent: str


class MyGraph:
    def __init__(self, llm: ChatOllama):
        self.llm = llm

    async def node_orchestrator(self, state: State):
        class Intent(BaseModel):
            intent: Literal["agent", "entity_extractor"] = Field(
                ...,
                description="The intent of the user could be to interact to extract the entitities Name and Surname from a text or to interact with an agent able to provide the current date and time or count the number of a specific letter in a text.",
            )

        structured_llm = self.llm.with_structured_output(Intent)
        system = """You are an orchestrator and your aim is to determine the user's intent based on the messages provided.
The intent could be:
- extract the entities Name and Surname
- interact with an agent able to provide the current date and time or count the number of a specific letter in a text.
"""
        prompt = ChatPromptTemplate.from_messages(
            [("system", system), ("human", "{input}")]
        )
        chain = prompt | structured_llm
        intent = await chain.ainvoke(state["messages"])
        state["intent"] = intent
        return state

    async def node_extractor(self, state: State):
        class Person(BaseModel):
            name: str = Field(None, description="The name of the person")
            surname: str = Field(None, description="The surname of the person")

        structured_llm = self.llm.with_structured_output(Person)
        person: Person = await structured_llm.ainvoke(state["messages"])
        message = f"Extracted Name: {person.name}, Surname: {person.surname}"
        return {"messages": [("assistant", message)]}

    async def node_agent(self, state: State):
        agent = await MyAgent(self.llm).create_agent()
        res = await agent.ainvoke({"messages": state["messages"]})
        return {"messages": res["messages"]}

    async def edge_orchestrator(self, state: State) -> Literal["agent", "extractor"]:
        if state["intent"].intent == "agent":
            return "agent"
        elif state["intent"].intent == "entity_extractor":
            return "extractor"
        else:
            raise ValueError(f"Unknown intent: {state['intent'].intent}")

    async def create_graph(self):
        graph_builder = StateGraph(State)
        graph_builder.add_node("orchestrator", self.node_orchestrator)
        graph_builder.add_node("extractor", self.node_extractor)
        graph_builder.add_node("agent", self.node_agent)
        graph_builder.add_conditional_edges("orchestrator", self.edge_orchestrator)
        graph_builder.add_edge(START, "orchestrator")
        graph_builder.add_edge("agent", END)
        graph_builder.add_edge("extractor", END)
        graph = graph_builder.compile()
        return graph

import typer
from rich import print
from rich.prompt import Prompt, Confirm

from langgraph_ollama.graphs.mygraph import MyGraph
from langgraph_ollama.interface.langchain import InterfaceLangchain
import asyncio
from typing_extensions import Annotated
import os
app = typer.Typer()


async def run_chatbot():
    llm = InterfaceLangchain().get_llm()
    graph = await MyGraph(llm).create_graph()
    print(
        "[bold red]This chatbot allows you to interact with an agent or extract entities Name and Surname from from a text.[/bold red]"
    )
    messages = []
    while True:
        user_input = Prompt.ask("[bold blue]You[/bold blue]")
        if user_input.lower() == "exit":
            print("Exiting chatbot.")
            break
        messages.append(("human", user_input))
        state = {"messages": messages}
        state = await graph.ainvoke(state)
        response = state["messages"][-1].content
        messages.append(("assistant", response))
        print(f"[bold green]Bot:[/bold green] {response}")
        print("Type 'exit' to quit the chatbot.")


@app.command()
def chatbot():
    """
    Start the chatbot
    """
    asyncio.run(run_chatbot())


@app.command()
def draw_graph(
    filepath: Annotated[str, typer.Option(help="File path to save the graph image.")] = os.path.join("images", "mygraph.png"),
):
    """
    Draw the graph
    """
    llm = InterfaceLangchain().get_llm()
    graph = asyncio.run(MyGraph(llm).create_graph())
    image = graph.get_graph().draw_mermaid_png()
    with open(filepath, "wb") as f:
        f.write(image)
    print(f"[bold green]Graph drawn and saved as {filepath}[/bold green]")

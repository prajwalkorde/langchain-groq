from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from common import get_chat_model


@tool
def calculate_margin(revenue: float, cost: float) -> float:
    """Calculate profit margin percentage from revenue and cost."""

    profit = revenue - cost
    return round((profit / revenue) * 100, 2)


@tool
def recommend_dashboard(metric: str) -> str:
    """Recommend a dashboard type for a business metric."""

    metric = metric.lower()
    if "sales" in metric or "revenue" in metric:
        return "Use an executive revenue dashboard with trend, target, and segment views."
    if "support" in metric or "ticket" in metric:
        return "Use an operations dashboard with backlog, SLA, and issue category views."
    return "Use a discovery dashboard with filters, breakdowns, and anomaly notes."


def main() -> None:
    llm = get_chat_model()
    tools = [calculate_margin, recommend_dashboard]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a business analytics assistant. Use tools when calculations or lookups are needed.",
            ),
            ("human", "{input}"),
            # The agent scratchpad stores intermediate tool calls and observations.
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # create_tool_calling_agent works with chat models that support tool calling.
    # Groq-hosted Llama models support this style through LangChain's ChatGroq wrapper.
    agent = create_tool_calling_agent(llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
    )

    result = executor.invoke(
        {
            "input": (
                "Revenue is 50000 and cost is 31000. What is the margin, "
                "and what dashboard should we use for revenue?"
            )
        }
    )

    print(result["output"])


if __name__ == "__main__":
    main()

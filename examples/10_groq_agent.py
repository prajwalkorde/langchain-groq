from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
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

    system_prompt = (
        "You are a business analytics assistant. "
        "Use tools when calculations or lookups are needed."
    )

    # create_agent builds the modern LangChain agent loop.
    # The agent can read the user message, decide whether a tool is needed,
    # call that tool, inspect the result, and then produce a final answer.
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Revenue is 50000 and cost is 31000. What is the margin, "
                        "and what dashboard should we use for revenue?"
                    )
                )
            ]
        }
    )

    # The agent returns the full message history.
    # The final assistant response is the last message in that history.
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()

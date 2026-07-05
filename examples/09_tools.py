from langchain_core.tools import tool


@tool
def calculate_margin(revenue: float, cost: float) -> float:
    """Calculate profit margin percentage from revenue and cost."""

    # Tools are normal Python functions with descriptions.
    # Agents use the name, docstring, and type hints to decide when to call them.
    profit = revenue - cost
    return round((profit / revenue) * 100, 2)


@tool
def lookup_customer_tier(monthly_spend: float) -> str:
    """Return a customer tier based on monthly spend."""

    if monthly_spend >= 10000:
        return "enterprise"
    if monthly_spend >= 2500:
        return "growth"
    return "starter"


def main() -> None:
    # You can call tools directly like normal functions during testing.
    print(calculate_margin.invoke({"revenue": 12000, "cost": 7800}))
    print(lookup_customer_tier.invoke({"monthly_spend": 3200}))


if __name__ == "__main__":
    main()

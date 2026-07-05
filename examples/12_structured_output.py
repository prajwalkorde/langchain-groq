from pydantic import BaseModel, Field

from common import get_chat_model


class StudyPlan(BaseModel):
    topic: str = Field(description="The topic being studied")
    difficulty: str = Field(description="beginner, intermediate, or advanced")
    steps: list[str] = Field(description="Ordered study steps")


def main() -> None:
    llm = get_chat_model()

    # with_structured_output asks the model to return data matching a Pydantic schema.
    # This is better than hoping the model prints valid JSON.
    structured_llm = llm.with_structured_output(StudyPlan)

    plan = structured_llm.invoke(
        "Create a 3-step study plan for mastering LangChain agents."
    )

    print(plan)
    print(plan.steps[0])


if __name__ == "__main__":
    main()

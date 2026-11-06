import pytest

from widowxai_quest_teleop.model import WidowXAIModel


@pytest.fixture(scope="session")
def model() -> WidowXAIModel:
    return WidowXAIModel()


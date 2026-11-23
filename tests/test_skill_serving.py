from ace import Skillbook

from app.agent import RetailAgent
from app.harness import BASE_CODE
from app.simulation import get_environment


def test_procedural_harness_skills_reach_the_business_agent_in_skill_modes():
    book = Skillbook()
    business = book.add_skill(
        "context",
        issue="Business policy pattern",
        insight="For pending orders, confirm status before editing.",
    )
    procedure = book.add_skill(
        "harness",
        issue="Confirmation is mistaken for execution",
        insight="After confirmation, execute the business write tool.",
    )
    diagnostic = book.add_skill("harness", issue="No reliable workaround yet")
    version = {"harness_code": BASE_CODE, "skillbook": book.to_dict()}
    for mode in ["skills", "combined"]:
        agent = RetailAgent(get_environment(), version, mode)
        assert set(agent.injected_skill_ids) == {business.id, procedure.id}
        assert procedure.insight in agent.system_prompt
        assert business.insight in agent.system_prompt
        assert diagnostic.id not in agent.system_prompt
    for mode in ["baseline", "harness"]:
        agent = RetailAgent(get_environment(), version, mode)
        assert not agent.injected_skill_ids
        assert procedure.insight not in agent.system_prompt

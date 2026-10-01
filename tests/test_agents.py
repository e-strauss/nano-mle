from nano_mle.agents import lm_settings


def test_gpt61_sol_keeps_a_bounded_reasoning_request_without_sampling():
    settings = lm_settings("openai/gpt-6.1-sol", 6000)
    assert settings["temperature"] is None
    assert settings["max_tokens"] is None
    assert settings["max_completion_tokens"] == 6000
    assert settings["reasoning_effort"] == "low"
    assert settings["num_retries"] == 0


def test_non_openai_models_keep_the_existing_settings():
    settings = lm_settings("gemini/gemini-3.8-flash", 6000)
    assert settings["temperature"] == 0.2
    assert settings["max_tokens"] == 6000
    assert "reasoning_effort" not in settings

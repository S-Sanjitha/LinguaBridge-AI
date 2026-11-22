import sys
from unittest.mock import Mock, call

import pytest

from linguabridge import __main__ as cli
from linguabridge.agent import Context, Decision, Result


def run_cli(monkeypatch, args, responses=()):
    monkeypatch.setattr(sys, "argv", ["linguabridge", *args])
    answers = iter(responses)
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    cli.main()


def test_single_marian_output_unchanged(monkeypatch, capsys):
    mt = Mock()
    mt.translate.return_value = "Bonjour."
    load_agent = Mock()
    monkeypatch.setattr(cli, "MarianTranslator", Mock(return_value=mt))
    monkeypatch.setattr(cli, "create_agent", load_agent)
    run_cli(monkeypatch, ["--mt-only"], ["Hello."])
    assert capsys.readouterr().out == "Bonjour.\n"
    load_agent.assert_not_called()


def test_single_agent_displays_refinement_and_passes_context(monkeypatch, capsys):
    agent = Mock()
    agent.translate.return_value = Result("Tu es prêt ?", Decision.REFINE, "Tu es prête ?")
    monkeypatch.setattr(cli, "create_agent", Mock(return_value=agent))
    run_cli(monkeypatch, ["Are you ready?", "--formality", "informal", "--listener-gender", "feminine"])
    agent.translate.assert_called_once_with(
        "Are you ready?", Context(formality="informal", listener_gender="feminine"),
    )
    output = capsys.readouterr().out
    assert "MarianMT: Tu es prêt ?" in output
    assert "Decision: REFINE" in output
    assert "Explanation:" in output
    assert "Final French: Tu es prête ?" in output


def test_interactive_reuses_agent_and_scopes_clarification(monkeypatch, capsys):
    agent = Mock()
    agent.translate.side_effect = [
        Result("Je vois une batte.", Decision.CLARIFY, question="Animal or baseball equipment?"),
        Result("Je vois une batte.", Decision.ACCEPT_MT, "Je vois une batte."),
        Result("Tu pars.", Decision.FLAG, reason="Conflicting context."),
    ]
    load_agent = Mock(return_value=agent)
    monkeypatch.setattr(cli, "create_agent", load_agent)
    run_cli(monkeypatch, ["-i", "--formality", "informal"], [
        "", "I see a bat.", "", "Baseball equipment.", "You are leaving.", "", ":quit",
    ])
    load_agent.assert_called_once_with()
    assert agent.translate.call_args_list == [
        call("I see a bat.", Context(formality="informal")),
        call("I see a bat.", Context(formality="informal", meaning="Baseball equipment.")),
        call("You are leaving.", Context(formality="informal")),
    ]
    output = capsys.readouterr().out
    assert "Enter non-empty English text." in output
    assert output.count("Status: Unresolved") == 2
    assert output.count("Final French:") == 1
    assert "Details: Conflicting context." in output
    assert "Goodbye." in output


@pytest.mark.parametrize("error", [EOFError, KeyboardInterrupt])
def test_interactive_quits_without_loading_on_eof_or_interrupt(monkeypatch, capsys, error):
    load_agent = Mock()
    monkeypatch.setattr(cli, "create_agent", load_agent)
    monkeypatch.setattr(sys, "argv", ["linguabridge", "-i"])
    monkeypatch.setattr("builtins.input", Mock(side_effect=error))
    cli.main()
    load_agent.assert_not_called()
    assert "Goodbye." in capsys.readouterr().out


@pytest.mark.parametrize("args", [
    [" "], ["Hello", "--formality", "casual"], ["Hello", "--mt-only", "--meaning", "Context"],
    ["Hello", "--interactive"],
])
def test_invalid_arguments_do_not_load_models(monkeypatch, args):
    load_agent, load_mt = Mock(), Mock()
    monkeypatch.setattr(cli, "create_agent", load_agent)
    monkeypatch.setattr(cli, "MarianTranslator", load_mt)
    with pytest.raises(SystemExit) as error:
        run_cli(monkeypatch, args)
    assert error.value.code == 2
    load_agent.assert_not_called()
    load_mt.assert_not_called()


def test_interactive_marian_reuses_model(monkeypatch, capsys):
    model = Mock()
    model.translate.side_effect = ("Bonjour.", "Au revoir.")
    load = Mock(return_value=model)
    monkeypatch.setattr(cli, "MarianTranslator", load)
    run_cli(monkeypatch, ["-i", "--mt-only"], ["Hello.", "Goodbye.", ":exit"])
    load.assert_called_once()
    assert model.translate.call_count == 2
    assert "Final French: Au revoir." in capsys.readouterr().out

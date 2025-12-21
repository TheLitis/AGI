import ast
import inspect
import textwrap

from trainer import Trainer


def _last_loss_assignment(src: str) -> ast.Assign:
    tree = ast.parse(src)
    func = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    assert func is not None, "expected a FunctionDef in parsed source"

    loss_assigns = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "loss":
                loss_assigns.append(node)
                break
    assert loss_assigns, "expected at least one 'loss = ...' assignment"
    return loss_assigns[-1]


def test_self_model_loss_uses_only_loss_weights():
    src = textwrap.dedent(inspect.getsource(Trainer._train_self_model_from_buffer))
    loss_assign = _last_loss_assignment(src)
    names = {n.id for n in ast.walk(loss_assign.value) if isinstance(n, ast.Name)}

    assert {"lw_surv", "lw_food", "lw_dmg", "lw_move", "lw_unc", "lw_ret"}.issubset(names)
    assert not ({"pref_surv", "pref_food", "pref_danger", "pref_move"} & names)


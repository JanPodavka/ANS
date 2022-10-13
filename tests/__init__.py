import ast
import inspect
from typing import Any, Callable, Optional, Union
import unittest


class ANSTestCase(unittest.TestCase):

    def __init__(self, methodName: str = '', **params: Any):
        super().__init__(methodName=methodName)
        self.params = params

    @classmethod
    def eval(
            cls,
            tests: Optional[str] = None,
            verbosity: int = 2,
            **params: dict[str, Any]
    ) -> unittest.TestResult:
        class _TestCaseClass(cls):
            def __init__(self, methodName: str = ''):
                super().__init__(methodName=methodName, **params)
        if tests is not None:
            suite = unittest.TestSuite()
            suite.addTests([_TestCaseClass(methodName=t) for t in tests.split(',')])
        else:
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(_TestCaseClass)
        runner = unittest.TextTestRunner(verbosity=verbosity)
        return runner.run(suite)

    @staticmethod
    def _ast_node_name(node: ast.AST) -> str:
        if isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Name):
            return node.id
        else:
            # TODO: probably shouldn't happen
            raise ValueError("This shouldn't happen and probably is a bug in testing")

    @staticmethod
    def inspect_function_dependencies(func: Callable) -> list[str]:
        return [
            ANSTestCase._ast_node_name(node.func)
            for node in ast.walk(ast.parse(inspect.getsource(func)))
            if isinstance(node, ast.Call)
        ]

    def assertNotCalling(self, func: Callable, names: list[str]) -> None:
        for node in ast.walk(ast.parse(inspect.getsource(func))):
            if isinstance(node, ast.Call):  # check not directly calling forbidden name
                call_name = self._ast_node_name(node.func)
            elif isinstance(node, ast.Assign):  # check not assigning alias to forbidden name
                if isinstance(node.value, (ast.Attribute, ast.Name)):
                    call_name = self._ast_node_name(node.value)
                else:
                    continue
            else:
                continue
            self.assertNotIn(call_name, names)

    def assertCalling(self, func: Callable, names: list[str]) -> None:
        call_names = self.inspect_function_dependencies(func)
        for name in names:
            self.assertIn(name, call_names, msg=f"Function {func.__qualname__} should call {name}")

    def assertNoLoops(self, func: Callable) -> None:
        if any(
            isinstance(e, (ast.For, ast.While, ast.ListComp, ast.GeneratorExp))
            for e in ast.walk(ast.parse(inspect.getsource(func)))
        ):
            self.fail(msg=f"Manual loops are not allowed inside {func.__qualname__}")

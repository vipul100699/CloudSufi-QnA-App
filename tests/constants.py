"""
Shared test constants — stable UUIDs used as parent_id values in fixtures
and directly referenced in test assertions.

Defined here (not in conftest.py) because conftest.py is a pytest-managed
file not intended for direct module imports.
"""

PARENT_ID_A = "aaaaaaaa-0000-0000-0000-000000000001"
PARENT_ID_B = "bbbbbbbb-0000-0000-0000-000000000002"

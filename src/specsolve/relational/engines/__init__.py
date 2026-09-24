"""Engines: implementations of the contract one level up.

There is no ``Engine`` base class to subclass — the contract is the two types
that cross the boundary, ``program.Program`` going in and ``sinks.Tables``
coming out, and an engine is whatever turns one into the other. One ships
(``polars``).

``tests/test_architecture.py`` reads membership off the path, and the fence
names no engine.
"""

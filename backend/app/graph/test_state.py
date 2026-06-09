from app.graph.state import (
    append_extractions,
    append_validations,
    append_trace_events,
    append_errors,
    append_warnings,
)

# --------------------------------------------------
# Reducer tests
# --------------------------------------------------

assert append_extractions(None, None) == []
assert append_extractions([1], None) == [1]
assert append_extractions(None, [2]) == [2]
assert append_extractions([1], [2]) == [1, 2]

assert append_validations(["a"], ["b"]) == ["a", "b"]

assert append_trace_events(["e1"], ["e2"]) == ["e1", "e2"]

assert append_errors(["err1"], ["err2"]) == ["err1", "err2"]

assert append_warnings(["w1"], ["w2"]) == ["w1", "w2"]

print("✅ All state.py assertions passed.")
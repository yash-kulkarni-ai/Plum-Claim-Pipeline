from app.graph.claim_graph import (
    _build_graph,
    app_graph,
)

# --------------------------------------------------
# Build graph
# --------------------------------------------------

graph = _build_graph()

assert graph is not None

# --------------------------------------------------
# Compiled graph
# --------------------------------------------------

assert app_graph is not None

# --------------------------------------------------
# Compile again
# --------------------------------------------------

compiled = graph.compile()

assert compiled is not None

print("✅ Graph build successful")
print("✅ Graph compile successful")
print("✅ app_graph export exists")
print("✅ All claim_graph assertions passed")
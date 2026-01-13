from fastmcp import FastMCP

mcp = FastMCP("calculator")


def _as_number(x):
    """Accept int/float or numeric string and convert to float."""
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        return float(x.strip())
    raise TypeError("Expected a number (int/float or numeric string)")


@mcp.tool
async def add(a: float, b: float) -> float:
    """Return a + b"""
    return _as_number(a) + _as_number(b)

if __name__ == "__main__":
    mcp.run()

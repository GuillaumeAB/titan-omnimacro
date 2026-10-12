"""
OmniMacro Engine - Web Server Launcher
Starts the interactive quantitative terminal on http://127.0.0.1:8000
"""

import uvicorn


if __name__ == "__main__":
    print("=" * 75)
    print("  LAUNCHING TITAN OMNIMACRO QUANTITATIVE SIMULATION PLATFORM")
    print("  Access the interactive web terminal at: http://127.0.0.1:8000")
    print("=" * 75)
    uvicorn.run("web.server:app", host="127.0.0.1", port=8000, reload=False)

"""`python -m faceguard_api` 실행 진입점."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("faceguard_api.app:app", host="0.0.0.0", port=8000)

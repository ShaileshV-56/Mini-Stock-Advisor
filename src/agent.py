from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from tavily import TavilyClient
from src.data import get_stock_data
from src.models import load_forecaster
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.1
)

forecaster = load_forecaster()
tavily = TavilyClient(api_key=TAVILY_API_KEY)

@tool
def get_forecast(ticker: str) -> str:
    """Get 1-week price forecast."""
    data = get_stock_data(ticker)
    pred = forecaster.predict(data["historical"])
    return f"1-week forecast for {ticker}: ${pred:.2f} (current: ${data['fundamentals']['current_price']:.2f})"

@tool
def get_news_sentiment(ticker: str) -> str:
    """Get recent news headlines."""
    results = tavily.search(f"{ticker} stock news last 7 days", max_results=3)
    headlines = [r.get("content", "")[:120] for r in results.get("results", [])]
    return f"Recent headlines: {' | '.join(headlines)}"

@tool
def get_fundamentals(ticker: str) -> str:
    """Get key financial metrics."""
    data = get_stock_data(ticker)
    funds = data["fundamentals"]
    return (
        f"P/E: {funds['pe_ratio']}, "
        f"Dividend: {funds['dividend_yield']:.1f}%, "
        f"1Y Change: {funds['price_change_1y']:.1f}%"
    )

tools = [get_forecast, get_news_sentiment, get_fundamentals]
agent = create_react_agent(llm, tools)

def get_stock_advice(ticker: str) -> str:
    result = agent.invoke({
        "messages": [("user", f"Analyze {ticker}. Use all tools. Give a clear Buy/Hold/Sell with reasoning.")]
    })
    return result["messages"][-1].content
# Wersja 5 - Pełny, działający kod produkcyjny z obsługą Gemini 3.5 i czystym formatowaniem
import os
import numpy as np
import streamlit as st
import asyncio
import nest_asyncio

# CRITICAL FIX: Naprawia błąd "There is no current event loop" w Streamlit Cloud
nest_asyncio.apply()

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# ==========================================
# 1. FUNKCJE MATEMATYCZNE (Mózg analityczny)
# ==========================================
def calculate_kelly_math(win_rate: float, avg_win: float, avg_loss: float) -> dict:
    R = avg_win / avg_loss
    full_kelly = win_rate - ((1 - win_rate) / R)
    if full_kelly < 0:
        return {"error": "Strategia ma ujemną wartość oczekiwaną (EV). Kelly wynosi ujemnie, nie inwestuj."}
    half_kelly = full_kelly / 2
    return {
        "full_kelly_pct": round(full_kelly * 100, 2),
        "half_kelly_pct": round(half_kelly * 100, 2),
        "risk_reward_ratio": round(R, 2)
    }

def monte_carlo_math(win_rate: float, avg_win: float, avg_loss: float, kelly_fraction_used: float = 0.5) -> dict:
    R = avg_win / avg_loss
    position_size = (win_rate - ((1 - win_rate) / R)) * kelly_fraction_used
    if position_size <= 0:
        return {"error": "Ujemne EV lub błędne dane, symulacja przerwana."}

    num_simulations = 1000
    num_trades = 100
    max_drawdowns = []
    final_capitals = []
    
    for _ in range(num_simulations):
        capital = 10000
        peak = capital
        max_dd = 0
        for _ in range(num_trades):
            if np.random.rand() < win_rate:
                profit_loss = capital * position_size * avg_win
            else:
                profit_loss = -(capital * position_size * avg_loss)
            
            capital += profit_loss
            if capital <= 0: 
                capital = 0
            
            peak = max(peak, capital)
            dd = (peak - capital) / peak
            max_dd = max(max_dd, dd)
            
            if capital == 0: 
                break
                
        final_capitals.append(capital)
        max_drawdowns.append(max_dd)

    ruin_prob = (len([c for c in final_capitals if c == 0]) / num_simulations) * 100
    valid_finals = [c for c in final_capitals if c > 0]
    dd_80_percentile = round(np.percentile(max_drawdowns, 80) * 100, 2)
    
    return {
        "szansa_na_bankructwo_%": round(ruin_prob, 2),
        "oczekiwany_kapital_koncowy_$": round(np.median(valid_finals), 2) if valid_finals else 0,
        "max_drawdown_w_80%_scenariuszy_%": dd_80_percentile,
        "uzyty_rozmiar_pozycji_kelly": f"{kelly_fraction_used} (Half Kelly)"
    }

# ==========================================
# 2. NARZĘDZIA DLA AGENTA (Z opisami Pydantic dla AI)
# ==========================================
class KellyInput(BaseModel):
    win_rate: float = Field(description="Prawdopodobieństwo wygranej jako ułamek od 0 do 1 (np. 0.55)")
    avg_win: float = Field(description="Średni zysk jako wielokrotność ryzyka (np. 2.0)")
    avg_loss: float = Field(description="Średnia strata jako wielokrotność ryzyka (zwykle 1.0)")

class MonteCarloInput(BaseModel):
    win_rate: float = Field(description="Prawdopodobieństwo wygranej (identyczne jak w kelly_calculator)")
    avg_win: float = Field(description="Średni zysk (identyczny jak w kelly_calculator)")
    avg_loss: float = Field(description="Średnia strata (identyczna jak w kelly_calculator)")

kelly_calculator = StructuredTool.from_function(
    func=calculate_kelly_math,
    name="kelly_calculator",
    description="Oblicza optymalny rozmiar pozycji (Kelly Criterion).",
    args_schema=KellyInput
)

monte_carlo_simulation = StructuredTool.from_function(
    func=monte_carlo_math,
    name="monte_carlo_simulation",
    description="Przeprowadza symulację Monte Carlo i sprawdza ryzyko wariantowe.",
    args_schema=MonteCarloInput
)

tools = [kelly_calculator, monte_carlo_simulation]

# ==========================================
# 3. PERSONA AGENTA (Instrukcje zachowania)
# ==========================================
SYSTEM_PROMPT = """
Jesteś Szefem Zarządzania Ryzykiem (Chief Risk Officer) w firmie inwestycyjnej. 
Twoim celem nie jest tylko podawanie matematycznych wyników, ale ochrona kapitału i psychiki inwestora.

ZASADY DZIAŁANIA (WYKONUJ KROK PO KROKU):
1. ZAWSZE najpierw wywołaj narzędzie `kelly_calculator`, by poznać rozmiar pozycji.
2. ZAWSZE potem wywołaj narzędzie `monte_carlo_simulation`, by zbadać ryzyko wariantowe. Użyj tych samych argumentów co w kroku 1 oraz kelly_fraction_used=0.5.
3. Kiedy otrzymasz wyniki obu narzędzi, MUSISZ przetłumaczyć suche liczby na język emocji i doświadczenia inwestora.

STYLE ODPOWIEDZI:
- Zawsze zaczynaj od podania kluczowych wyliczeń (Half Kelly, Risk/Reward).
- Następnie przechodź do "Sekcji Ostrzeżeń" – opisz, co oznacza Max Drawdown w praktyce (np. "z 10 000 zł zrobi się 6 000 zł"). Używaj słów takich jak: "UWAGA", "nie wytrzymasz psychologicznie", "to bolesna droga".
- Porównaj suchy zysk z bólem potencjalnej straty (np. "Zarobisz X, ale po drodze konto spadnie o Y... Czy będziesz spał po nocach?").
- Zakończ konkretną rekomendacją: czy użyć Half Kelly, czy może Quarter Kelly (0.25), lub czy porzucić strategię.
"""

# ==========================================
# 4. KONFIGURACJA STREAMLIT (Interfejs Użytkownika)
# ==========================================
st.set_page_config(page_title="Agent Ryzyka Kelly", page_icon="🛡️", layout="wide")

with st.sidebar:
    st.header("🔑 Konfiguracja API")
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("Klucz API wczytany automatycznie z chmury! 🚀")
    else:
        api_key = st.text_input("Wklej swój klucz Google Gemini API:", type="password")
        st.markdown("[Kliknij tutaj, aby zdobyć darmowy klucz](https://google.com)")
        
        if not api_key:
            st.warning("Aby agent działał, musisz wkleić klucz API.")
        else:
            st.success("Klucz API podany. Agent gotowy do pracy!")

@st.cache_resource
def init_agent(_api_key):
    os.environ["GOOGLE_API_KEY"] = _api_key
    # Nowoczesny model z pełną obsługą wymaganych podpisów myślowych dla narzędzi
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
    
    # Inicjalizacja stabilnej architektury agenta poprzez jawne przekazanie promptu systemowego
    agent_executor = create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)
    return agent_executor

st.title("🛡️ Agent Zarządzania Ryzykiem")
st.markdown("Wprowadź parametry swojej strategii inwestycyjnej (akcje/opcje). Agent obliczy Kryterium Kelly'ego oraz przeprowadzi analizę Monte Carlo.")

if api_key:
    agent_executor = init_agent(api_key)
    
    with st.form("risk_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            win_rate_input = st.number_input("Win Rate (np. 0.55 dla 55%)", min_value=0.01, max_value=0.99, value=0.55, step=0.01)
        with col2:
            avg_win_input = st.number_input("Średni Zysk (wielokrotność ryzyka, np. 2.0)", min_value=0.1, value=2.0, step=0.1)
        with col3:
            avg_loss_input = st.number_input("Średnia Strata (wielokrotność ryzyka, np. 1.0)", min_value=0.1, value=1.0, step=0.1)
            
        submit_button = st.form_submit_button("Uruchom Analizę Agenta")
        
    if submit_button:
        with st.spinner("Agent analizuje ryzyko i uruchamia symulacje..."):
            try:
                user_query = f"Zanalizuj strategię: win_rate={win_rate_input}, avg_win={avg_win_input}, avg_loss={avg_loss_input}."
                
                # Wywołanie agenta za pomocą nowego interfejsu opartego o historię wiadomości
                response = agent_executor.invoke({"messages": [("user", user_query)]})
                
                # Pobranie surowej odpowiedzi z ostatniego kroku
                raw_output = response["messages"][-1].content
                
                # Oczyszczanie struktury danych - ekstrakt właściwego tekstu Markdown, odrzucenie metadanych JSON
                if isinstance(raw_output, list) and len(raw_output) > 0:
                    final_text = raw_output[0].get('text', str(raw_output))
                elif isinstance(raw_output, dict):
                    final_text = raw_output.get('text', str(raw_output))
                else:
                    final_text = str(raw_output)
                
                # Renderowanie pięknego raportu w Streamlit
                st.markdown("---")
                st.subheader("🛡️ Oficjalny Raport Szefa Zarządzania Ryzykiem (CRO)")
                st.markdown(final_text)
                
            except Exception as e:
                st.error(f"Wystąpił błąd podczas analizy: {str(e)}")



# Wersja 3 - naprawa zaleznosci
import os
import numpy as np
import streamlit as st
import asyncio
import nest_asyncio

# CRITICAL FIX: Naprawia błąd "There is no current event loop" w Streamlit Cloud
nest_asyncio.apply()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# ==========================================
# 1. FUNKCJE MATEMATYCZNE (Mózg analityczny)
# ==========================================
def calculate_kelly_math(win_rate: float, avg_win: float, avg_loss: float) -> dict:
    R = avg_win / avg_loss
    full_kelly = win_rate - ((1 - win_rate) / R)
    if full_kelly < 0:
        return {"error": "Strategia ma ujemne wartość oczekiwaną (EV). Kelly wynosi ujemnie, nie inwestuj."}
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
# 2. NARZĘDZIA DLA AGENTA (Z bardzo dokładnymi opisami dla AI!)
# ==========================================
from langchain_core.tools import tool

@tool
def kelly_calculator(win_rate: float, avg_win: float, avg_loss: float) -> dict:
    """Użyj tego narzędzia jako PIERWSZEGO, aby obliczyć optymalny rozmiar pozycji (Kelly Criterion). 
    Argumenty to:
    - win_rate: Prawdopodobieństwo wygranej jako ułamek od 0 do 1 (np. 0.55 dla 55%).
    - avg_win: Średni zysk wyrażony jako wielokrotność ryzyka (np. 2.0).
    - avg_loss: Średnia strata wyrażona jako wielokrotność ryzyka (zawsze 1.0 w standardowym tradingu).
    """
    return calculate_kelly_math(win_rate, avg_win, avg_loss)

@tool
def monte_carlo_simulation(win_rate: float, avg_win: float, avg_loss: float) -> dict:
    """Użyj tego narzędzia jako DRUGIEGO, zaraz po kelly_calculator. 
    Służy do przeprowadzenia symulacji Monte Carlo i zbadania ryzyka wariantowego (drawdown, bankructwo).
    Zawsze używaj Half Kelly (0.5).
    Argumenty win_rate, avg_win, avg_loss muszą być IDENTYCZNE jak te podane do kelly_calculator.
    """
    # Wymuszamy Half Kelly (0.5) wewnątrz funkcji, żeby agent nie musiał się nad tym zastanawiać
    return monte_carlo_math(win_rate, avg_win, avg_loss, kelly_fraction_used=0.5)

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

# Pobieranie klucza API - najpierw z chmury (Secrets), potem z paska bocznego
with st.sidebar:
    st.header("🔑 Konfiguracja API")
    # Sprawdzamy, czy klucz jest w bezpiecznym sejfie Streamlit Cloud
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("Klucz API wczytany automatycznie z chmury! 🚀")
    else:
        # Jeśli nie ma w chmurze (np. odpalasz u siebie na laptopie), pytamy
        api_key = st.text_input("Wklej swój klucz Google Gemini API:", type="password")
        st.markdown("[Kliknij tutaj, aby zdobyć darmowy klucz](https://aistudio.google.com/app/apikey)")
        
        if not api_key:
            st.warning("Aby agent działał, musisz wkleić klucz API.")
        else:
            st.success("Klucz API podany. Agent gotowy do pracy!")

# Funkcja cachująca - agent tworzy się tylko RAZ, a nie przy każdym kliknięciu!
@st.cache_resource
def init_agent(_api_key):
    os.environ["GOOGLE_API_KEY"] = _api_key
    llm = ChatGoogleGenerativeAI(model="models/gemini-pro", temperature=0)
#    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    return agent_executor

st.title("🛡️ Agent Zarządzania Ryzykiem")
st.markdown("Wprowadź parametry swojej strategii inwestycyjnej (akcje/opcje). Agent obliczy Kryterium Kelly'ego, przejdzie symulację Monte Carlo 1000 wariantów przyszłości i oceni, czy **psychicznie przeżyjesz** tę strategię.")

# Formularz z polami
with st.form("parametry_strategii"):
    col1, col2, col3 = st.columns(3)
    with col1:
        win_rate = st.number_input("Win Rate (%)", min_value=1.0, max_value=99.0, value=50.0, step=5.0, help="Jak często wygrywasz? (np. 50 dla 50%)")
    with col2:
        avg_win = st.number_input("Średni Zysk (R)", min_value=0.1, value=2.0, step=0.1, help="Ile dolarów zyskujesz na 1 dolar ryzyka? (Risk/Reward)")
    with col3:
        avg_loss = st.number_input("Średnia Strata", min_value=0.1, value=1.0, step=0.1, help="Zwykle 1.0 (tracisz to, co zaryzykowałeś)")
    
    submitted = st.form_submit_button("🧠 Przeanalizuj strategię")

# Logika po kliknięciu przycisku
if submitted:
    if not api_key:
        st.error("Zapomniałeś podać klucza API w menu po lewej stronie!")
    else:
        agent_executor = init_agent(api_key)
        
        win_rate_decimal = win_rate / 100  # Konwersja z np. 50 na 0.50 dla matematyki
        
        with st.spinner('Agent oblicza Kelly\'ego i symuluje 1000 równoległych wszechświatów... (To potrwa ok. 10-20 sekund)'):
            pytanie = (
                f"Chcę przeanalizować strategię. Mój Win Rate to {win_rate_decimal} "
                f"(czyli {win_rate}%), mój średni zysk (avg_win) to {avg_win}, "
                f"a moja średnia strata (avg_loss) to {avg_loss}. "
                f"Co o tym myślisz i jakiego rozmiaru pozycji mam użyć?"
            )
            
            try:
                odpowiedz = agent_executor.invoke({"input": pytanie})
                
                st.markdown("---")
                st.subheader("📊 Raport Agenta:")
                # Wyświetlenie odpowiedzi, zachowując formatowanie tekstu od AI (pogrubienia itp.)
                st.markdown(odpowiedz["output"])
                
            except Exception as e:
                st.error(f"Wystąpił błąd. Sprawdź czy klucz API jest poprawny. Błąd: {e}")

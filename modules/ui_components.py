import streamlit as st

try:
    from streamlit_option_menu import option_menu
except Exception:  # pragma: no cover
    option_menu = None


def apply_theme():
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b1220;
            --panel: #111b2d;
            --panel-strong: #16263f;
            --text: #eaf2ff;
            --muted: #9cb6d9;
            --accent: #5ec7ff;
            --accent-strong: #7cf0c6;
            --warning: #fecc66;
            --danger: #ff7a7a;
        }
        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #060d18 0%, #0d1728 100%);
            color: var(--text);
        }
        .stApp {
            background: transparent;
        }
        .stSidebar {
            background: rgba(10, 18, 30, 0.96);
        }
        .metric-card {
            background: linear-gradient(135deg, rgba(30, 51, 82, 0.9), rgba(10, 20, 35, 0.9));
            border: 1px solid rgba(94, 199, 255, 0.22);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            min-height: 120px;
            box-shadow: 0 8px 18px rgba(10, 20, 35, 0.18);
            margin-bottom: 1rem;
        }
        .metric-title {
            color: var(--muted);
            font-size: 0.8rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .metric-value {
            font-size: clamp(1.3rem, 2vw, 2rem);
            font-weight: 700;
            margin-top: 0.5rem;
            color: var(--text);
        }
        .metric-hint {
            margin-top: 0.8rem;
            font-size: 0.8rem;
            color: var(--accent-strong);
        }
        .stDataFrame, .stTable {
            background: rgba(14, 24, 37, 0.8);
        }
        h1, h2, h3 {
            letter-spacing: -0.02em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_nav(pages, default_index=0):
    if option_menu is not None:
        return option_menu(
            "Whisper AV Studio",
            pages,
            default_index=default_index,
            menu_icon="cast",
            icons=["house", "upload", "mic", "translate", "volume-up", "graph-up-arrow", "clock-history", "info-circle"],
            styles={
                "container": {"padding": "0!important", "background-color": "#0b1220"},
                "icon": {"color": "#5ec7ff", "font-size": "16px"},
                "nav-link": {"font-size": "14px", "text-align": "left", "margin": "0px", "color": "#d9ecff"},
                "nav-link-selected": {"background-color": "#1c68a6"},
            },
        )
    return st.sidebar.radio("Navigation", pages, index=default_index)

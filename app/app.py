import streamlit as st

st.write('Welcome to Euroguessr')

def main():
    # Create a sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose an option", ["Home", "Models", "Results"])

if __name__ == "__main__":
    main()

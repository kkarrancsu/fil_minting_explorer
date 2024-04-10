import streamlit as st

st.set_page_config(
    page_title="Introduction",
    page_icon="👋",
)

st.markdown("[![CryptoEconLab](./app/static/cover.png)](https://cryptoeconlab.io)")

st.sidebar.success("Select a Page above.")

st.markdown(
    """
    ### Filecoin Minting Explorer
    This web-app implements a digital twin of the Filecoin Economy. It can be used to forecast components underlying Filecoin's circulating supply (i.e., minting, vesting, locking, and burning), based on a set of parameters that encode storage provider behavior.
    The model is best suited to test a hypothesis about how a change in storage provider behavior will impact the main drivers of circulating supply. It can also be used to test changes in certain economic parameters of Filecoin.
    To learn more about the model assumptions and design, please refer to:
     1. Our [GitHub repository](https://github.com/celtd/mechafil-jax).
     2. Section 3 of our paper on [Agent Based Modeling of the Filecoin Economy](https://arxiv.org/pdf/2307.15200.pdf), published at Chainscience 2023.
    
    ### How to use this app
    
    **👈 Select "Minting Exploration" from the sidebar** to get started. 
    
    In that page, you will see two slider bars allow you to configure the onboarding behavior that you want to test. You can change these parameters to see how they impact the main drivers of circulating supply.
    The parameters are:

     - **RB Onboarding Rate** - This is the amount of raw-byte power that is onboarded every day onto the network, in units of PiB/day.
     - **Renewal Rate** - This is the percentage of sectors which are expiring that are renewed.
    
    Once these values are configured, click the "Forecast" button. The digital-twin runs in the background, taking into account both historical data pulled directly from the Filecoin blockchain and the mechanistic laws defining the various aspects of Filecoin's circulating supply to forecast 
    these statistics until 2040. The results are displayed in the form of a graph, which you can download as a PNG file by clicking the "Download" button.  Additionally, a table showing
    the forecasted total supply (TS = Minting + Vesting - Burning) is displayed below the graph.  The three columns represent three relevant scenarios: 
    - **Simple** - This scenario assumes that only simple minting occurs going forward.
    - **Configured** - This scenario uses the configured onboarding and renewal rates.
    - **Max-Historical** - This scenario assumes that the historical maximum renewal and onboarding rates are maintained.

    ### Want to learn more?

    - Check out [CryptoEconLab](https://cryptoeconlab.io)

    - Engage with us on [X](https://x.com/cryptoeconlab)

    - Read more of our research on [Medium](https://medium.com/cryptoeconlab) and [HackMD](https://hackmd.io/@cryptoecon/almanac/)

    ### Disclaimer
    CryptoEconLab designed this application for informational purposes only. CryptoEconLab does not provide legal, tax, financial or investment advice. No party should act in reliance upon, or with the expectation of, any such advice.
"""
)
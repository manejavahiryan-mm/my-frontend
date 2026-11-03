
import streamlit as st
import numpy as np
import pandas as pd

st.title("Gaussian Thompson Sampling")
st.write("Continuous Gaussian rewards with known noise standard deviation.")

def simulate(true_means, sigma, rounds, seed):
    rng = np.random.default_rng(seed)

    # Normal(0,25) prior for each alternative
    posterior_means = np.zeros(3)
    posterior_vars = np.full(3, 25.0)
    counts = np.zeros(3, dtype=int)
    total_reward = 0.0

    for _ in range(rounds):
        # Sample a possible mean from every posterior
        samples = rng.normal(
            posterior_means, np.sqrt(posterior_vars)
        )

        # Choose the highest sampled mean
        chosen = int(np.argmax(samples))

        # Generate continuous Gaussian reward
        reward = rng.normal(true_means[chosen], sigma)
        counts[chosen] += 1
        total_reward += reward

        # Update only the selected posterior
        old_var = posterior_vars[chosen]
        new_var = 1 / (1 / old_var + 1 / sigma**2)

        posterior_means[chosen] = new_var * (
            posterior_means[chosen] / old_var
            + reward / sigma**2
        )
        posterior_vars[chosen] = new_var

    results = pd.DataFrame({
        "Alternative": [1, 2, 3],
        "True Mean": true_means,
        "Posterior Estimated Mean": posterior_means,
        "Selection Count": counts
    })

    recommendation = int(np.argmax(posterior_means)) + 1

    return results, total_reward, recommendation


st.subheader("Experiment Settings")

mu1 = st.number_input("True Mean 1", value=1.0)
mu2 = st.number_input("True Mean 2", value=2.0)
mu3 = st.number_input("True Mean 3", value=3.0)

sigma = st.number_input(
    "Known Noise Standard Deviation",
    min_value=0.01,
    value=1.0
)

rounds = st.number_input(
    "Number of Rounds",
    min_value=1,
    value=100,
    step=1
)

seed = st.number_input("Random Seed", value=42, step=1)

if st.button("Run Gaussian Thompson Sampling"):
    results, total_reward, recommendation = simulate(
        [mu1, mu2, mu3],
        sigma,
        int(rounds),
        int(seed)
    )

    st.subheader("Results")
    st.dataframe(results, use_container_width=True)

    st.metric("Total Observed Reward", f"{total_reward:.2f}")

    st.success(
        f"Final Recommendation: Alternative {recommendation}"
    )

    st.write(
        "Total selections:",
        int(results["Selection Count"].sum())
    )

    st.subheader("Selection Counts")
    st.bar_chart(
        results.set_index("Alternative")["Selection Count"]
    )

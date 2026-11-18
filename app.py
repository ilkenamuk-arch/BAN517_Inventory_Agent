#!/usr/bin/env python3
"""
Supply Chain Analytics: Inventory Decision Agent Application
============================================================
Dual-mode Application (CLI & Streamlit):
1. Interactive CLI: Rich ANSI-formatted terminal decision support system with preset benchmarks.
2. Single-query CLI: `python app.py --query "..."`
3. Benchmark suite: `python app.py --demo`
4. Web Dashboard: `streamlit run app.py` (auto-detected when run via Streamlit)

Supports 5 Inventory Decision Models:
1. Economic Order Quantity (EOQ)
2. Quantity Discount Model
3. Safety Stock and Reorder Point (Continuous Review ROP)
4. Fixed-Period Review Model (Order-Up-To Level)
5. Newsvendor Model (Single-Period Critical Fractile)
"""

import argparse
import os
import sys
from typing import List, Optional

from inventory_agent import InventoryDecisionAgent, ModelType, AgentDecision


# ---------------------------------------------------------------------------
# Terminal Styling Utilities (ANSI Colors & Formatting)
# ---------------------------------------------------------------------------
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def print_banner():
    print(f"{Colors.BOLD}{Colors.CYAN}")
    print("=" * 78)
    print("      SUPPLY CHAIN ANALYTICS: INVENTORY DECISION AGENT")
    print("   Deterministic Multi-Model Operations Management Assistant")
    print("=" * 78)
    print(f"{Colors.END}")
    print(f"{Colors.YELLOW}Architecture Pipeline:{Colors.END}")
    print(f"Customer Question -> Identify Model -> Extract Inputs -> Select Tool -> Calculate -> Verify -> Explain\n")


def print_decision_report(decision: AgentDecision):
    """Render a structured, color-coded report in the terminal."""
    print("\n" + "=" * 78)
    print(f"{Colors.BOLD}{Colors.HEADER}--- AGENT PIPELINE EXECUTION TRACE ---{Colors.END}")
    print("=" * 78)

    for stage in decision.pipeline_trace:
        status_color = Colors.GREEN if stage.status in ["SUCCESS", "VERIFIED"] else Colors.RED
        print(f"{Colors.BOLD}[{stage.stage_name}]{Colors.END} Status: {status_color}{stage.status}{Colors.END}")
        print(f"  {Colors.CYAN}--> {stage.summary}{Colors.END}")

    print("\n" + "=" * 78)
    print(f"{Colors.BOLD}{Colors.GREEN}--- SUPPLY CHAIN DECISION & MANAGERIAL REPORT ---{Colors.END}")
    print("=" * 78)
    print(decision.explanation)
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# Pre-loaded Benchmark Scenarios
# ---------------------------------------------------------------------------
BENCHMARK_SCENARIOS = [
    {
        "model": "1. Economic Order Quantity (EOQ)",
        "question": (
            "Our annual demand is 10,000 units, order cost is $50 per order, and holding cost is $2 per unit per year. "
            "Our purchase price is $10 per unit. What is our optimal order size and how often should we order?"
        ),
    },
    {
        "model": "2. Quantity Discount Model",
        "question": (
            "Annual demand is 5,000 units, order cost is $49 per order, and our annual holding cost rate is 20%. "
            "Our supplier offers the following volume discount schedule: "
            "$5.00 for orders under 1000, $4.80 for 1000 to 1999 units, and $4.75 for 2000 or more units. "
            "What order quantity minimizes our total annual procurement and inventory cost?"
        ),
    },
    {
        "model": "3. Safety Stock & Reorder Point (Continuous Review)",
        "question": (
            "Daily demand averages 50 units with standard deviation 10 units. Lead time is 9 days. "
            "What safety stock and reorder point do we need to maintain a 95% cycle service level?"
        ),
    },
    {
        "model": "4. Fixed-Period Review Model (Order-Up-To Level)",
        "question": (
            "We review store inventory every 14 days with a replenishment lead time of 7 days. "
            "Average daily demand is 20 units with a standard deviation of 5 units. Target service level is 98%. "
            "We currently have 150 units on-hand, 50 units on-order, and 10 backorders. "
            "What is our order-up-to level and how many units should we order today?"
        ),
    },
    {
        "model": "5. Newsvendor Model (Single-Period Perishable/Seasonal)",
        "question": (
            "We sell a seasonal winter jacket for $100, which costs us $40 to purchase wholesale. "
            "Any unsold jacket at the end of winter is salvaged for $20. "
            "Customer demand is normally distributed with mean 1,000 units and standard deviation 200 units. "
            "How many jackets should we order for the upcoming season?"
        ),
    },
]


def run_demo(agent: InventoryDecisionAgent):
    """Run all 5 benchmark scenarios sequentially."""
    print_banner()
    print(f"{Colors.BOLD}{Colors.YELLOW}RUNNING FULL INVENTORY BENCHMARK SUITE (5 MODELS){Colors.END}\n")

    for i, scenario in enumerate(BENCHMARK_SCENARIOS, 1):
        print(f"\n{Colors.BOLD}{Colors.BLUE}>>> BENCHMARK SCENARIO {i}: {scenario['model']}{Colors.END}")
        print(f"{Colors.YELLOW}Question:{Colors.END} \"{scenario['question']}\"\n")
        decision = agent.process_question(scenario["question"])
        print_decision_report(decision)
        print("-" * 78)


def interactive_cli(agent: InventoryDecisionAgent):
    """Interactive shell prompt for customer questions."""
    print_banner()
    print("Welcome! Type your natural-language inventory question below.")
    print("Or enter a number (1-5) to run a textbook benchmark scenario.")
    print("Type 'demo' to run all 5 models, or 'exit' / 'quit' to end.\n")

    print(f"{Colors.BOLD}Benchmark Options:{Colors.END}")
    for idx, sc in enumerate(BENCHMARK_SCENARIOS, 1):
        print(f"  [{idx}] {sc['model']}")
    print("")

    while True:
        try:
            user_input = input(f"{Colors.BOLD}{Colors.CYAN}Inventory Question > {Colors.END}").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print(f"\n{Colors.GREEN}Thank you for using the Supply Chain Analytics Inventory Decision Agent! Goodbye.{Colors.END}\n")
                break

            if user_input.lower() == "demo":
                run_demo(agent)
                continue

            if user_input in ["1", "2", "3", "4", "5"]:
                choice_idx = int(user_input) - 1
                selected = BENCHMARK_SCENARIOS[choice_idx]
                print(f"\n{Colors.BOLD}{Colors.BLUE}Selected: {selected['model']}{Colors.END}")
                print(f"{Colors.YELLOW}Scenario Question:{Colors.END} {selected['question']}")
                user_input = selected["question"]

            print(f"\n{Colors.YELLOW}Processing question through 6-stage architecture...{Colors.END}")
            decision = agent.process_question(user_input)
            print_decision_report(decision)

        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{Colors.GREEN}Session terminated. Goodbye!{Colors.END}\n")
            break


# ---------------------------------------------------------------------------
# Optional Streamlit Web Dashboard Support
# ---------------------------------------------------------------------------
def run_streamlit_app():
    """Renders an interactive Streamlit UI if launched with `streamlit run app.py`."""
    import streamlit as st  # type: ignore

    st.set_page_config(
        page_title="Supply Chain Analytics Inventory Agent",
        page_icon="📦",
        layout="wide",
    )

    st.title("📦 Supply Chain Analytics: Inventory Decision Agent")
    st.markdown(
        "**Deterministic Operations Management Assistant** powered by a 6-stage agentic workflow:  \n"
        "`Customer Question` → `Identify Model` → `Extract Inputs` → `Select Tool` → `Calculate` → `Verify` → `Explain`"
    )

    agent = InventoryDecisionAgent()

    st.sidebar.header("Benchmark Scenarios")
    preset_choice = st.sidebar.selectbox(
        "Load a preset benchmark:",
        ["(Custom Question)"] + [f"{s['model']}" for s in BENCHMARK_SCENARIOS]
    )

    default_text = ""
    if preset_choice != "(Custom Question)":
        for s in BENCHMARK_SCENARIOS:
            if s["model"] in preset_choice:
                default_text = s["question"]
                break

    question_input = st.text_area(
        "Enter your natural language inventory question:",
        value=default_text,
        height=120,
        placeholder="e.g. Our annual demand is 10,000 units, order cost is $50, and holding cost is $2 per unit. What is our EOQ?",
    )

    if st.button("Solve Inventory Problem", type="primary"):
        if not question_input.strip():
            st.warning("Please enter or select a question first.")
        else:
            with st.spinner("Analyzing question through decision pipeline..."):
                decision = agent.process_question(question_input)

            st.subheader(f"Pipeline Result: {decision.model_type.value}")

            # Execution trace columns
            trace_cols = st.columns(len(decision.pipeline_trace))
            for i, stage in enumerate(decision.pipeline_trace):
                with trace_cols[i]:
                    st.metric(
                        label=stage.stage_name,
                        value=stage.status,
                    )
                    st.caption(stage.summary)

            st.divider()
            st.markdown(decision.explanation)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main():
    # Detect if invoked inside Streamlit
    if "streamlit" in sys.modules or os.environ.get("STREAMLIT_SERVER_PORT") or (len(sys.argv) > 0 and "streamlit" in sys.argv[0]):
        run_streamlit_app()
        return

    parser = argparse.ArgumentParser(
        description="Supply Chain Analytics Inventory Decision Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single natural-language inventory question to process.",
    )
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Run the 5-model textbook benchmark demonstration.",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Launch interactive terminal mode (default).",
    )

    args = parser.parse_args()
    agent = InventoryDecisionAgent()

    if args.demo:
        run_demo(agent)
    elif args.query:
        print_banner()
        print(f"{Colors.YELLOW}Customer Question:{Colors.END} \"{args.query}\"")
        decision = agent.process_question(args.query)
        print_decision_report(decision)
    else:
        interactive_cli(agent)


if __name__ == "__main__":
    main()

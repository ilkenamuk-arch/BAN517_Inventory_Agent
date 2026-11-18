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
from inventory_tools import classify_multicriteria_inventory, ClassificationResult


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
# Optional Streamlit Web Dashboard Support & CSV Data Analytics
# ---------------------------------------------------------------------------
def analyze_inventory_dataframe(df):
    """
    Perform pandas calculations on an inventory dataframe.
    Validates required columns: Product, Category, Units, Unit Cost.
    Returns (missing_columns, results_dict).
    """
    import pandas as pd  # type: ignore

    required_columns = ["Product", "Category", "Units", "Unit Cost"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        return missing_columns, {}

    df_calc = df.copy()
    df_calc["Units"] = pd.to_numeric(df_calc["Units"], errors="coerce").fillna(0)
    df_calc["Unit Cost"] = pd.to_numeric(df_calc["Unit Cost"], errors="coerce").fillna(0.0)

    # 1. Calculate total Units
    total_units = df_calc["Units"].sum()

    # 2. Calculate average Unit Cost
    avg_unit_cost = df_calc["Unit Cost"].mean() if len(df_calc) > 0 else 0.0

    # 3. Identify the Product with the highest Units
    if len(df_calc) > 0:
        highest_idx = df_calc["Units"].idxmax()
        highest_product = str(df_calc.loc[highest_idx, "Product"])
        highest_units_val = df_calc.loc[highest_idx, "Units"]
    else:
        highest_product = "N/A"
        highest_units_val = 0

    # 4. Group data by Category and calculate total Units for each Category
    units_by_category = (
        df_calc.groupby("Category", as_index=False)["Units"]
        .sum()
        .sort_values(by="Units", ascending=False)
    )

    # 5. Create new calculated column: Inventory Value = Units * Unit Cost
    df_calc["Inventory Value"] = df_calc["Units"] * df_calc["Unit Cost"]

    # 6. Calculate total Inventory Value across all products
    total_inventory_value = df_calc["Inventory Value"].sum()

    results = {
        "total_units": total_units,
        "avg_unit_cost": avg_unit_cost,
        "highest_units_product": highest_product,
        "highest_units_val": highest_units_val,
        "units_by_category": units_by_category,
        "total_inventory_value": total_inventory_value,
        "dataframe": df_calc,
    }
    return [], results


def run_streamlit_app():
    """Renders an interactive Streamlit UI if launched with `streamlit run app.py`."""
    import streamlit as st  # type: ignore
    import pandas as pd  # type: ignore

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

    # Sidebar: Benchmark Scenarios
    st.sidebar.header("Benchmark Scenarios")
    preset_choice = st.sidebar.selectbox(
        "Load a preset benchmark:",
        ["(Custom Question)"] + [f"{s['model']}" for s in BENCHMARK_SCENARIOS]
    )

    # Sidebar: Data Upload Section (Last section in sidebar)
    st.sidebar.divider()
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV File",
        type=["csv"],
        help="Upload an inventory CSV file with Product, Category, Units, and Unit Cost columns."
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

    # -----------------------------------------------------------------------
    # CSV Data Processing & Analysis Section (Supports Both CSV Schemas)
    # -----------------------------------------------------------------------
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return

        # Detect CSV schema
        classification_cols = ["Item Number", "Item Name", "UoM", "Sales ($)", "Hits"]
        analysis_cols = ["Product", "Category", "Units", "Unit Cost"]

        df_cols_clean = [str(c).strip() for c in df.columns]
        is_classification_format = all(c in df_cols_clean for c in classification_cols)
        is_analysis_format = all(c in df_cols_clean for c in analysis_cols)

        # -------------------------------------------------------------------
        # Schema 1: Multicriteria Inventory Classification (BAN 517 Methodology)
        # -------------------------------------------------------------------
        if is_classification_format:
            st.divider()
            st.header("🏷️ Multicriteria Inventory Classification (BAN 517 Methodology)")
            st.caption("Pareto Analysis: 65% Monetary Value (Sales) + 35% Transaction Velocity (Hits)")

            # Display uploaded dataframe preview
            st.subheader("Uploaded Project Data Preview")
            st.dataframe(df, use_container_width=True)

            # Display dataset dimensions
            num_rows, num_cols = df.shape
            col_names_formatted = ", ".join(f"`{c}`" for c in df.columns)
            st.markdown(
                f"- **Number of Rows:** {num_rows:,}  \n"
                f"- **Number of Columns:** {num_cols}  \n"
                f"- **Column Names:** {col_names_formatted}"
            )

            try:
                result = classify_multicriteria_inventory(df, sales_weight=0.65, hits_weight=0.35)
            except Exception as e:
                st.error(f"Classification calculation error: {e}")
                return

            # Executive KPI Cards
            st.subheader("Executive Classification Overview")
            kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
            with kpi_c1:
                st.metric("Total Items", f"{result.total_items:,}")
            with kpi_c2:
                st.metric("Total Sales", f"${result.total_sales:,.2f}")
            with kpi_c3:
                st.metric("Total Hits", f"{result.total_hits:,.0f}")
            with kpi_c4:
                pct_a_sales = (result.class_sales['A'] / result.total_sales * 100.0) if result.total_sales > 0 else 0.0
                st.metric("Class A Sales Share", f"{pct_a_sales:.1f}%", help=f"${result.class_sales['A']:,.2f} across {result.class_counts['A']} items")

            # 6. Classification Summary Table (Always includes A, B, C, D)
            st.subheader("Multicriteria Inventory Classification Summary")
            st.markdown(
                "Final class assignment thresholds (applied to unrounded Combined Weighted Score):  \n"
                "**Class A:** [3.25–4.00] | **Class B:** [2.50–3.24] | **Class C:** [1.75–2.49] | **Class D:** Below 1.75"
            )
            summary_display = result.summary_df.copy()
            st.dataframe(
                summary_display.style.format({
                    "Number of Items": "{:,.0f}",
                    "Total Sales ($)": "${:,.2f}",
                    "% of Total Items": "{:,.1f}%",
                    "% of Total Sales": "{:,.1f}%",
                }),
                use_container_width=True
            )

            # 5. Detailed Classified Dataframe with all 10 calculated columns
            st.subheader("Detailed Classified Inventory Dataset")
            st.caption(
                "Includes all 10 calculated attributes: Sales Percentage, Cumulative Sales Percentage, Sales Class, "
                "Hits Percentage, Cumulative Hits Percentage, Hits Class, Sales Score, Hits Score, "
                "Combined Weighted Score (0.65×Sales + 0.35×Hits), and Final Inventory Class."
            )
            classified_display = result.classified_df.copy()
            st.dataframe(
                classified_display.style.format({
                    "Sales ($)": "${:,.2f}",
                    "Sales Percentage": "{:,.2f}%",
                    "Cumulative Sales Percentage": "{:,.2f}%",
                    "Hits": "{:,.0f}",
                    "Hits Percentage": "{:,.2f}%",
                    "Cumulative Hits Percentage": "{:,.2f}%",
                    "Sales Score": "{:.0f}",
                    "Hits Score": "{:.0f}",
                    "Combined Weighted Score": "{:.4f}",
                }),
                use_container_width=True
            )

            # Verification Checks Expander
            with st.expander("🔍 Verification Checks & Methodology Details", expanded=False):
                st.markdown("#### Mathematical & Reconciliation Verification:")
                for chk in result.verification_checks:
                    st.markdown(f"- {chk}")
                st.markdown("#### Methodology & Threshold Rules:")
                for idx, asm in enumerate(result.assumptions, 1):
                    st.markdown(f"{idx}. {asm}")

        # -------------------------------------------------------------------
        # Schema 2: Stock Units & Valuation Analysis (Existing Format)
        # -------------------------------------------------------------------
        elif is_analysis_format:
            st.divider()
            st.header("📊 Inventory Data Analysis")

            # Display the uploaded dataframe
            st.subheader("Uploaded Inventory Data")
            st.dataframe(df, use_container_width=True)

            # Display dataset properties
            num_rows, num_cols = df.shape
            col_names_formatted = ", ".join(f"`{c}`" for c in df.columns)
            st.markdown(
                f"- **Number of Rows:** {num_rows:,}  \n"
                f"- **Number of Columns:** {num_cols}  \n"
                f"- **Column Names:** {col_names_formatted}"
            )

            # Perform data analysis using pandas
            _, analysis_results = analyze_inventory_dataframe(df)

            total_units = analysis_results["total_units"]
            avg_unit_cost = analysis_results["avg_unit_cost"]
            highest_product = analysis_results["highest_units_product"]
            highest_units_val = analysis_results["highest_units_val"]
            units_by_category = analysis_results["units_by_category"]
            total_inventory_value = analysis_results["total_inventory_value"]
            augmented_df = analysis_results["dataframe"]

            # Key Inventory Metrics Display
            st.subheader("Key Inventory Metrics")
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            with kpi_col1:
                st.metric(
                    label="Total Units",
                    value=f"{int(total_units):,}" if total_units == int(total_units) else f"{total_units:,.2f}"
                )
            with kpi_col2:
                st.metric(
                    label="Average Unit Cost",
                    value=f"${avg_unit_cost:,.2f}"
                )
            with kpi_col3:
                st.metric(
                    label="Product with Highest Units",
                    value=f"{highest_product}",
                    help=f"{highest_units_val:,.0f} units in stock"
                )
            with kpi_col4:
                st.metric(
                    label="Total Inventory Value",
                    value=f"${total_inventory_value:,.2f}"
                )

            st.write(f"**Product with Highest Units:** **{highest_product}** ({int(highest_units_val):,} units)")

            # Total Units by Category
            st.subheader("Total Units by Category")
            category_display = units_by_category.copy()
            st.dataframe(
                category_display.style.format({"Units": "{:,.0f}"}),
                use_container_width=True
            )

            # Augmented dataframe with currency formatting
            st.subheader("Inventory Data with Inventory Value")
            display_df = augmented_df.copy()
            st.dataframe(
                display_df.style.format({
                    "Units": "{:,.0f}",
                    "Unit Cost": "${:,.2f}",
                    "Inventory Value": "${:,.2f}",
                }),
                use_container_width=True
            )

        # -------------------------------------------------------------------
        # Unrecognized Schema Error Handling
        # -------------------------------------------------------------------
        else:
            missing_class = [c for c in classification_cols if c not in df_cols_clean]
            missing_anal = [c for c in analysis_cols if c not in df_cols_clean]
            st.error(
                "⚠️ **Unrecognized CSV Schema:** The uploaded file does not match either supported inventory schema:\n\n"
                f"1. **Multicriteria Classification Schema:** Requires `Item Number`, `Item Name`, `UoM`, `Sales ($)`, `Hits`. (Missing: {', '.join(missing_class)})\n"
                f"2. **Inventory Stock Analysis Schema:** Requires `Product`, `Category`, `Units`, `Unit Cost`. (Missing: {', '.join(missing_anal)})\n\n"
                "Please verify your CSV file header names."
            )


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

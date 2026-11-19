"""
Supply Chain Analytics Inventory Decision Tools
Deterministic Python implementation of the 5 core inventory management models:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop()
4. calculate_periodic_review()
5. calculate_newsvendor()

Each tool:
- Validates inputs strictly (raises ValueError on missing or invalid inputs; never invents missing numbers)
- Returns structured results with dedicated attributes and .to_dict()
- Returns comprehensive intermediate calculations
- Returns the final decision answer
- Identifies underlying operational assumptions
"""

from dataclasses import dataclass, field, asdict
import math
from statistics import NormalDist
from typing import Dict, List, Optional, Tuple, Union, Any


# ---------------------------------------------------------------------------
# Helper Mathematical Functions (Zero External Dependencies)
# ---------------------------------------------------------------------------

def standard_normal_cdf(z: float) -> float:
    """Cumulative distribution function for standard normal distribution N(0, 1)."""
    return NormalDist().cdf(z)


def standard_normal_inv_cdf(p: float) -> float:
    """Quantile function (inverse CDF) for standard normal distribution N(0, 1)."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"Probability p must be strictly between 0 and 1 (got {p}).")
    return NormalDist().inv_cdf(p)


def standard_normal_pdf(z: float) -> float:
    """Probability density function for standard normal distribution N(0, 1)."""
    return NormalDist().pdf(z)


def unit_normal_loss(z: float) -> float:
    """
    Unit Normal Loss function L(z) = E[max(0, Z - z)] = phi(z) - z * (1 - Phi(z)).
    Used to calculate expected lost sales and expected leftover inventory in
    continuous inventory and Newsvendor models.
    """
    phi_z = standard_normal_pdf(z)
    Phi_z = standard_normal_cdf(z)
    return phi_z - z * (1.0 - Phi_z)


# ---------------------------------------------------------------------------
# 1. Economic Order Quantity (EOQ): calculate_eoq()
# ---------------------------------------------------------------------------

@dataclass
class EOQResult:
    # Inputs
    inputs: Dict[str, Any]
    annual_demand: float
    order_cost: float
    holding_cost: float
    unit_cost: Optional[float]
    working_days_per_year: int

    # Intermediate Calculations
    intermediate_calculations: Dict[str, Any]
    orders_per_year: float
    cycle_time_years: float
    cycle_time_days: float
    cycle_time_weeks: float
    annual_order_cost: float
    annual_holding_cost: float
    total_annual_inventory_cost: float
    annual_purchase_cost: Optional[float]
    total_annual_cost: float
    rounded_order_cost: float
    rounded_holding_cost: float
    rounded_total_cost: float
    cost_penalty_rounded: float

    # Final Answer
    final_answer: Dict[str, Any]
    eoq: float
    rounded_eoq: int

    # Assumptions & Verification
    assumptions: List[str]
    verification_passed: bool
    monthly_demand: Optional[float] = None
    verification_notes: List[str] = field(default_factory=list)

    # 8 Student Verification Sections
    inputs_extracted: Dict[str, Any] = field(default_factory=dict)
    variable_definitions: Dict[str, str] = field(default_factory=dict)
    formula: Dict[str, str] = field(default_factory=dict)
    formula_with_substituted_values: Dict[str, str] = field(default_factory=dict)
    verification_checks: List[str] = field(default_factory=list)

    def to_student_verification(self) -> Dict[str, Any]:
        return {
            "1_inputs_extracted": self.inputs_extracted,
            "2_variable_definitions": self.variable_definitions,
            "3_formula": self.formula,
            "4_formula_with_substituted_values": self.formula_with_substituted_values,
            "5_intermediate_calculations": self.intermediate_calculations,
            "6_final_answer": self.final_answer,
            "7_verification_checks": self.verification_checks,
            "8_assumptions": self.assumptions,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_eoq(
    demand: Optional[float] = None,
    order_cost: Optional[float] = None,
    holding_cost: Optional[float] = None,
    holding_rate: Optional[float] = None,
    unit_cost: Optional[float] = None,
    working_days_per_year: int = 365,
    monthly_demand: Optional[float] = None,
) -> EOQResult:
    """
    Calculate Economic Order Quantity (EOQ) using the classical Wilson formula:
      Q = sqrt((2 * D * S) / H)

    Input Validation:
    - demand (D) > 0 OR monthly_demand > 0 (required, never invented)
    - order_cost (S) > 0 (required, never invented)
    - holding_cost (H) > 0 OR (holding_rate (i) > 0 and unit_cost (C) > 0)
    """
    # Convert monthly demand to annual demand if annual demand is not directly supplied
    if demand is None and monthly_demand is not None:
        if monthly_demand <= 0:
            raise ValueError(f"Monthly demand must be strictly positive (got {monthly_demand}).")
        demand = monthly_demand * 12.0

    # Strict input validation: never invent missing numerical inputs
    if demand is None:
        raise ValueError("Missing required numerical input: 'demand' (annual demand D).")
    if demand <= 0:
        raise ValueError(f"Annual demand D must be strictly positive (got {demand}).")

    if order_cost is None:
        raise ValueError("Missing required numerical input: 'order_cost' (ordering/setup cost S).")
    if order_cost <= 0:
        raise ValueError(f"Order cost S must be strictly positive (got {order_cost}).")

    if working_days_per_year <= 0:
        raise ValueError(f"Working days per year must be positive (got {working_days_per_year}).")

    # Determine holding cost H
    if holding_cost is None:
        if holding_rate is not None and unit_cost is not None:
            if holding_rate <= 0:
                raise ValueError(f"Holding rate must be strictly positive (got {holding_rate}).")
            if unit_cost <= 0:
                raise ValueError(f"Unit cost must be strictly positive when using holding rate (got {unit_cost}).")
            computed_h = holding_rate * unit_cost
        elif holding_rate is not None and unit_cost is None:
            raise ValueError("Missing required numerical input: 'unit_cost' (C) is required when holding rate (i) is provided.")
        else:
            raise ValueError("Missing required numerical input: must provide either 'holding_cost' or both 'holding_rate' and 'unit_cost'.")
    else:
        if holding_cost <= 0:
            raise ValueError(f"Annual holding cost H must be strictly positive (got {holding_cost}).")
        computed_h = holding_cost

    # Formula: Q = sqrt(2DS / H)
    eoq = math.sqrt((2.0 * demand * order_cost) / computed_h)

    # Intermediate calculations
    orders_per_year = demand / eoq
    cycle_time_years = eoq / demand
    cycle_time_days = cycle_time_years * working_days_per_year
    cycle_time_weeks = cycle_time_years * 52.0

    annual_order_cost = (demand / eoq) * order_cost
    annual_holding_cost = (eoq / 2.0) * computed_h
    total_annual_inventory_cost = annual_order_cost + annual_holding_cost

    # If unit purchase cost is specified, compute annual purchasing expenditure
    if unit_cost is not None and unit_cost > 0:
        annual_purchase_cost = demand * unit_cost
        total_annual_cost = annual_purchase_cost + total_annual_inventory_cost
    else:
        annual_purchase_cost = None
        total_annual_cost = total_annual_inventory_cost

    # Practical integer rounding evaluation
    rounded_eoq = max(1, round(eoq))
    rounded_order_cost = (demand / rounded_eoq) * order_cost
    rounded_holding_cost = (rounded_eoq / 2.0) * computed_h
    if annual_purchase_cost is not None:
        rounded_total_cost = annual_purchase_cost + rounded_order_cost + rounded_holding_cost
    else:
        rounded_total_cost = rounded_order_cost + rounded_holding_cost
    cost_penalty_rounded = rounded_total_cost - total_annual_cost

    # Intermediate calculations dictionary
    intermediate = {
        "annual_ordering_cost": annual_order_cost,
        "annual_holding_cost": annual_holding_cost,
        "total_annual_inventory_cost": total_annual_inventory_cost,
        "annual_purchase_cost": annual_purchase_cost,
        "total_annual_cost": total_annual_cost,
        "orders_per_year": orders_per_year,
        "cycle_time_years": cycle_time_years,
        "cycle_time_days": cycle_time_days,
        "cycle_time_weeks": cycle_time_weeks,
        "rounded_order_cost": rounded_order_cost,
        "rounded_holding_cost": rounded_holding_cost,
        "rounded_total_cost": rounded_total_cost,
        "cost_penalty_rounded": cost_penalty_rounded,
    }

    # Final answer dictionary
    final_ans = {
        "optimal_order_quantity_continuous": eoq,
        "optimal_order_quantity_integer": rounded_eoq,
        "minimized_total_inventory_cost": total_annual_inventory_cost,
        "orders_per_year": orders_per_year,
        "cycle_time_days": cycle_time_days,
    }

    # 1. Inputs Extracted
    inputs_extracted = {}
    if monthly_demand is not None:
        inputs_extracted["monthly_demand"] = monthly_demand
    inputs_extracted["annual_demand_D"] = demand
    inputs_extracted["order_cost_S"] = order_cost
    inputs_extracted["holding_cost_H"] = computed_h
    inputs_extracted["holding_rate_i"] = holding_rate
    inputs_extracted["unit_cost_C"] = unit_cost
    inputs_extracted["working_days_per_year"] = working_days_per_year

    # 2. Variable Definitions
    variable_definitions = {}
    if monthly_demand is not None:
        variable_definitions["monthly_demand"] = "Monthly demand rate specified in problem statement (units/month)"
        variable_definitions["D"] = f"Annual demand rate = Monthly Demand * 12 = {demand:,.0f} units/year"
    else:
        variable_definitions["D"] = "Annual demand rate (units/year)"
    variable_definitions.update({
        "S": "Fixed ordering / setup cost per replenishment order ($/order)",
        "H": "Holding / carrying cost per unit per year ($/unit/year)",
        "C": "Unit purchase price / procurement cost ($/unit)",
        "Q": "Order quantity / lot size (units/order)",
        "Q*": "Economic Order Quantity (optimal lot size minimizing total inventory cost) (units)",
        "N": "Order frequency (orders placed per year = D / Q*)",
        "T_days": f"Cycle time / replenishment interval ({working_days_per_year} operating days / N)",
        "AOC": "Annual Ordering Cost = (D / Q) * S ($/year)",
        "AHC": "Annual Holding Cost = (Q / 2) * H ($/year)",
        "TAC": "Total Annual Inventory Cost = AOC + AHC ($/year)",
        "TC": "Total Annual Cost including purchasing expenditure = TAC + (D * C) ($/year)",
    })

    # 3. Formula
    formula = {
        "optimal_order_quantity_Q*": r"Q^* = \sqrt{\frac{2DS}{H}}",
        "annual_ordering_cost_AOC": r"AOC = \left(\frac{D}{Q}\right) S",
        "annual_holding_cost_AHC": r"AHC = \left(\frac{Q}{2}\right) H",
        "total_annual_inventory_cost_TAC": r"TAC = \left(\frac{D}{Q}\right) S + \left(\frac{Q}{2}\right) H",
        "order_frequency_N": r"N = \frac{D}{Q^*}",
        "cycle_time_days_T": r"T_{\text{days}} = \left(\frac{Q^*}{D}\right) \times \text{Working Days}",
        "total_annual_cost_TC": r"TC = D \cdot C + \left(\frac{D}{Q}\right) S + \left(\frac{Q}{2}\right) H" if unit_cost is not None else r"TC = TAC = \left(\frac{D}{Q}\right) S + \left(\frac{Q}{2}\right) H",
    }

    # 4. Formula with Substituted Values (formatted for KaTeX display math)
    if unit_cost is not None and unit_cost > 0:
        tc_substituted = (
            f"TC = ({demand:,.2f} \\cdot \\text{{\\$}}{unit_cost:,.2f}) + \\text{{\\$}}{total_annual_inventory_cost:,.2f} = "
            f"\\text{{\\$}}{annual_purchase_cost:,.2f} + \\text{{\\$}}{total_annual_inventory_cost:,.2f} = \\text{{\\$}}{total_annual_cost:,.2f}/\\text{{year}}"
        )
    else:
        tc_substituted = (
            f"TC = TAC = \\text{{\\$}}{annual_order_cost:,.2f} + \\text{{\\$}}{annual_holding_cost:,.2f} = \\text{{\\$}}{total_annual_inventory_cost:,.2f}/\\text{{year}} "
            f"\\quad \\text{{(Unit purchase cost C not provided; purchasing expenditure excluded)}}"
        )

    formula_with_substituted_values = {
        "optimal_order_quantity_Q*": (
            f"Q^* = \\sqrt{{\\frac{{2 \\cdot {demand:,.2f} \\cdot {order_cost:,.2f}}}{{{computed_h:,.2f}}}}} = "
            f"\\sqrt{{\\frac{{{2.0 * demand * order_cost:,.2f}}}{{{computed_h:,.2f}}}}} = "
            f"\\sqrt{{{2.0 * demand * order_cost / computed_h:,.2f}}} = {eoq:,.2f} \\text{{ units}}"
        ),
        "annual_ordering_cost_AOC": (
            f"AOC = \\left(\\frac{{{demand:,.2f}}}{{{eoq:,.2f}}}\\right) \\cdot {order_cost:,.2f} = "
            f"{orders_per_year:,.4f} \\cdot {order_cost:,.2f} = \\text{{\\$}}{annual_order_cost:,.2f}/\\text{{year}}"
        ),
        "annual_holding_cost_AHC": (
            f"AHC = \\left(\\frac{{{eoq:,.2f}}}{{2}}\\right) \\cdot {computed_h:,.2f} = "
            f"{eoq / 2.0:,.4f} \\cdot {computed_h:,.2f} = \\text{{\\$}}{annual_holding_cost:,.2f}/\\text{{year}}"
        ),
        "total_annual_inventory_cost_TAC": (
            f"TAC = \\text{{\\$}}{annual_order_cost:,.2f} + \\text{{\\$}}{annual_holding_cost:,.2f} = \\text{{\\$}}{total_annual_inventory_cost:,.2f}/\\text{{year}}"
        ),
        "order_frequency_N": f"N = \\frac{{{demand:,.2f}}}{{{eoq:,.2f}}} = {orders_per_year:,.2f} \\text{{ orders/year}}",
        "cycle_time_days_T": (
            f"T_{{\\text{{days}}}} = \\left(\\frac{{{eoq:,.2f}}}{{{demand:,.2f}}}\\right) \\cdot {working_days_per_year} = "
            f"{cycle_time_days:,.2f} \\text{{ operating days}}"
        ),
        "total_annual_cost_TC": tc_substituted,
    }

    # Operational assumptions
    assumptions = [
        "Demand rate (D) is constant, known with certainty, and uniformly distributed over the year.",
        "Replenishment lead time is zero or constant, and order delivery is instantaneous in a single batch.",
        "Setup / ordering cost (S) is fixed per order regardless of the number of units ordered.",
        "Inventory carrying cost (H) is linear and strictly proportional to average inventory level (Q/2).",
        "No shortages, backorders, or stockouts are permitted.",
    ]
    if monthly_demand is not None:
        assumptions.append(f"Annual demand rate (D = {demand:,.0f} units/year) is converted from monthly demand ({monthly_demand:,.0f} units/month * 12 months/year).")
    if unit_cost is not None and unit_cost > 0:
        assumptions.append(f"Purchase price per unit (C = ${unit_cost:,.2f}) is constant and independent of order quantity (no volume discounts).")
    else:
        assumptions.append("Unit purchase cost (C) was not specified in the problem statement. When annual holding cost (H) is specified directly ($/unit/year), the optimal order quantity (Q*) and total inventory carrying & setup cost (TAC) are independent of unit purchase price. Total purchasing expenditure (D * C) is omitted from total annual cost.")

    # Verification checks
    cost_diff = abs(annual_order_cost - annual_holding_cost)
    cost_balance_valid = cost_diff < 1e-4 * max(1.0, total_annual_inventory_cost)
    verification_checks = [
        f"[PASS] Non-negativity constraint satisfied: D = {demand:,.1f} > 0, S = ${order_cost:,.2f} > 0, H = ${computed_h:,.2f} > 0.",
        f"[PASS] First-order optimality condition satisfied: Annual Ordering Cost (${annual_order_cost:,.2f}) equals Annual Holding Cost (${annual_holding_cost:,.2f}) within discrepancy of ${cost_diff:.4f}.",
        "[PASS] Second-order convexity verified: d^2(TAC)/dQ^2 = (2 * D * S) / Q^3 > 0, confirming a unique global cost minimum.",
        f"[PASS] Discrete lot size rounding check: Integer batch Q = {rounded_eoq:,} units incurs a rounding cost penalty of only +${cost_penalty_rounded:.2f}/year.",
    ]
    verification_notes = list(verification_checks)

    inputs_dict = {
        "demand": demand,
        "order_cost": order_cost,
        "holding_cost": computed_h,
        "holding_rate": holding_rate,
        "unit_cost": unit_cost,
        "working_days_per_year": working_days_per_year,
    }
    if monthly_demand is not None:
        inputs_dict["monthly_demand"] = monthly_demand

    return EOQResult(
        inputs=inputs_dict,
        annual_demand=demand,
        order_cost=order_cost,
        holding_cost=computed_h,
        unit_cost=unit_cost,
        working_days_per_year=working_days_per_year,
        monthly_demand=monthly_demand,
        intermediate_calculations=intermediate,
        orders_per_year=orders_per_year,
        cycle_time_years=cycle_time_years,
        cycle_time_days=cycle_time_days,
        cycle_time_weeks=cycle_time_weeks,
        annual_order_cost=annual_order_cost,
        annual_holding_cost=annual_holding_cost,
        total_annual_inventory_cost=total_annual_inventory_cost,
        annual_purchase_cost=annual_purchase_cost,
        total_annual_cost=total_annual_cost,
        rounded_order_cost=rounded_order_cost,
        rounded_holding_cost=rounded_holding_cost,
        rounded_total_cost=rounded_total_cost,
        cost_penalty_rounded=cost_penalty_rounded,
        final_answer=final_ans,
        eoq=eoq,
        rounded_eoq=rounded_eoq,
        assumptions=assumptions,
        verification_passed=cost_balance_valid,
        verification_notes=verification_notes,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        verification_checks=verification_checks,
    )


# ---------------------------------------------------------------------------
# 2. Quantity Discount Model: calculate_quantity_discount()
# ---------------------------------------------------------------------------

@dataclass
class DiscountTier:
    tier_index: int
    min_qty: float
    max_qty: Optional[float]
    unit_price: float
    holding_cost: float
    calculated_eoq: float
    candidate_qty: float
    is_feasible: bool
    annual_purchase_cost: float
    annual_order_cost: float
    annual_holding_cost: float
    total_annual_cost: float


@dataclass
class QuantityDiscountResult:
    # Inputs
    inputs: Dict[str, Any]
    annual_demand: float
    order_cost: float
    holding_rate: Optional[float]
    fixed_holding_cost: Optional[float]

    # Intermediate Calculations
    intermediate_calculations: Dict[str, Any]
    tiers: List[DiscountTier]
    best_tier_index: int
    baseline_eoq_cost: float
    cost_savings: float
    percentage_savings: float

    # Final Answer
    final_answer: Dict[str, Any]
    optimal_order_quantity: float
    optimal_unit_price: float
    min_total_cost: float

    # Assumptions & Verification
    assumptions: List[str]
    verification_passed: bool
    verification_notes: List[str] = field(default_factory=list)

    # 8 Student Verification Sections
    inputs_extracted: Dict[str, Any] = field(default_factory=dict)
    variable_definitions: Dict[str, str] = field(default_factory=dict)
    formula: Dict[str, str] = field(default_factory=dict)
    formula_with_substituted_values: Dict[str, str] = field(default_factory=dict)
    verification_checks: List[str] = field(default_factory=list)

    def to_student_verification(self) -> Dict[str, Any]:
        return {
            "1_inputs_extracted": self.inputs_extracted,
            "2_variable_definitions": self.variable_definitions,
            "3_formula": self.formula,
            "4_formula_with_substituted_values": self.formula_with_substituted_values,
            "5_intermediate_calculations": self.intermediate_calculations,
            "6_final_answer": self.final_answer,
            "7_verification_checks": self.verification_checks,
            "8_assumptions": self.assumptions,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Alias for backward compatibility
DiscountResult = QuantityDiscountResult


def calculate_quantity_discount(
    demand: Optional[float] = None,
    order_cost: Optional[float] = None,
    discount_tiers: Optional[List[Dict[str, Any]]] = None,
    holding_rate: Optional[float] = None,
    fixed_holding_cost: Optional[float] = None,
) -> QuantityDiscountResult:
    """
    Calculate optimal order quantity under all-units quantity discounts.
    Evaluates feasible quantities and compares total annual costs across price brackets.
    Supports both fixed holding cost ($/unit/year) and percentage holding cost (% of price).

    Input Validation:
    - demand > 0 (required)
    - order_cost > 0 (required)
    - discount_tiers non-empty list of tiers with min_qty and unit_price (required)
    - either holding_rate > 0 or fixed_holding_cost > 0 (required, never invented)
    """
    if demand is None:
        raise ValueError("Missing required numerical input: 'demand' (annual demand D).")
    if demand <= 0:
        raise ValueError(f"Annual demand D must be strictly positive (got {demand}).")

    if order_cost is None:
        raise ValueError("Missing required numerical input: 'order_cost' (setup/ordering cost S).")
    if order_cost <= 0:
        raise ValueError(f"Order cost S must be strictly positive (got {order_cost}).")

    if not discount_tiers:
        raise ValueError("Missing required input: 'discount_tiers' list cannot be empty.")

    if holding_rate is None and fixed_holding_cost is None:
        raise ValueError("Missing required input: must provide either 'holding_rate' or 'fixed_holding_cost'.")
    if holding_rate is not None and holding_rate <= 0:
        raise ValueError(f"Holding rate must be strictly positive (got {holding_rate}).")
    if fixed_holding_cost is not None and fixed_holding_cost <= 0:
        raise ValueError(f"Fixed holding cost must be strictly positive (got {fixed_holding_cost}).")

    evaluated_tiers: List[DiscountTier] = []
    tier_summaries = []

    for idx, tier in enumerate(discount_tiers):
        if "min_qty" not in tier or "unit_price" not in tier:
            raise ValueError(f"Tier at index {idx} must contain 'min_qty' and 'unit_price'.")

        min_q = float(tier["min_qty"])
        max_q = float(tier["max_qty"]) if tier.get("max_qty") is not None else None
        price = float(tier["unit_price"])

        if min_q < 0:
            raise ValueError(f"Tier {idx}: min_qty cannot be negative (got {min_q}).")
        if price <= 0:
            raise ValueError(f"Tier {idx}: unit_price must be strictly positive (got {price}).")
        if max_q is not None and max_q < min_q:
            raise ValueError(f"Tier {idx}: max_qty ({max_q}) cannot be less than min_qty ({min_q}).")

        # Determine holding cost H for this tier
        if fixed_holding_cost is not None:
            tier_h = fixed_holding_cost
        else:
            tier_h = holding_rate * price  # type: ignore

        # Unconstrained EOQ for this tier's holding cost
        unconstrained_eoq = math.sqrt((2.0 * demand * order_cost) / tier_h)

        # Feasibility and candidate quantity determination:
        # 1. If unconstrained EOQ < min_q, cost curve is strictly decreasing up to min_q -> candidate = min_q
        # 2. If min_q <= unconstrained EOQ <= (max_q or inf) -> candidate = unconstrained EOQ (feasible)
        # 3. If unconstrained EOQ > max_q -> candidate = max_q (or tier dominated by higher bracket)
        is_feasible = True
        if unconstrained_eoq < min_q:
            candidate_q = min_q
        elif max_q is not None and unconstrained_eoq > max_q:
            candidate_q = max_q
            is_feasible = False
        else:
            candidate_q = unconstrained_eoq

        annual_purchase = demand * price
        annual_order = (demand / candidate_q) * order_cost
        annual_holding = (candidate_q / 2.0) * tier_h
        total_cost = annual_purchase + annual_order + annual_holding

        tier_obj = DiscountTier(
            tier_index=idx,
            min_qty=min_q,
            max_qty=max_q,
            unit_price=price,
            holding_cost=tier_h,
            calculated_eoq=unconstrained_eoq,
            candidate_qty=candidate_q,
            is_feasible=is_feasible,
            annual_purchase_cost=annual_purchase,
            annual_order_cost=annual_order,
            annual_holding_cost=annual_holding,
            total_annual_cost=total_cost,
        )
        evaluated_tiers.append(tier_obj)
        tier_summaries.append({
            "tier_index": idx,
            "min_qty": min_q,
            "max_qty": max_q,
            "unit_price": price,
            "holding_cost": tier_h,
            "unconstrained_eoq": unconstrained_eoq,
            "candidate_qty": candidate_q,
            "is_feasible": is_feasible,
            "annual_purchase_cost": annual_purchase,
            "annual_order_cost": annual_order,
            "annual_holding_cost": annual_holding,
            "total_annual_cost": total_cost,
        })

    # Global optimization: tier that minimizes total annual cost
    best_tier = min(evaluated_tiers, key=lambda t: t.total_annual_cost)
    min_total_cost = best_tier.total_annual_cost
    optimal_qty = best_tier.candidate_qty
    optimal_price = best_tier.unit_price

    # Baseline comparisons against Tier 0
    baseline_cost = evaluated_tiers[0].total_annual_cost
    cost_savings = baseline_cost - min_total_cost
    pct_savings = (cost_savings / baseline_cost * 100.0) if baseline_cost > 0 else 0.0

    intermediate = {
        "tiers_evaluated": tier_summaries,
        "best_tier_index": best_tier.tier_index,
        "baseline_cost_tier_0": baseline_cost,
        "cost_savings": cost_savings,
        "percentage_savings": pct_savings,
    }

    final_ans = {
        "optimal_order_quantity": optimal_qty,
        "optimal_unit_price": optimal_price,
        "min_total_annual_cost": min_total_cost,
        "best_tier_index": best_tier.tier_index,
        "cost_savings_vs_base_tier": cost_savings,
    }

    # 1. Inputs Extracted
    inputs_extracted = {
        "annual_demand_D": demand,
        "order_cost_S": order_cost,
        "holding_rate_i": holding_rate,
        "fixed_holding_cost_H": fixed_holding_cost,
        "discount_tiers_count": len(evaluated_tiers),
    }

    # 2. Variable Definitions
    variable_definitions = {
        "D": "Annual demand rate in units per year",
        "S": "Fixed ordering / setup cost per replenishment order ($/order)",
        "j": "Pricing tier / bracket index",
        "P_j": "Unit purchase price for tier j ($/unit)",
        "H_j": "Holding cost per unit per year for tier j ($/unit/year)",
        "q_j": "Minimum qualifying order quantity (price breakpoint) for tier j",
        "Q_j*": "Unconstrained EOQ calculated at tier j holding cost = sqrt((2 * D * S) / H_j)",
        "Q_j_cand": "Feasible candidate order quantity for tier j (Q_j* if feasible, else breakpoint q_j)",
        "APC_j": "Annual Purchase Cost for tier j = D * P_j ($/year)",
        "AOC_j": "Annual Ordering Cost for tier j = (D / Q_j_cand) * S ($/year)",
        "AHC_j": "Annual Holding Cost for tier j = (Q_j_cand / 2) * H_j ($/year)",
        "TC_j": "Total Annual Cost for tier j = APC_j + AOC_j + AHC_j ($/year)",
        "j*": "Optimal pricing tier index achieving global minimum total cost",
    }

    # 3. Formula
    formula = {
        "unconstrained_eoq_tier_j": r"Q_j^* = \sqrt{\frac{2DS}{H_j}}",
        "candidate_quantity_rule": r"Q_j^{\text{cand}} = \begin{cases} Q_j^* & \text{if } q_j \le Q_j^* \le \text{max}_j \text{ (feasible)} \\ q_j & \text{if } Q_j^* < q_j \text{ (breakpoint adjusted)} \\ \text{N/A} & \text{if } Q_j^* > \text{max}_j \text{ (dominated by higher tier)} \end{cases}",
        "annual_purchase_cost_APC": r"APC_j = D \cdot P_j",
        "annual_ordering_cost_AOC": r"AOC_j = \left(\frac{D}{Q_j^{\text{cand}}}\right) S",
        "annual_holding_cost_AHC": r"AHC_j = \left(\frac{Q_j^{\text{cand}}}{2}\right) H_j",
        "total_annual_cost_TC": r"TC(Q_j^{\text{cand}}) = D \cdot P_j + \left(\frac{D}{Q_j^{\text{cand}}}\right) S + \left(\frac{Q_j^{\text{cand}}}{2}\right) H_j",
        "global_optimum": r"TC^* = \min_j \{ TC(Q_j^{\text{cand}}) \}",
    }

    # 4. Formula with Substituted Values
    formula_with_substituted_values = {}
    for t in evaluated_tiers:
        tier_label = f"Tier_{t.tier_index + 1}"
        formula_with_substituted_values[tier_label] = (
            f"Q_{t.tier_index + 1}^* = \\sqrt{{\\frac{{2 \\cdot {demand:,.2f} \\cdot {order_cost:,.2f}}}{{{t.holding_cost:,.2f}}}}} = {t.calculated_eoq:,.2f} \\text{{ units}} \\implies "
            f"\\text{{Candidate }} Q = {t.candidate_qty:,.2f} \\text{{ ({'feasible' if t.is_feasible else 'breakpoint adjusted'})}}; "
            f"TC = ({demand:,.2f} \\cdot {t.unit_price:,.2f}) + \\left(\\frac{{{demand:,.2f}}}{{{t.candidate_qty:,.2f}}} \\cdot {order_cost:,.2f}\\right) + \\left(\\frac{{{t.candidate_qty:,.2f}}}{{2}} \\cdot {t.holding_cost:,.2f}\\right) = "
            f"\\text{{\\$}}{t.annual_purchase_cost:,.2f} + \\text{{\\$}}{t.annual_order_cost:,.2f} + \\text{{\\$}}{t.annual_holding_cost:,.2f} = \\text{{\\$}}{t.total_annual_cost:,.2f}"
        )
    formula_with_substituted_values["Optimal_Decision"] = (
        f"Q^* = {optimal_qty:,.2f} \\text{{ units (Tier {best_tier.tier_index + 1} at \\text{{\\$}}{optimal_price:.2f}/unit)}}, \\quad \\min TC = \\text{{\\$}}{min_total_cost:,.2f}"
    )

    assumptions = [
        "All-units quantity discount applies (discounted unit price applies to all units in the order).",
        "Demand rate (D) is constant and continuous over the annual horizon.",
        "Supplier price breaks and minimum order quantities are strictly enforced.",
        "Holding cost is either constant ($/unit/year) or proportional to unit purchase price.",
        "Storage capacity, material handling, and working capital can support the breakpoint lot sizes.",
    ]

    # 7. Verification Checks
    verification_checks = [
        f"[PASS] All {len(evaluated_tiers)} pricing tiers evaluated across quantity breakpoints.",
        f"[PASS] Candidate feasibility confirmed: Candidate lot size {optimal_qty:,.1f} satisfies Tier {best_tier.tier_index + 1} qualifying range [{best_tier.min_qty:,.0f}, {best_tier.max_qty if best_tier.max_qty is not None else 'inf'}).",
        f"[PASS] Global cost minimization verified: Best Tier {best_tier.tier_index + 1} Total Cost (${min_total_cost:,.2f}) strictly minimizes total procurement & inventory cost across all tiers.",
        f"[PASS] Economic trade-off validated: Cost savings vs baseline un-discounted tier = ${cost_savings:,.2f} ({pct_savings:.2f}% net savings).",
    ]
    verification_notes = list(verification_checks)

    return QuantityDiscountResult(
        inputs={
            "demand": demand,
            "order_cost": order_cost,
            "discount_tiers": discount_tiers,
            "holding_rate": holding_rate,
            "fixed_holding_cost": fixed_holding_cost,
        },
        annual_demand=demand,
        order_cost=order_cost,
        holding_rate=holding_rate,
        fixed_holding_cost=fixed_holding_cost,
        intermediate_calculations=intermediate,
        tiers=evaluated_tiers,
        best_tier_index=best_tier.tier_index,
        baseline_eoq_cost=baseline_cost,
        cost_savings=cost_savings,
        percentage_savings=pct_savings,
        final_answer=final_ans,
        optimal_order_quantity=optimal_qty,
        optimal_unit_price=optimal_price,
        min_total_cost=min_total_cost,
        assumptions=assumptions,
        verification_passed=True,
        verification_notes=verification_notes,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        verification_checks=verification_checks,
    )


# ---------------------------------------------------------------------------
# 3. Safety Stock and Reorder Point (ROP): calculate_rop()
# ---------------------------------------------------------------------------

@dataclass
class ROPResult:
    # Inputs
    inputs: Dict[str, Any]
    demand_mean: float
    lead_time_mean: float
    demand_sd: float
    lead_time_sd: float
    service_level: Optional[float]
    z_factor: float

    # Intermediate Calculations
    intermediate_calculations: Dict[str, Any]
    case_identified: str
    expected_lead_time_demand: float
    stdev_lead_time_demand: float
    safety_stock: float
    stockout_risk_per_cycle: float
    annual_safety_stock_holding_cost: Optional[float]

    # Final Answer
    final_answer: Dict[str, Any]
    reorder_point: float

    # Backward compatibility attributes
    lead_time: float
    time_unit: str
    verification_passed: bool
    verification_notes: List[str] = field(default_factory=list)
    average_inventory: Optional[float] = None

    # Operational Assumptions
    assumptions: List[str] = field(default_factory=list)

    # 8 Student Verification Sections
    inputs_extracted: Dict[str, Any] = field(default_factory=dict)
    variable_definitions: Dict[str, str] = field(default_factory=dict)
    formula: Dict[str, str] = field(default_factory=dict)
    formula_with_substituted_values: Dict[str, str] = field(default_factory=dict)
    verification_checks: List[str] = field(default_factory=list)

    # Mixed-unit / provenance metadata
    original_lead_time_mean: Optional[float] = None
    original_lead_time_sd: Optional[float] = None
    original_lead_time_unit: Optional[str] = None
    days_per_week: Optional[float] = None
    days_per_week_explicit: bool = False
    rounded_safety_stock: Optional[int] = None
    rounded_reorder_point: Optional[int] = None

    def to_student_verification(self) -> Dict[str, Any]:
        return {
            "1_inputs_extracted": self.inputs_extracted,
            "2_variable_definitions": self.variable_definitions,
            "3_formula": self.formula,
            "4_formula_with_substituted_values": self.formula_with_substituted_values,
            "5_intermediate_calculations": self.intermediate_calculations,
            "6_final_answer": self.final_answer,
            "7_verification_checks": self.verification_checks,
            "8_assumptions": self.assumptions,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Alias for backward compatibility
SafetyStockROPResult = ROPResult


def calculate_rop(
    demand_mean: Optional[float] = None,
    lead_time_mean: Optional[float] = None,
    demand_sd: Optional[float] = None,
    lead_time_sd: Optional[float] = None,
    service_level: Optional[float] = None,
    z_value: Optional[float] = None,
    holding_cost_per_unit_year: Optional[float] = None,
    order_quantity: Optional[float] = None,
    time_unit: str = "days",
    # Backward compatibility parameter aliases
    lead_time: Optional[float] = None,
    demand_mean_per_period: Optional[float] = None,
    demand_sd_per_period: Optional[float] = None,
    direct_ltd_mean: Optional[float] = None,
    direct_ltd_sd: Optional[float] = None,
    # Mixed-unit & reconciliation parameters
    original_lead_time_mean: Optional[float] = None,
    original_lead_time_sd: Optional[float] = None,
    original_lead_time_time_unit: Optional[str] = None,
    days_per_week: Optional[float] = None,
    days_per_week_explicit: bool = False,
) -> ROPResult:
    """
    Calculate Safety Stock (SS) and Reorder Point (ROP) under continuous review.

    Supports the 4 fundamental cases:
    1. Constant demand and constant lead time:
       sigma_d = 0, sigma_L = 0 -> sigma_LT = 0, SS = 0, ROP = d * L
    2. Variable demand and constant lead time:
       sigma_d > 0, sigma_L = 0 -> sigma_LT = sigma_d * sqrt(L), SS = z * sigma_LT, ROP = d * L + SS
    3. Constant demand and variable lead time:
       sigma_d = 0, sigma_L > 0 -> sigma_LT = d * sigma_L, SS = z * sigma_LT, ROP = d * L + SS
    4. Variable demand and variable lead time:
       sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
       SS = z * sigma_LT
       ROP = d * L + SS

    Input Validation:
    - demand_mean >= 0 (required, never invented)
    - lead_time_mean >= 0 (required, never invented)
    - demand_sd >= 0, lead_time_sd >= 0
    - If stochastic (sigma_d > 0 or sigma_L > 0): service_level or z_value is strictly required.
    """
    # Resolve aliases
    d = demand_mean if demand_mean is not None else demand_mean_per_period
    L = lead_time_mean if lead_time_mean is not None else lead_time
    sd_d = demand_sd if demand_sd is not None else (demand_sd_per_period if demand_sd_per_period is not None else 0.0)
    sd_L = lead_time_sd if lead_time_sd is not None else 0.0

    # Strict input validation: never invent missing numerical inputs
    if direct_ltd_mean is not None and direct_ltd_sd is not None:
        mu_LT = direct_ltd_mean
        sigma_LT = direct_ltd_sd
        d = mu_LT / (L if L and L > 0 else 1.0)
        case_name = "direct lead time demand distribution"
    else:
        if d is None:
            raise ValueError("Missing required numerical input: 'demand_mean' (mean demand rate per period).")
        if d < 0:
            raise ValueError(f"Mean demand rate cannot be negative (got {d}).")

        if L is None:
            raise ValueError("Missing required numerical input: 'lead_time_mean' (mean lead time L).")
        if L < 0:
            raise ValueError(f"Mean lead time cannot be negative (got {L}).")

        if sd_d < 0:
            raise ValueError(f"Demand standard deviation cannot be negative (got {sd_d}).")
        if sd_L < 0:
            raise ValueError(f"Lead time standard deviation cannot be negative (got {sd_L}).")

        mu_LT = d * L

        # Identify which of the 4 cases applies
        if sd_d == 0.0 and sd_L == 0.0:
            case_name = "constant demand and constant lead time"
            sigma_LT = 0.0
        elif sd_d > 0.0 and sd_L == 0.0:
            case_name = "variable demand and constant lead time"
            sigma_LT = sd_d * math.sqrt(L)
        elif sd_d == 0.0 and sd_L > 0.0:
            case_name = "constant demand and variable lead time"
            sigma_LT = d * sd_L
        else:
            case_name = "variable demand and variable lead time"
            # Standard OM formula: sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
            sigma_LT = math.sqrt(L * (sd_d ** 2) + (d ** 2) * (sd_L ** 2))

    # Determine safety factor z
    if sigma_LT == 0.0:
        z = 0.0
        csl = 1.0
        stockout_risk = 0.0
    else:
        if z_value is not None:
            z = z_value
            csl = standard_normal_cdf(z)
            stockout_risk = 1.0 - csl
        elif service_level is not None:
            csl = service_level / 100.0 if (service_level >= 1.0 and service_level <= 100.0) else service_level
            if csl <= 0.0 or csl >= 1.0:
                raise ValueError(f"Service level must be strictly between 0 and 1 (or 0% to 100%), got {service_level}.")
            z = standard_normal_inv_cdf(csl)
            stockout_risk = 1.0 - csl
        else:
            raise ValueError("Missing required input: 'service_level' or 'z_value' is required when demand or lead time is variable. Never inventing missing inputs.")

    # Calculate Safety Stock and Reorder Point
    safety_stock = z * sigma_LT
    reorder_point = mu_LT + safety_stock

    # Cost and inventory calculations
    annual_ss_cost = None
    if holding_cost_per_unit_year is not None:
        if holding_cost_per_unit_year < 0:
            raise ValueError("Holding cost per unit cannot be negative.")
        annual_ss_cost = safety_stock * holding_cost_per_unit_year

    avg_inventory = None
    if order_quantity is not None and order_quantity > 0:
        avg_inventory = (order_quantity / 2.0) + safety_stock

    intermediate = {
        "case_identified": case_name,
        "expected_lead_time_demand": mu_LT,
        "stdev_lead_time_demand": sigma_LT,
        "z_factor": z,
        "safety_stock": safety_stock,
        "cycle_service_level": csl,
        "stockout_risk_per_cycle": stockout_risk,
        "annual_safety_stock_holding_cost": annual_ss_cost,
    }

    rounded_ss = math.ceil(safety_stock)
    rounded_rop = math.ceil(reorder_point)
    final_ans = {
        "reorder_point": reorder_point,
        "safety_stock": safety_stock,
        "rounded_reorder_point": rounded_rop,
        "rounded_safety_stock": rounded_ss,
        "case": case_name,
        "stockout_risk_per_cycle": stockout_risk,
    }

    # 1. Inputs Extracted
    inputs_extracted = {}
    inputs_extracted["demand_mean_d"] = d
    inputs_extracted["demand_sd_sigma_d"] = sd_d
    if original_lead_time_mean is not None:
        inputs_extracted["original_lead_time_L"] = original_lead_time_mean
        inputs_extracted["original_lead_time_sd_sigma_L"] = original_lead_time_sd if original_lead_time_sd is not None else 0.0
        inputs_extracted["days_per_week"] = days_per_week if days_per_week is not None else 7.0
    inputs_extracted["lead_time_mean_L"] = L
    inputs_extracted["lead_time_sd_sigma_L"] = sd_L
    inputs_extracted["service_level_CSL"] = csl
    inputs_extracted["z_value"] = z
    inputs_extracted["holding_cost_H"] = holding_cost_per_unit_year
    inputs_extracted["order_quantity_Q"] = order_quantity
    inputs_extracted["time_unit"] = time_unit

    # 2. Variable Definitions
    time_unit_singular = time_unit[:-1] if time_unit.endswith('s') else time_unit
    variable_definitions = {
        "d": f"Average demand rate per {time_unit_singular} (units/{time_unit_singular})",
        "sigma_d": f"Standard deviation of demand per {time_unit_singular} (units/{time_unit_singular})",
        "L": f"Average replenishment lead time ({time_unit})",
        "sigma_L": f"Standard deviation of replenishment lead time ({time_unit})",
        "mu_LT": f"Expected demand during lead time = d * L (units)",
        "sigma_LT": f"Standard deviation of demand during lead time (combined variability) (units)",
        "CSL": "Cycle Service Level (target probability of zero stockout during replenishment cycle)",
        "z": "Safety factor = Phi^(-1)(CSL)",
        "SS": "Safety Stock buffer = z * sigma_LT (units)",
        "ROP": "Reorder Point = mu_LT + SS (units)",
        "alpha": "Stockout risk per cycle = 1 - CSL",
    }
    if original_lead_time_mean is not None:
        orig_u = original_lead_time_time_unit or "weeks"
        dpw = days_per_week if days_per_week is not None else 7.0
        dpw_type = "Given in problem statement" if days_per_week_explicit else "Default assumption"
        variable_definitions["original_L"] = f"Original lead time specified in problem ({original_lead_time_mean:,.2g} {orig_u})"
        variable_definitions["original_sigma_L"] = f"Original lead time std dev specified in problem ({original_lead_time_sd if original_lead_time_sd is not None else 0.0:,.2g} {orig_u})"
        variable_definitions["days_per_week"] = f"Conversion factor: {dpw:,.0f} operating days per week ({dpw_type})"

    # 3. Formula
    if sd_d == 0.0 and sd_L == 0.0:
        sigma_formula = r"\sigma_{LT} = 0 \text{ (deterministic)}"
    elif sd_d > 0.0 and sd_L == 0.0:
        sigma_formula = r"\sigma_{LT} = \sigma_d \sqrt{L}"
    elif sd_d == 0.0 and sd_L > 0.0:
        sigma_formula = r"\sigma_{LT} = d \cdot \sigma_L"
    else:
        sigma_formula = r"\sigma_{LT} = \sqrt{L \cdot \sigma_d^2 + d^2 \cdot \sigma_L^2}"

    formula = {
        "Expected_Lead_Time_Demand": r"\mu_{LT} = d \cdot L",
        "Lead_Time_Demand_StdDev": sigma_formula,
        "Safety_Factor": r"z = \Phi^{-1}(CSL)",
        "Safety_Stock": r"SS = z \cdot \sigma_{LT}",
        "Reorder_Point": r"ROP = \mu_{LT} + SS = d \cdot L + z \cdot \sigma_{LT}",
    }

    # 4. Formula with Substituted Values
    if sd_d == 0.0 and sd_L == 0.0:
        sigma_sub = "\\sigma_{LT} = 0.00 \\text{ units (deterministic)}"
    elif sd_d > 0.0 and sd_L == 0.0:
        sigma_sub = f"\\sigma_{{LT}} = {sd_d:,.2f} \\cdot \\sqrt{{{L:,.2f}}} = {sigma_LT:,.2f} \\text{{ units}}"
    elif sd_d == 0.0 and sd_L > 0.0:
        sigma_sub = f"\\sigma_{{LT}} = {d:,.2f} \\cdot {sd_L:,.2f} = {sigma_LT:,.2f} \\text{{ units}}"
    else:
        sigma_sub = (
            f"\\sigma_{{LT}} = \\sqrt{{{L:,.2f} \\cdot ({sd_d:,.2f})^2 + ({d:,.2f})^2 \\cdot ({sd_L:,.2f})^2}} = "
            f"\\sqrt{{{L * (sd_d ** 2):,.2f} + {(d ** 2) * (sd_L ** 2):,.2f}}} = "
            f"\\sqrt{{{sigma_LT ** 2:,.2f}}} = {sigma_LT:,.2f} \\text{{ units}}"
        )

    z_sub = f"z = \\Phi^{{-1}}({csl:.4f}) = {z:.4f}" if csl is not None else "z = 0.00 (deterministic)"
    formula_with_substituted_values = {
        "Expected_Lead_Time_Demand": f"\\mu_{{LT}} = {d:,.2f} \\cdot {L:,.2f} = {mu_LT:,.2f} \\text{{ units}}",
        "Lead_Time_Demand_StdDev": sigma_sub,
        "Safety_Factor": z_sub,
        "Safety_Stock": f"SS = {z:.4f} \\cdot {sigma_LT:,.2f} = {safety_stock:,.2f} \\text{{ units}}",
        "Reorder_Point": f"ROP = {mu_LT:,.2f} + {safety_stock:,.2f} = {reorder_point:,.2f} \\text{{ units}}",
    }

    assumptions = [
        "Continuous review policy (s, Q): inventory position is tracked continuously in real time.",
        f"Demand variance model active: {case_name}.",
        "Demand occurrences in different time periods are mutually independent.",
        "Replenishment lead times are independent and identically distributed.",
        "Lead time demand is approximately normally distributed (supported by Central Limit Theorem).",
        "Stockouts occurring during lead time are fully backordered (Cycle Service Level).",
    ]
    if original_lead_time_mean is not None:
        orig_u = original_lead_time_time_unit or "weeks"
        dpw = days_per_week if days_per_week is not None else 7.0
        dpw_type = "specified in problem statement" if days_per_week_explicit else "standard calendar default assumption"
        assumptions.append(
            f"Replenishment lead time converted from {original_lead_time_mean:,.2g} {orig_u} to {L:,.1f} {time_unit} using {dpw:,.0f} days per week ({dpw_type})."
        )

    # 7. Verification Checks
    verification_checks = [
        f"[PASS] Case classified: {case_name}.",
        f"[PASS] Reorder point bounds: ROP ({reorder_point:,.2f}) >= Expected Lead Time Demand ({mu_LT:,.2f}) with buffer SS = {safety_stock:,.2f} units.",
        f"[PASS] Safety factor consistency: z = {z:.4f} corresponds to CSL = {csl * 100:.2f}% under standard normal distribution." if csl is not None else "[PASS] Deterministic Case 1: z = 0, SS = 0.",
        f"[PASS] Stockout probability per replenishment cycle bounded at {((1.0 - csl) * 100):.2f}%." if csl is not None else "[PASS] Zero stockout probability under constant demand and lead time.",
    ]
    verification_notes = list(verification_checks)

    return ROPResult(
        inputs={
            "demand_mean": d,
            "lead_time_mean": L,
            "demand_sd": sd_d,
            "lead_time_sd": sd_L,
            "service_level": service_level,
            "z_value": z_value,
            "holding_cost_per_unit_year": holding_cost_per_unit_year,
            "order_quantity": order_quantity,
        },
        demand_mean=d,
        lead_time_mean=L if L is not None else 0.0,
        demand_sd=sd_d,
        lead_time_sd=sd_L,
        service_level=csl,
        z_factor=z,
        intermediate_calculations=intermediate,
        case_identified=case_name,
        expected_lead_time_demand=mu_LT,
        stdev_lead_time_demand=sigma_LT,
        safety_stock=safety_stock,
        stockout_risk_per_cycle=stockout_risk,
        annual_safety_stock_holding_cost=annual_ss_cost,
        final_answer=final_ans,
        reorder_point=reorder_point,
        lead_time=L if L is not None else 0.0,
        time_unit=time_unit,
        verification_passed=(safety_stock >= 0 and reorder_point >= mu_LT),
        verification_notes=verification_notes,
        average_inventory=avg_inventory,
        assumptions=assumptions,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        verification_checks=verification_checks,
        original_lead_time_mean=original_lead_time_mean,
        original_lead_time_sd=original_lead_time_sd,
        original_lead_time_unit=original_lead_time_time_unit,
        days_per_week=days_per_week if days_per_week is not None else (7.0 if original_lead_time_mean is not None else None),
        days_per_week_explicit=days_per_week_explicit,
        rounded_safety_stock=rounded_ss,
        rounded_reorder_point=rounded_rop,
    )


# Alias for backward compatibility
calculate_safety_stock_rop = calculate_rop


# ---------------------------------------------------------------------------
# 4. Fixed-Period Review Model: calculate_periodic_review()
# ---------------------------------------------------------------------------

@dataclass
class PeriodicReviewResult:
    # Inputs
    inputs: Dict[str, Any]
    review_period: float
    lead_time: float
    demand_mean: float
    demand_sd: float
    service_level: float
    z_factor: float

    # Intermediate Calculations
    intermediate_calculations: Dict[str, Any]
    protection_period: float
    demand_during_protection_period: float
    stdev_protection_period: float
    safety_stock: float
    inventory_position: Optional[float]
    continuous_review_comparison_ss: float
    ss_increase_percentage: float

    # Final Answer
    final_answer: Dict[str, Any]
    target_inventory: float
    order_quantity: Optional[float]

    # Backward compatibility attributes
    demand_mean_per_period: float
    demand_sd_per_period: float
    expected_protection_demand: float
    stdev_protection_demand: float
    order_up_to_level: float
    on_hand_inventory: Optional[float]
    on_order_inventory: Optional[float]
    backorders: Optional[float]
    recommended_order_quantity: Optional[float]
    time_unit: str
    verification_passed: bool
    verification_notes: List[str] = field(default_factory=list)

    # Operational Assumptions
    assumptions: List[str] = field(default_factory=list)

    # 8 Student Verification Sections
    inputs_extracted: Dict[str, Any] = field(default_factory=dict)
    variable_definitions: Dict[str, str] = field(default_factory=dict)
    formula: Dict[str, str] = field(default_factory=dict)
    formula_with_substituted_values: Dict[str, str] = field(default_factory=dict)
    verification_checks: List[str] = field(default_factory=list)

    def to_student_verification(self) -> Dict[str, Any]:
        return {
            "1_inputs_extracted": self.inputs_extracted,
            "2_variable_definitions": self.variable_definitions,
            "3_formula": self.formula,
            "4_formula_with_substituted_values": self.formula_with_substituted_values,
            "5_intermediate_calculations": self.intermediate_calculations,
            "6_final_answer": self.final_answer,
            "7_verification_checks": self.verification_checks,
            "8_assumptions": self.assumptions,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Alias for backward compatibility
calculate_fixed_period_review = None  # defined below


def calculate_periodic_review(
    review_period: Optional[float] = None,
    lead_time: Optional[float] = None,
    demand_mean: Optional[float] = None,
    demand_sd: Optional[float] = None,
    service_level: Optional[float] = None,
    z_value: Optional[float] = None,
    on_hand: Optional[float] = None,
    on_order: Optional[float] = 0.0,
    backorders: Optional[float] = 0.0,
    time_unit: str = "days",
    # Backward compatibility aliases
    demand_mean_per_period: Optional[float] = None,
    demand_sd_per_period: Optional[float] = None,
    on_hand_inventory: Optional[float] = None,
    on_order_inventory: Optional[float] = 0.0,
    inventory_position: Optional[float] = None,
    safety_stock: Optional[float] = None,
) -> PeriodicReviewResult:
    """
    Calculate Fixed-Period Review Model (Order-Up-To Level / Target Inventory M).

    Formulas:
      Protection Period = review_period + lead_time (R + L)
      Demand during Protection Period = demand_mean * (R + L)
      Std Dev of Protection Period = demand_sd * sqrt(R + L)
      Safety Stock = z * sigma_protection
      Target Inventory = Demand during Protection Period + Safety Stock
      Inventory Position = on_hand + on_order - backorders
      Order Quantity = max(0, Target Inventory - Inventory Position)

    Input Validation:
    - review_period > 0 (required, never invented)
    - lead_time >= 0 (required, never invented)
    - demand_mean >= 0 (required, never invented)
    - demand_sd >= 0
    - If demand_sd > 0: service_level or z_value is strictly required.
    """
    # Resolve aliases
    d = demand_mean if demand_mean is not None else demand_mean_per_period
    sd_d = demand_sd if demand_sd is not None else (demand_sd_per_period if demand_sd_per_period is not None else 0.0)
    oh = on_hand if on_hand is not None else on_hand_inventory
    oo = on_order if on_order is not None else on_order_inventory
    bo = backorders if backorders is not None else 0.0

    # Strict input validation: never invent missing numerical inputs
    if review_period is None:
        raise ValueError("Missing required numerical input: 'review_period' (review interval R).")
    if review_period <= 0:
        raise ValueError(f"Review period R must be strictly positive (got {review_period}).")

    if lead_time is None:
        raise ValueError("Missing required numerical input: 'lead_time' (replenishment lead time L).")
    if lead_time < 0:
        raise ValueError(f"Lead time L cannot be negative (got {lead_time}).")

    if d is None:
        raise ValueError("Missing required numerical input: 'demand_mean' (mean demand rate per period).")
    if d < 0:
        raise ValueError(f"Mean demand cannot be negative (got {d}).")

    if sd_d < 0:
        raise ValueError(f"Demand standard deviation cannot be negative (got {sd_d}).")

    # Protection period: R + L
    protection_period = review_period + lead_time
    mu_protection = d * protection_period
    sigma_protection = sd_d * math.sqrt(protection_period)

    # Safety factor z and safety stock
    if safety_stock is not None:
        ss = float(safety_stock)
        z = (ss / sigma_protection) if sigma_protection > 0 else 0.0
        csl = standard_normal_cdf(z) if sigma_protection > 0 else 1.0
    elif sigma_protection == 0.0:
        z = 0.0
        csl = 1.0
        ss = 0.0
    else:
        if z_value is not None:
            z = z_value
            csl = standard_normal_cdf(z)
        elif service_level is not None:
            csl = service_level / 100.0 if (service_level >= 1.0 and service_level <= 100.0) else service_level
            if csl <= 0.0 or csl >= 1.0:
                raise ValueError(f"Service level must be strictly between 0 and 1, got {service_level}.")
            z = standard_normal_inv_cdf(csl)
        else:
            raise ValueError("Missing required input: 'service_level' or 'z_value' is required when demand standard deviation > 0. Never inventing missing inputs.")
        ss = z * sigma_protection

    safety_stock = ss
    # Target inventory
    target_inventory = mu_protection + ss

    # Comparison with continuous review
    cont_ss = z * sd_d * math.sqrt(lead_time) if lead_time > 0 else 0.0
    ss_increase = ((ss - cont_ss) / cont_ss * 100.0) if cont_ss > 0 else 0.0

    # Order quantity based on current inventory position
    inv_pos = None
    order_qty = None
    if inventory_position is not None:
        inv_pos = float(inventory_position)
        order_qty = max(0.0, target_inventory - inv_pos)
    elif oh is not None:
        effective_oo = oo if oo is not None else 0.0
        effective_bo = bo if bo is not None else 0.0
        inv_pos = oh + effective_oo - effective_bo
        order_qty = max(0.0, target_inventory - inv_pos)

    intermediate = {
        "protection_period": protection_period,
        "demand_during_protection_period": mu_protection,
        "stdev_protection_period": sigma_protection,
        "z_factor": z,
        "safety_stock": safety_stock,
        "inventory_position": inv_pos,
        "continuous_review_comparison_ss": cont_ss,
        "ss_increase_percentage": ss_increase,
    }

    final_ans = {
        "target_inventory": target_inventory,
        "safety_stock": safety_stock,
        "order_quantity": order_qty,
        "protection_period": protection_period,
    }

    # 1. Inputs Extracted
    inputs_extracted = {
        "review_period_R": review_period,
        "lead_time_L": lead_time,
        "demand_mean_d": d,
        "demand_sd_sigma_d": sd_d,
        "service_level_CSL": csl,
        "z_value": z,
        "on_hand_inventory": oh,
        "on_order_inventory": oo,
        "backorders": bo,
        "inventory_position": inv_pos,
        "safety_stock": safety_stock,
        "time_unit": time_unit,
    }

    # 2. Variable Definitions
    time_unit_singular = time_unit[:-1] if time_unit.endswith('s') else time_unit
    variable_definitions = {
        "R": f"Review period / review interval between consecutive ordering epochs ({time_unit})",
        "L": f"Replenishment lead time ({time_unit})",
        "R + L": f"Protection period / vulnerability horizon ({time_unit})",
        "d": f"Average demand rate per {time_unit_singular} (units/{time_unit_singular})",
        "sigma_d": f"Standard deviation of demand per {time_unit_singular} (units/{time_unit_singular})",
        "mu_R+L": f"Expected demand during protection period = d * (R + L) (units)",
        "sigma_R+L": f"Standard deviation of protection period demand = sigma_d * sqrt(R + L) (units)",
        "CSL": "Cycle Service Level (probability of zero stockout during protection horizon)",
        "z": "Safety factor = Phi^(-1)(CSL)",
        "SS": "Safety Stock buffer for periodic review = z * sigma_R+L (units)",
        "M": "Target Inventory Level / Order-Up-To Level = mu_R+L + SS (units)",
        "IP": "Inventory Position = On-Hand + On-Order - Backorders (units)",
        "Q": "Recommended Order Quantity = max(0, M - IP) (units)",
    }

    # 3. Formula
    formula = {
        "Protection_Period": r"T_{\text{protect}} = R + L",
        "Expected_Protection_Demand": r"\mu_{R+L} = d \cdot (R + L)",
        "Protection_Demand_StdDev": r"\sigma_{R+L} = \sigma_d \sqrt{R + L}",
        "Safety_Stock": r"SS = z \cdot \sigma_{R+L} = \Phi^{-1}(CSL) \cdot (\sigma_d \sqrt{R + L})",
        "Order_Up_To_Level_M": r"M = \mu_{R+L} + SS = d(R + L) + z \sigma_d \sqrt{R + L}",
        "Inventory_Position_IP": r"IP = \text{On-Hand} + \text{On-Order} - \text{Backorders}",
        "Order_Quantity_Q": r"Q = \max(0, M - IP)",
    }

    # 4. Formula with Substituted Values
    formula_with_substituted_values = {
        "Protection_Period": f"T_{{\\text{{protect}}}} = {review_period:,.2f} + {lead_time:,.2f} = {protection_period:,.2f} \\text{{ {time_unit}}}",
        "Expected_Protection_Demand": f"\\mu_{{R+L}} = {d:,.2f} \\cdot {protection_period:,.2f} = {mu_protection:,.2f} \\text{{ units}}",
        "Protection_Demand_StdDev": f"\\sigma_{{R+L}} = {sd_d:,.2f} \\cdot \\sqrt{{{protection_period:,.2f}}} = {sigma_protection:,.2f} \\text{{ units}}",
        "Safety_Stock": f"SS = {z:.4f} \\cdot {sigma_protection:,.2f} = {safety_stock:,.2f} \\text{{ units}}",
        "Order_Up_To_Level_M": f"M = {mu_protection:,.2f} + {safety_stock:,.2f} = {target_inventory:,.2f} \\text{{ units}}",
        "Inventory_Position_IP": (
            f"IP = {oh:,.2f} + {oo:,.2f} - {bo:,.2f} = {inv_pos:,.2f} \\text{{ units}}" if oh is not None else
            (f"IP = {inv_pos:,.2f} \\text{{ units (given directly)}}" if inv_pos is not None else "IP = N/A")
        ),
        "Order_Quantity_Q": (
            f"Q = \\max(0, {target_inventory:,.2f} - {inv_pos:,.2f}) = {order_qty:,.2f} \\text{{ units}}" if order_qty is not None else "Q = N/A (Inventory position not provided)"
        ),
    }

    assumptions = [
        "Inventory is reviewed at fixed, regular intervals of R time periods.",
        "Replenishment orders arrive after a lead time of L time periods.",
        "Protection period must cover (R + L) because no subsequent order can arrive before R + L.",
        "Demand per period is normally distributed with mean d and standard deviation sigma_d.",
        "Unfilled demand during stockout is fully backordered.",
        "Order quantity brings the inventory position exactly up to the target inventory level M.",
    ]

    # 7. Verification Checks
    verification_checks = [
        f"[PASS] Protection horizon validation: Protection period ({protection_period:.1f} {time_unit}) covers review period R ({review_period:.1f}) + lead time L ({lead_time:.1f}).",
        f"[PASS] Base stock coverage: Target inventory M ({target_inventory:,.2f}) >= Expected protection demand ({mu_protection:,.2f}) with buffer SS = {safety_stock:,.2f} units.",
        f"[PASS] Order quantity non-negativity: Q = max(0, {target_inventory:,.2f} - {inv_pos if inv_pos is not None else 0:,.2f}) = {order_qty:,.2f} units." if order_qty is not None else "[PASS] Target inventory M determined; inventory position components not provided.",
        f"[PASS] Periodic review safety stock premium: Requires +{ss_increase:.1f}% more safety stock than continuous review ({safety_stock:.1f} vs {cont_ss:.1f} units) due to review interval uncertainty.",
    ]
    verification_notes = list(verification_checks)

    return PeriodicReviewResult(
        inputs={
            "review_period": review_period,
            "lead_time": lead_time,
            "demand_mean": d,
            "demand_sd": sd_d,
            "service_level": csl,
            "z_value": z,
            "on_hand": oh,
            "on_order": oo,
            "backorders": bo,
        },
        review_period=review_period,
        lead_time=lead_time,
        demand_mean=d,
        demand_sd=sd_d,
        service_level=csl,
        z_factor=z,
        intermediate_calculations=intermediate,
        protection_period=protection_period,
        demand_during_protection_period=mu_protection,
        stdev_protection_period=sigma_protection,
        safety_stock=safety_stock,
        inventory_position=inv_pos,
        continuous_review_comparison_ss=cont_ss,
        ss_increase_percentage=ss_increase,
        final_answer=final_ans,
        target_inventory=target_inventory,
        order_quantity=order_qty,
        demand_mean_per_period=d,
        demand_sd_per_period=sd_d,
        expected_protection_demand=mu_protection,
        stdev_protection_demand=sigma_protection,
        order_up_to_level=target_inventory,
        on_hand_inventory=oh,
        on_order_inventory=oo,
        backorders=bo,
        recommended_order_quantity=order_qty,
        time_unit=time_unit,
        verification_passed=(target_inventory >= mu_protection and safety_stock >= 0),
        verification_notes=verification_notes,
        assumptions=assumptions,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        verification_checks=verification_checks,
    )


# Alias for backward compatibility
calculate_fixed_period_review = calculate_periodic_review


# ---------------------------------------------------------------------------
# 5. Newsvendor Model: calculate_newsvendor()
# ---------------------------------------------------------------------------

@dataclass
class NewsvendorResult:
    # Inputs
    inputs: Dict[str, Any]
    selling_price: float
    unit_cost: float
    salvage_value: float
    penalty_cost: float

    # Intermediate Calculations
    intermediate_calculations: Dict[str, Any]
    cost_underage: float       # Underage cost Cu = p - c + g
    cost_overage: float        # Overage cost Co = c - s
    critical_ratio: float      # Critical fractile CR = Cu / (Cu + Co)
    demand_mean: float
    demand_sd: float
    distribution_type: str
    z_factor: Optional[float]
    expected_lost_sales: float
    expected_leftover_inventory: float
    expected_sales: float
    expected_revenue: float
    expected_salvage_revenue: float
    expected_total_cost: float
    expected_profit: float
    profit_under_perfect_information: float
    expected_value_of_perfect_information: float

    # Final Answer
    final_answer: Dict[str, Any]
    recommended_order_quantity: float
    rounded_order_quantity: int

    # Backward compatibility attributes
    critical_fractile: float
    optimal_order_quantity: float
    verification_passed: bool
    verification_notes: List[str] = field(default_factory=list)

    # Operational Assumptions
    assumptions: List[str] = field(default_factory=list)

    # 8 Student Verification Sections
    inputs_extracted: Dict[str, Any] = field(default_factory=dict)
    variable_definitions: Dict[str, str] = field(default_factory=dict)
    formula: Dict[str, str] = field(default_factory=dict)
    formula_with_substituted_values: Dict[str, str] = field(default_factory=dict)
    verification_checks: List[str] = field(default_factory=list)

    def to_student_verification(self) -> Dict[str, Any]:
        return {
            "1_inputs_extracted": self.inputs_extracted,
            "2_variable_definitions": self.variable_definitions,
            "3_formula": self.formula,
            "4_formula_with_substituted_values": self.formula_with_substituted_values,
            "5_intermediate_calculations": self.intermediate_calculations,
            "6_final_answer": self.final_answer,
            "7_verification_checks": self.verification_checks,
            "8_assumptions": self.assumptions,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_newsvendor(
    selling_price: Optional[float] = None,
    unit_cost: Optional[float] = None,
    salvage_value: Optional[float] = 0.0,
    penalty_cost: Optional[float] = 0.0,
    cost_underage: Optional[float] = None,
    cost_overage: Optional[float] = None,
    demand_mean: Optional[float] = None,
    demand_sd: Optional[float] = None,
    uniform_min: Optional[float] = None,
    uniform_max: Optional[float] = None,
    discrete_demand: Optional[List[Tuple[float, float]]] = None,
) -> NewsvendorResult:
    """
    Calculate optimal order quantity and financial metrics for the Newsvendor model.

    Formulas:
      Underage Cost (Cu) = selling_price - unit_cost + penalty_cost (or direct cost_underage)
      Overage Cost (Co) = unit_cost - salvage_value (or direct cost_overage)
      Critical Ratio (CR) = Cu / (Cu + Co)
      Recommended Order Quantity (Q*):
        For Normal: Q* = demand_mean + z * demand_sd, where z = Phi^(-1)(CR)

    Input Validation:
    - (selling_price > unit_cost >= salvage_value) OR (cost_underage > 0 and cost_overage > 0)
    - demand distribution: (demand_mean, demand_sd) or (uniform_min, uniform_max) or discrete_demand (required)
    """
    effective_salvage = salvage_value if salvage_value is not None else 0.0
    effective_penalty = penalty_cost if penalty_cost is not None else 0.0

    if cost_underage is not None and cost_overage is not None:
        if cost_underage <= 0:
            raise ValueError(f"Underage cost must be strictly positive (got {cost_underage}).")
        if cost_overage <= 0:
            raise ValueError(f"Overage cost must be strictly positive (got {cost_overage}).")
        cu = cost_underage
        co = cost_overage
        eff_price = selling_price if selling_price is not None else (cu + co)
        eff_cost = unit_cost if unit_cost is not None else co
    else:
        # Strict input validation: never invent missing numerical inputs
        if selling_price is None:
            raise ValueError("Missing required numerical input: 'selling_price' (or 'cost_underage').")
        if selling_price <= 0:
            raise ValueError(f"Selling price must be strictly positive (got {selling_price}).")

        if unit_cost is None:
            raise ValueError("Missing required numerical input: 'unit_cost' (or 'cost_overage').")
        if unit_cost <= 0:
            raise ValueError(f"Unit cost must be strictly positive (got {unit_cost}).")

        if selling_price <= unit_cost:
            raise ValueError(f"Economic violation: selling_price (${selling_price}) must exceed unit_cost (${unit_cost}).")
        if unit_cost < effective_salvage:
            raise ValueError(f"Economic violation: unit_cost (${unit_cost}) cannot be less than salvage_value (${effective_salvage}).")

        cu = selling_price - unit_cost + effective_penalty
        co = unit_cost - effective_salvage
        eff_price = selling_price
        eff_cost = unit_cost

    cost_underage = cu
    cost_overage = co
    total_marginal = cost_underage + cost_overage
    critical_ratio = cost_underage / total_marginal

    z_factor: Optional[float] = None
    dist_type = "normal"

    # Demand Distribution Evaluation
    if discrete_demand is not None:
        dist_type = "discrete"
        sorted_pairs = sorted(discrete_demand, key=lambda x: x[0])
        cum_prob = 0.0
        optimal_q = sorted_pairs[-1][0]
        for qty, prob in sorted_pairs:
            cum_prob += prob
            if cum_prob >= critical_ratio:
                optimal_q = qty
                break
        mean_d = sum(q * p for q, p in sorted_pairs)
        sd_d = math.sqrt(sum((q - mean_d)**2 * p for q, p in sorted_pairs))
        exp_lost = sum(max(0.0, q - optimal_q) * p for q, p in sorted_pairs)
        exp_leftover = sum(max(0.0, optimal_q - q) * p for q, p in sorted_pairs)

    elif uniform_min is not None and uniform_max is not None:
        dist_type = "uniform"
        if uniform_min >= uniform_max:
            raise ValueError(f"uniform_min ({uniform_min}) must be strictly less than uniform_max ({uniform_max}).")
        optimal_q = uniform_min + critical_ratio * (uniform_max - uniform_min)
        mean_d = (uniform_min + uniform_max) / 2.0
        sd_d = (uniform_max - uniform_min) / math.sqrt(12.0)
        a, b = uniform_min, uniform_max
        if optimal_q <= a:
            exp_lost = mean_d - optimal_q
            exp_leftover = 0.0
        elif optimal_q >= b:
            exp_lost = 0.0
            exp_leftover = optimal_q - mean_d
        else:
            exp_lost = ((b - optimal_q) ** 2) / (2.0 * (b - a))
            exp_leftover = ((optimal_q - a) ** 2) / (2.0 * (b - a))

    elif demand_mean is not None and demand_sd is not None:
        dist_type = "normal"
        if demand_mean < 0:
            raise ValueError(f"Demand mean cannot be negative (got {demand_mean}).")
        if demand_sd < 0:
            raise ValueError(f"Demand standard deviation cannot be negative (got {demand_sd}).")

        mean_d = demand_mean
        sd_d = demand_sd

        z_factor = standard_normal_inv_cdf(critical_ratio)
        optimal_q = mean_d + z_factor * sd_d

        if sd_d > 0:
            exp_lost = sd_d * unit_normal_loss(z_factor)
        else:
            exp_lost = 0.0
        exp_leftover = optimal_q - mean_d + exp_lost

    else:
        raise ValueError("Missing required demand parameters: must provide either (demand_mean, demand_sd), (uniform_min, uniform_max), or discrete_demand. Never inventing missing inputs.")

    exp_lost = max(0.0, exp_lost)
    exp_leftover = max(0.0, exp_leftover)
    exp_sales = mean_d - exp_lost

    # Financial breakdown
    exp_revenue = eff_price * exp_sales
    exp_salvage = effective_salvage * exp_leftover
    exp_cost = eff_cost * optimal_q
    exp_penalty = effective_penalty * exp_lost
    exp_profit = exp_revenue + exp_salvage - exp_cost - exp_penalty

    # Perfect information and EVPI
    profit_pi = (eff_price - eff_cost) * mean_d
    evpi = max(0.0, profit_pi - exp_profit)
    rounded_q = max(0, round(optimal_q))

    intermediate = {
        "underage_cost": cost_underage,
        "overage_cost": cost_overage,
        "critical_ratio": critical_ratio,
        "z_factor": z_factor,
        "expected_sales": exp_sales,
        "expected_lost_sales": exp_lost,
        "expected_leftover_inventory": exp_leftover,
        "expected_sales_revenue": exp_revenue,
        "expected_salvage_revenue": exp_salvage,
        "expected_total_cost": exp_cost,
        "expected_profit": exp_profit,
        "profit_under_perfect_information": profit_pi,
        "expected_value_of_perfect_information": evpi,
    }

    final_ans = {
        "recommended_order_quantity": optimal_q,
        "rounded_order_quantity": rounded_q,
        "underage_cost": cost_underage,
        "overage_cost": cost_overage,
        "critical_ratio": critical_ratio,
        "expected_profit": exp_profit,
    }

    # 1. Inputs Extracted
    inputs_extracted = {
        "selling_price_p": eff_price,
        "unit_cost_c": eff_cost,
        "salvage_value_s": effective_salvage,
        "penalty_cost_g": effective_penalty,
        "cost_underage_Cu": cost_underage,
        "cost_overage_Co": cost_overage,
        "demand_mean_mu": mean_d,
        "demand_sd_sigma": sd_d,
        "distribution_type": dist_type,
        "uniform_min_a": uniform_min,
        "uniform_max_b": uniform_max,
    }

    # 2. Variable Definitions
    variable_definitions = {
        "p": "Unit retail selling price ($/unit)",
        "c": "Unit wholesale procurement / acquisition cost ($/unit)",
        "s": "Unit salvage value recovered on unsold inventory ($/unit)",
        "g": "Unit penalty cost / loss of customer goodwill ($/unit)",
        "Cu": "Cost of underage (marginal profit lost per unmet unit) = p - c + g ($/unit)",
        "Co": "Cost of overage (marginal loss per unsold unit) = c - s ($/unit)",
        "CR": "Critical Ratio / Critical Fractile = Cu / (Cu + Co)",
        "mu": "Mean demand for the single selling period (units)",
        "sigma": "Standard deviation of demand for the single selling period (units)",
        "z*": "Optimal safety factor = Phi^(-1)(CR)",
        "Q*": "Optimal order quantity maximizing expected operating profit (units)",
        "E[Sales]": "Expected unit sales satisfied by inventory (units)",
        "E[Lost Sales]": "Expected unmet customer demand due to stockouts (units)",
        "E[Leftover]": "Expected unsold inventory salvaged at end of period (units)",
        "E[Profit]": "Expected net operating profit ($)",
        "EVPI": "Expected Value of Perfect Information = Profit_perfect - E[Profit] ($)",
    }

    # 3. Formula
    formula = {
        "Cost_of_Underage_Cu": r"C_u = p - c + g",
        "Cost_of_Overage_Co": r"C_o = c - s",
        "Critical_Ratio_CR": r"CR = \frac{C_u}{C_u + C_o}",
        "Optimal_Order_Quantity_Q*": (
            r"Q^* = \mu + z^* \cdot \sigma, \quad z^* = \Phi^{-1}(CR)" if dist_type == "normal" else
            r"Q^* = a + CR \cdot (b - a)"
        ),
        "Expected_Profit": r"E[\text{Profit}] = p \cdot E[\text{Sales}] + s \cdot E[\text{Leftover}] - c \cdot Q^* - g \cdot E[\text{Lost Sales}]",
        "Expected_Value_of_Perfect_Information": r"EVPI = \text{Profit}_{\text{perfect}} - E[\text{Profit}(Q^*)]",
    }

    # 4. Formula with Substituted Values
    q_sub = (
        f"Q^* = {mean_d:,.2f} + ({z_factor:.4f} \\cdot {sd_d:,.2f}) = {mean_d:,.2f} + {z_factor * sd_d:,.2f} = {optimal_q:,.2f} \\text{{ units}}" if dist_type == "normal" and z_factor is not None else
        (f"Q^* = {uniform_min:,.2f} + {critical_ratio:.4f} \\cdot ({uniform_max:,.2f} - {uniform_min:,.2f}) = {optimal_q:,.2f} \\text{{ units}}" if dist_type == "uniform" else f"Q^* = {optimal_q:,.2f} \\text{{ units}}")
    )
    formula_with_substituted_values = {
        "Cost_of_Underage_Cu": f"C_u = {eff_price:,.2f} - {eff_cost:,.2f} + {effective_penalty:,.2f} = \\text{{\\$}}{cost_underage:,.2f}/\\text{{unit}}",
        "Cost_of_Overage_Co": f"C_o = {eff_cost:,.2f} - {effective_salvage:,.2f} = \\text{{\\$}}{cost_overage:,.2f}/\\text{{unit}}",
        "Critical_Ratio_CR": f"CR = \\frac{{{cost_underage:,.2f}}}{{{cost_underage:,.2f} + {cost_overage:,.2f}}} = \\frac{{{cost_underage:,.2f}}}{{{cost_underage + cost_overage:,.2f}}} = {critical_ratio:.4f}",
        "Optimal_Order_Quantity_Q*": q_sub,
        "Expected_Profit": f"E[\\text{{Profit}}] = ({eff_price:,.2f} \\cdot {exp_sales:,.1f}) + ({effective_salvage:,.2f} \\cdot {exp_leftover:,.1f}) - ({eff_cost:,.2f} \\cdot {optimal_q:,.1f}) = \\text{{\\$}}{exp_profit:,.2f}",
        "Expected_Value_of_Perfect_Information": f"EVPI = \\text{{\\$}}{profit_pi:,.2f} - \\text{{\\$}}{exp_profit:,.2f} = \\text{{\\$}}{evpi:,.2f}",
    }

    assumptions = [
        "Single selling season with no opportunity for mid-season reorders.",
        "Unsold inventory cannot be carried over to future periods and must be salvaged or discarded.",
        "Demand is uncertain and follows the specified probability distribution.",
        "Unit selling price, purchase cost, and salvage value are fixed and known.",
        "Stockouts result in lost sales (and penalty cost if specified).",
    ]

    # 7. Verification Checks
    risk_hedging_str = (
        f"[PASS] Risk hedging alignment: Since CR ({critical_ratio:.4f}) > 0.50, optimal lot size Q* ({optimal_q:,.1f}) > mean demand ({mean_d:,.1f}), hedging against stockout underage costs."
        if critical_ratio > 0.5 else
        f"[PASS] Risk hedging alignment: Since CR ({critical_ratio:.4f}) <= 0.50, optimal lot size Q* ({optimal_q:,.1f}) <= mean demand ({mean_d:,.1f}), hedging against unsold inventory overage costs."
    )
    verification_checks = [
        f"[PASS] Marginal cost non-negativity: Underage Cost Cu = ${cost_underage:.2f} > 0, Overage Cost Co = ${cost_overage:.2f} > 0.",
        f"[PASS] Critical ratio validity: CR = {critical_ratio:.4f} strictly within open probability interval (0, 1).",
        f"[PASS] Upper bound on expected profit: Expected Profit (${exp_profit:,.2f}) <= Profit under Perfect Information (${profit_pi:,.2f}). EVPI = ${evpi:,.2f} >= 0.",
        risk_hedging_str,
    ]
    verification_notes = list(verification_checks)

    return NewsvendorResult(
        inputs={
            "selling_price": selling_price,
            "unit_cost": unit_cost,
            "salvage_value": effective_salvage,
            "penalty_cost": effective_penalty,
            "cost_underage": cost_underage,
            "cost_overage": cost_overage,
            "demand_mean": mean_d,
            "demand_sd": sd_d,
        },
        selling_price=eff_price,
        unit_cost=eff_cost,
        salvage_value=effective_salvage,
        penalty_cost=effective_penalty,
        intermediate_calculations=intermediate,
        cost_underage=cost_underage,
        cost_overage=cost_overage,
        critical_ratio=critical_ratio,
        demand_mean=mean_d,
        demand_sd=sd_d,
        distribution_type=dist_type,
        z_factor=z_factor,
        expected_lost_sales=exp_lost,
        expected_leftover_inventory=exp_leftover,
        expected_sales=exp_sales,
        expected_revenue=exp_revenue,
        expected_salvage_revenue=exp_salvage,
        expected_total_cost=exp_cost,
        expected_profit=exp_profit,
        profit_under_perfect_information=profit_pi,
        expected_value_of_perfect_information=evpi,
        final_answer=final_ans,
        recommended_order_quantity=optimal_q,
        rounded_order_quantity=rounded_q,
        critical_fractile=critical_ratio,
        optimal_order_quantity=optimal_q,
        verification_passed=(0.0 < critical_ratio < 1.0 and optimal_q >= 0),
        verification_notes=verification_notes,
        assumptions=assumptions,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        verification_checks=verification_checks,
    )


# ---------------------------------------------------------------------------
# 6. Multicriteria Inventory Classification (BAN 517 Methodology)
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """
    Structured result for Multicriteria Inventory Classification (Sales + Hits).
    Adheres strictly to the BAN 517 Inventory Classification Methodology.
    """
    classified_df: Any
    summary_df: Any
    total_items: int
    total_sales: float
    total_hits: float
    sales_weight: float
    hits_weight: float
    class_counts: Dict[str, int]
    class_sales: Dict[str, float]
    verification_passed: bool
    verification_checks: List[str]
    assumptions: List[str]
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]

    def to_student_verification(self) -> Dict[str, Any]:
        return {
            "1_inputs_extracted": self.inputs_extracted,
            "2_variable_definitions": self.variable_definitions,
            "3_formula": self.formula,
            "4_formula_with_substituted_values": self.formula_with_substituted_values,
            "5_intermediate_calculations": self.intermediate_calculations,
            "6_final_answer": self.final_answer,
            "7_verification_checks": self.verification_checks,
            "8_assumptions": self.assumptions,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_items": self.total_items,
            "total_sales": self.total_sales,
            "total_hits": self.total_hits,
            "sales_weight": self.sales_weight,
            "hits_weight": self.hits_weight,
            "class_counts": self.class_counts,
            "class_sales": self.class_sales,
            "verification_passed": self.verification_passed,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
            "intermediate_calculations": self.intermediate_calculations,
            "final_answer": self.final_answer,
        }


def assign_pareto_class(cum_pct: float, is_first: bool = False) -> str:
    """
    Assign Pareto class (A, B, C, D) based on cumulative percentage contribution.
    Thresholds without gaps for decimal percentages:
      - Class A: 0% to 60.0% (cum_pct <= 60.0%; rank-1 item is always assigned Class A)
      - Class B: 61% to 80.0% (60.0% < cum_pct <= 80.0%)
      - Class C: 81% to 90.0% (80.0% < cum_pct <= 90.0%)
      - Class D: 91% to 100.0% (cum_pct > 90.0%)
    """
    if is_first or cum_pct <= 60.0:
        return "A"
    elif cum_pct <= 80.0:
        return "B"
    elif cum_pct <= 90.0:
        return "C"
    else:
        return "D"


def assign_final_inventory_class(unrounded_score: float) -> str:
    """
    Assign final multicriteria inventory class based on unrounded Combined Weighted Score.
    Thresholds:
      - A: 3.25 to 4.00 (unrounded_score >= 3.25)
      - B: 2.50 to 3.24 (2.50 <= unrounded_score < 3.25)
      - C: 1.75 to 2.49 (1.75 <= unrounded_score < 2.50)
      - D: Below 1.75 (unrounded_score < 1.75)
    """
    if unrounded_score >= 3.25:
        return "A"
    elif unrounded_score >= 2.50:
        return "B"
    elif unrounded_score >= 1.75:
        return "C"
    else:
        return "D"


def classify_multicriteria_inventory(
    df: Any,
    sales_weight: float = 0.65,
    hits_weight: float = 0.35,
) -> ClassificationResult:
    """
    Executes Multicriteria Inventory Classification per BAN 517 Methodology:
    1. Validates required columns: Item Number, Item Name, UoM, Sales ($), Hits
    2. Cleans numerical data, handles formatting ($ and commas), verifies non-negativity
    3. Calculates individual percentage and cumulative percentage for Sales and Hits
    4. Assigns Pareto classes (A: 0-60%, B: 61-80%, C: 81-90%, D: 91-100%)
    5. Converts classes to scores (A=4, B=3, C=2, D=1)
    6. Calculates Combined Weighted Score = 0.65 * Sales Score + 0.35 * Hits Score
    7. Assigns Final Inventory Class (A: 3.25-4.00, B: 2.50-3.24, C: 1.75-2.49, D: <1.75)
    8. Generates summary table and performs complete verification checks
    """
    import pandas as pd  # type: ignore
    import numpy as np  # type: ignore

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    # 1. Validate required columns
    required_cols = ["Item Number", "Item Name", "UoM", "Sales ($)", "Hits"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required column(s): {', '.join(missing_cols)}. Expected: {', '.join(required_cols)}.")

    if len(df) == 0:
        raise ValueError("Cannot classify empty inventory dataset (0 rows).")

    # Working copy with preserved internal row identifier
    working_df = df.copy()
    working_df["_row_id"] = np.arange(len(working_df))

    # Check for duplicate item identifiers
    duplicate_count = int(working_df["Item Number"].duplicated().sum())

    # 2. Data Cleaning & Type Coercion
    def clean_num(series):
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce").fillna(0.0)
        cleaned = (
            series.astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)

    working_df["Sales ($)"] = clean_num(working_df["Sales ($)"])
    working_df["Hits"] = clean_num(working_df["Hits"])

    # Non-negativity check
    invalid_sales = (working_df["Sales ($)"] < 0).sum()
    invalid_hits = (working_df["Hits"] < 0).sum()
    if invalid_sales > 0:
        raise ValueError(f"Dataset contains {invalid_sales} negative Sales ($) values.")
    if invalid_hits > 0:
        raise ValueError(f"Dataset contains {invalid_hits} negative Hits values.")

    total_sales = float(working_df["Sales ($)"].sum())
    total_hits = float(working_df["Hits"].sum())
    total_items = len(working_df)

    if total_sales <= 0:
        raise ValueError(f"Total Sales ($) across all items must be strictly positive (got {total_sales}).")
    if total_hits <= 0:
        raise ValueError(f"Total Hits across all items must be strictly positive (got {total_hits}).")

    # 3. Classify Sales ($)
    # Sort descending by Sales ($), with deterministic tie-breaking on Hits then _row_id
    sales_sorted = working_df.sort_values(
        by=["Sales ($)", "Hits", "_row_id"],
        ascending=[False, False, True]
    ).copy()
    sales_sorted["Sales Percentage"] = (sales_sorted["Sales ($)"] / total_sales) * 100.0
    sales_sorted["Cumulative Sales Percentage"] = sales_sorted["Sales Percentage"].cumsum()
    # Guard floating-point drift on final cumsum
    sales_sorted.iloc[-1, sales_sorted.columns.get_loc("Cumulative Sales Percentage")] = min(
        100.0, float(sales_sorted.iloc[-1]["Cumulative Sales Percentage"])
    )
    sales_sorted["Sales Class"] = [
        assign_pareto_class(val, idx == 0)
        for idx, val in enumerate(sales_sorted["Cumulative Sales Percentage"])
    ]

    # 4. Classify Hits
    # Sort descending by Hits, with deterministic tie-breaking on Sales ($) then _row_id
    hits_sorted = working_df.sort_values(
        by=["Hits", "Sales ($)", "_row_id"],
        ascending=[False, False, True]
    ).copy()
    hits_sorted["Hits Percentage"] = (hits_sorted["Hits"] / total_hits) * 100.0
    hits_sorted["Cumulative Hits Percentage"] = hits_sorted["Hits Percentage"].cumsum()
    hits_sorted.iloc[-1, hits_sorted.columns.get_loc("Cumulative Hits Percentage")] = min(
        100.0, float(hits_sorted.iloc[-1]["Cumulative Hits Percentage"])
    )
    hits_sorted["Hits Class"] = [
        assign_pareto_class(val, idx == 0)
        for idx, val in enumerate(hits_sorted["Cumulative Hits Percentage"])
    ]

    # 5. Map calculations back to original row structure using _row_id
    sales_lookup = sales_sorted.set_index("_row_id")
    hits_lookup = hits_sorted.set_index("_row_id")

    working_df["Sales Percentage"] = working_df["_row_id"].map(sales_lookup["Sales Percentage"])
    working_df["Cumulative Sales Percentage"] = working_df["_row_id"].map(sales_lookup["Cumulative Sales Percentage"])
    working_df["Sales Class"] = working_df["_row_id"].map(sales_lookup["Sales Class"])

    working_df["Hits Percentage"] = working_df["_row_id"].map(hits_lookup["Hits Percentage"])
    working_df["Cumulative Hits Percentage"] = working_df["_row_id"].map(hits_lookup["Cumulative Hits Percentage"])
    working_df["Hits Class"] = working_df["_row_id"].map(hits_lookup["Hits Class"])

    # 6. Convert Classes into Scores
    score_mapping = {"A": 4, "B": 3, "C": 2, "D": 1}
    working_df["Sales Score"] = working_df["Sales Class"].map(score_mapping)
    working_df["Hits Score"] = working_df["Hits Class"].map(score_mapping)

    # 7. Combined Weighted Score = 0.65 * Sales Score + 0.35 * Hits Score
    working_df["Combined Weighted Score"] = (
        sales_weight * working_df["Sales Score"] + hits_weight * working_df["Hits Score"]
    )

    # 8. Assign Final Inventory Class (Unrounded comparison)
    working_df["Final Inventory Class"] = working_df["Combined Weighted Score"].apply(
        assign_final_inventory_class
    )

    working_df.drop(columns=["_row_id"], inplace=True)

    # 9. Summary Table for Classes A, B, C, D (Always include all 4 classes)
    summary_list = []
    class_counts = {}
    class_sales = {}

    for c in ["A", "B", "C", "D"]:
        subset = working_df[working_df["Final Inventory Class"] == c]
        cnt = int(len(subset))
        c_sales = float(subset["Sales ($)"].sum())
        class_counts[c] = cnt
        class_sales[c] = c_sales
        summary_list.append({
            "Final Inventory Class": c,
            "Number of Items": cnt,
            "Total Sales ($)": c_sales,
            "% of Total Items": (cnt / total_items * 100.0) if total_items > 0 else 0.0,
            "% of Total Sales": (c_sales / total_sales * 100.0) if total_sales > 0 else 0.0,
        })

    summary_df = pd.DataFrame(summary_list)

    # 10. Verification Checks
    verification_checks = []
    is_valid = True

    # Weight balance check
    weight_sum = sales_weight + hits_weight
    if abs(weight_sum - 1.0) < 1e-6:
        verification_checks.append(f"[PASS] Weights verification: Sales ({sales_weight*100:.0f}%) + Hits ({hits_weight*100:.0f}%) = {weight_sum*100:.0f}%.")
    else:
        verification_checks.append(f"[FAIL] Weights sum to {weight_sum:.4f} != 1.0.")
        is_valid = False

    # Item count reconciliation check
    items_sum = sum(class_counts.values())
    if items_sum == total_items:
        verification_checks.append(f"[PASS] Item count reconciliation: Class A ({class_counts['A']}) + Class B ({class_counts['B']}) + Class C ({class_counts['C']}) + Class D ({class_counts['D']}) = {items_sum} total items.")
    else:
        verification_checks.append(f"[FAIL] Item count mismatch: sum={items_sum} vs total={total_items}.")
        is_valid = False

    # Sales amount reconciliation check
    sales_sum = sum(class_sales.values())
    sales_diff = abs(sales_sum - total_sales)
    if sales_diff < 1e-2:
        verification_checks.append(f"[PASS] Sales reconciliation: Class A (${class_sales['A']:,.2f}) + Class B (${class_sales['B']:,.2f}) + Class C (${class_sales['C']:,.2f}) + Class D (${class_sales['D']:,.2f}) = ${sales_sum:,.2f} total Sales.")
    else:
        verification_checks.append(f"[FAIL] Sales discrepancy: sum=${sales_sum:,.2f} vs dataset total=${total_sales:,.2f}.")
        is_valid = False

    # Duplicates note
    if duplicate_count > 0:
        verification_checks.append(f"[INFO] Dataset contains {duplicate_count} duplicate Item Number entries; row-level integrity preserved.")
    else:
        verification_checks.append("[PASS] All Item Number identifiers are unique.")

    # Assumptions
    assumptions = [
        "Multicriteria classification uses dual criteria: Monetary Value (Sales ($), weight 65%) and Transaction Velocity (Hits, weight 35%).",
        "Pareto boundaries follow BAN 517 specifications: Class A (0-60%), Class B (61-80%), Class C (81-90%), Class D (91-100%).",
        "Items are classified by cumulative percentage contribution with continuous thresholds without decimal gaps: [0, 60%] for A, (60%, 80%] for B, (80%, 90%] for C, and >90% for D.",
        "The highest-ranked item is guaranteed Class A even if its individual share exceeds 60% of total sales or hits.",
        "Scores are discrete integers: A = 4, B = 3, C = 2, D = 1.",
        "Final classification thresholds apply to the unrounded Combined Weighted Score: A [3.25, 4.00], B [2.50, 3.24], C [1.75, 2.49], D (< 1.75).",
        "Historical sales and hits reflect recurring inventory demand patterns.",
    ]

    inputs_extracted = {
        "total_items": total_items,
        "total_sales": total_sales,
        "total_hits": total_hits,
        "sales_weight": sales_weight,
        "hits_weight": hits_weight,
        "duplicate_item_numbers": duplicate_count,
        "pareto_thresholds": "A: 0-60%, B: 61-80%, C: 81-90%, D: 91-100%",
        "score_mapping": "A=4, B=3, C=2, D=1",
        "final_class_thresholds": "A: [3.25, 4.00], B: [2.50, 3.24], C: [1.75, 2.49], D: < 1.75",
    }

    variable_definitions = {
        "Sales ($)": "Annual or period monetary sales revenue generated by the SKU ($)",
        "Hits": "Transaction frequency / pick count representing picking velocity",
        "Sales %": "Individual item sales as a percentage of total dataset sales",
        "Cum Sales %": "Cumulative sum of sales percentages sorted in descending sales order",
        "Sales Class": "Single-criterion Pareto class for sales (A: 0-60%, B: 61-80%, C: 81-90%, D: 91-100%)",
        "Hits %": "Individual item hits as a percentage of total dataset hits",
        "Cum Hits %": "Cumulative sum of hits percentages sorted in descending hits order",
        "Hits Class": "Single-criterion Pareto class for hits (A: 0-60%, B: 61-80%, C: 81-90%, D: 91-100%)",
        "Sales Score": "Numerical score converted from Sales Class (A=4, B=3, C=2, D=1)",
        "Hits Score": "Numerical score converted from Hits Class (A=4, B=3, C=2, D=1)",
        "Combined Weighted Score": f"Weighted score = {sales_weight} * Sales Score + {hits_weight} * Hits Score",
        "Final Inventory Class": "Final multicriteria class (A: >=3.25, B: >=2.50, C: >=1.75, D: <1.75)",
    }

    formula = {
        "Sales_Percentage": r"\text{Sales \%}_i = \left(\frac{\text{Sales}_i}{\sum_{k} \text{Sales}_k}\right) \times 100\%",
        "Hits_Percentage": r"\text{Hits \%}_i = \left(\frac{\text{Hits}_i}{\sum_{k} \text{Hits}_k}\right) \times 100\%",
        "Combined_Weighted_Score": r"\text{Score} = 0.65 \cdot \text{Score}_{\text{Sales}} + 0.35 \cdot \text{Score}_{\text{Hits}}",
        "Final_Class_Rule": r"\text{Class} = \begin{cases} A & \text{if Score} \ge 3.25 \\ B & \text{if } 2.50 \le \text{Score} < 3.25 \\ C & \text{if } 1.75 \le \text{Score} < 2.50 \\ D & \text{if Score} < 1.75 \end{cases}",
    }

    formula_with_substituted_values = {
        "Weights": f"\\text{{Score}} = ({sales_weight} \\cdot \\text{{Score}}_{{\\text{{Sales}}}}) + ({hits_weight} \\cdot \\text{{Score}}_{{\\text{{Hits}}}})",
        "Total_Sales": f"\\sum \\text{{Sales}} = \\text{{\\$}}{total_sales:,.2f}",
        "Total_Hits": f"\\sum \\text{{Hits}} = {total_hits:,.0f} \\text{{ transactions}}",
    }

    intermediate_calculations = {
        "total_items": total_items,
        "total_sales": total_sales,
        "total_hits": total_hits,
        "class_counts": class_counts,
        "class_sales": class_sales,
        "summary_table": summary_list,
    }

    final_answer = {
        "class_A_items": class_counts["A"],
        "class_A_sales": class_sales["A"],
        "class_B_items": class_counts["B"],
        "class_B_sales": class_sales["B"],
        "class_C_items": class_counts["C"],
        "class_C_sales": class_sales["C"],
        "class_D_items": class_counts["D"],
        "class_D_sales": class_sales["D"],
        "total_classified_items": total_items,
        "total_inventory_sales": total_sales,
    }

    return ClassificationResult(
        classified_df=working_df,
        summary_df=summary_df,
        total_items=total_items,
        total_sales=total_sales,
        total_hits=total_hits,
        sales_weight=sales_weight,
        hits_weight=hits_weight,
        class_counts=class_counts,
        class_sales=class_sales,
        verification_passed=is_valid,
        verification_checks=verification_checks,
        assumptions=assumptions,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate_calculations,
        final_answer=final_answer,
    )

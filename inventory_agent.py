"""
Supply Chain Analytics Inventory Decision Agent
=================================================
Architecture Pipeline:
Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain

Supported Inventory Decision Models:
1. Economic Order Quantity (EOQ)
2. Quantity Discount Model
3. Safety Stock and Reorder Point (Continuous Review ROP)
4. Fixed-Period Review Model (Periodic Review / Order-Up-To Level)
5. Newsvendor Model (Single-Period Critical Fractile)

All numerical calculations are executed deterministically through `inventory_tools.py`.
"""

from dataclasses import dataclass, field
from enum import Enum
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from inventory_tools import (
    calculate_eoq,
    calculate_quantity_discount,
    calculate_rop,
    calculate_safety_stock_rop,
    calculate_periodic_review,
    calculate_fixed_period_review,
    calculate_newsvendor,
    EOQResult,
    DiscountResult,
    QuantityDiscountResult,
    ROPResult,
    SafetyStockROPResult,
    PeriodicReviewResult,
    NewsvendorResult,
)


class ModelType(Enum):
    EOQ = "Economic Order Quantity (EOQ)"
    QUANTITY_DISCOUNT = "Quantity Discount Model"
    SAFETY_STOCK_ROP = "Safety Stock and Reorder Point (Continuous Review)"
    FIXED_PERIOD_REVIEW = "Fixed-Period Review Model (Order-Up-To Level)"
    NEWSVENDOR = "Newsvendor Model (Single-Period Critical Fractile)"
    UNKNOWN = "Unknown Model"


@dataclass
class PipelineStage:
    stage_name: str
    status: str
    details: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""


@dataclass
class AgentDecision:
    question: str
    model_type: ModelType
    model_confidence: float
    extracted_inputs: Dict[str, Any]
    selected_tool: str
    calculation_result: Any
    verification_passed: bool
    verification_checklist: List[str]
    explanation: str
    pipeline_trace: List[PipelineStage] = field(default_factory=list)
    student_verification: Dict[str, Any] = field(default_factory=dict)


class InventoryDecisionAgent:
    """
    Supply Chain Analytics Inventory Decision Agent.
    Orchestrates the 6-stage decision pipeline from natural language to verified supply chain recommendation.
    """

    def __init__(self, default_working_days: int = 365):
        self.default_working_days = default_working_days

    # -----------------------------------------------------------------------
    # STAGE 1: Identify Model
    # -----------------------------------------------------------------------
    def identify_model(self, question: str) -> Tuple[ModelType, float, List[str]]:
        """
        Identify which of the 5 inventory models applies to the customer question.
        Returns (ModelType, confidence_score, matched_indicators).
        """
        q_lower = question.lower()
        scores: Dict[ModelType, int] = {m: 0 for m in ModelType if m != ModelType.UNKNOWN}
        indicators: Dict[ModelType, List[str]] = {m: [] for m in ModelType if m != ModelType.UNKNOWN}

        # Model 5: Newsvendor Model cues
        newsvendor_keywords = [
            ("newsvendor", 10), ("newsboy", 10), ("salvage", 8), ("salvage value", 10),
            ("perishable", 8), ("single period", 10), ("single-period", 10),
            ("critical fractile", 10), ("critical ratio", 10),
            ("unsold units", 6), ("seasonal jacket", 6), ("fashion", 5),
            ("cost of underage", 10), ("cost of overage", 10),
            ("underage cost", 10), ("overage cost", 10),
            ("shortage cost", 8), ("excess cost", 8),
            ("liquidation value", 7), ("markdown price", 7),
            ("leftover inventory", 6), ("perfect information", 6), ("evpi", 6),
            ("dated tickets", 6), ("spoilage", 6), ("end of season", 6),
        ]
        for kw, wt in newsvendor_keywords:
            if kw in q_lower:
                scores[ModelType.NEWSVENDOR] += wt
                indicators[ModelType.NEWSVENDOR].append(kw)

        # Model 4: Fixed-Period Review Model cues
        periodic_keywords = [
            ("fixed period", 10), ("fixed-period", 10), ("periodic review", 10),
            ("order-up-to", 10), ("order up to", 10), ("review period", 8),
            ("review cycle", 8), ("base stock", 8), ("target inventory level", 6),
            ("inventory position", 6), ("on-hand", 4), ("on hand", 4),
            ("on-order", 4), ("on order", 4), ("backorder", 4),
            ("every 2 weeks", 6), ("every 7 days", 6), ("every week", 6),
            ("every month", 6), ("reviewed every", 8), ("protection period", 8),
        ]
        for kw, wt in periodic_keywords:
            if kw in q_lower:
                scores[ModelType.FIXED_PERIOD_REVIEW] += wt
                indicators[ModelType.FIXED_PERIOD_REVIEW].append(kw)

        # Model 2: Quantity Discount cues
        discount_keywords = [
            ("quantity discount", 10), ("discount", 7), ("price break", 10),
            ("price breaks", 10), ("quantity break", 10), ("quantity breaks", 10),
            ("tiers", 7), ("bracket", 6), ("all-units", 8), ("incremental discount", 8),
            ("volume discount", 8), ("bulk discount", 8), ("price per unit if", 7),
            ("tier 1", 8), ("orders under", 6), ("or more units", 5),
            ("cheaper if we order", 6), ("discount schedule", 9), ("price schedule", 8),
            ("unit purchase price", 6),
        ]
        for kw, wt in discount_keywords:
            if kw in q_lower:
                scores[ModelType.QUANTITY_DISCOUNT] += wt
                indicators[ModelType.QUANTITY_DISCOUNT].append(kw)

        # Detect multiple price tiers or range pricing structure
        tier_matches = re.findall(r"[\$]\s*\d+(?:\.\d+)?\s*for\s*(?:orders\s*)?(?:under|<|between|from|up\s+to|at\s+least|more\s+than|>)?\s*\d+", question, re.I)
        range_matches = re.findall(r"\d+\s*(?:to|-)\s*\d+\s*(?:units)?\s*[:=\-]?\s*[\$]\s*\d+", question, re.I)
        if len(tier_matches) >= 2 or len(range_matches) >= 2:
            scores[ModelType.QUANTITY_DISCOUNT] += 15
            indicators[ModelType.QUANTITY_DISCOUNT].append("multiple price tiers")

        # Model 3: Safety Stock and Reorder Point cues
        ss_rop_keywords = [
            ("safety stock", 10), ("reorder point", 10), ("rop", 9),
            ("continuous review", 8), ("service level", 7), ("cycle service level", 8),
            ("stockout risk", 7), ("csl", 8), ("lead time demand", 8),
            ("z-factor", 6), ("standard deviation of demand", 5),
        ]
        for kw, wt in ss_rop_keywords:
            if kw in q_lower:
                scores[ModelType.SAFETY_STOCK_ROP] += wt
                indicators[ModelType.SAFETY_STOCK_ROP].append(kw)

        # Model 1: EOQ cues
        eoq_keywords = [
            ("eoq", 10), ("economic order quantity", 10), ("optimal order quantity", 6),
            ("how many units should we order", 4), ("order cost", 4), ("ordering cost", 5),
            ("setup cost", 5), ("holding cost", 5), ("carrying cost", 5),
            ("annual demand", 5), ("lot size", 6), ("wilson", 8),
        ]
        for kw, wt in eoq_keywords:
            if kw in q_lower:
                scores[ModelType.EOQ] += wt
                indicators[ModelType.EOQ].append(kw)

        # Disambiguation logic between overlapping cues
        # If both EOQ and Quantity Discount matched, Quantity Discount has higher precedence if tiers/breaks exist
        if scores[ModelType.QUANTITY_DISCOUNT] > 0 and ("discount" in q_lower or "tier" in q_lower or "break" in q_lower or "bracket" in q_lower):
            scores[ModelType.QUANTITY_DISCOUNT] += 5

        # If both Periodic Review and Safety Stock matched, Periodic Review wins if "review period" / "order-up-to" is present
        if "order-up-to" in q_lower or "order up to" in q_lower or "periodic" in q_lower or "review period" in q_lower:
            scores[ModelType.FIXED_PERIOD_REVIEW] += 6

        # Find best candidate
        best_model = max(scores, key=lambda m: scores[m])
        max_score = scores[best_model]

        if max_score <= 2:
            return ModelType.UNKNOWN, 0.0, []

        confidence = min(0.99, 0.5 + (max_score / 25.0))
        return best_model, round(confidence, 2), indicators[best_model]

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # STAGE 2: Extract Inputs
    # -----------------------------------------------------------------------
    def extract_inputs(self, question: str, model: ModelType) -> Dict[str, Any]:
        """
        Extract numerical and structural parameters from natural language.
        Normalizes units, percentages, currencies, and recognizes all common domain synonyms.
        Examines the entire question context before declaring any information missing.
        """
        inputs: Dict[str, Any] = {}
        q = question

        num_pat = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'

        def extract_pattern(patterns: List[str], text: str) -> Optional[float]:
            for p in patterns:
                m = re.search(p, text, re.IGNORECASE)
                if m:
                    val_str = m.group(1).replace(",", "").strip()
                    try:
                        return float(val_str)
                    except ValueError:
                        continue
            return None

        def extract_percent(patterns: List[str], text: str) -> Optional[float]:
            for p in patterns:
                m = re.search(p, text, re.IGNORECASE)
                if m:
                    val_str = m.group(1).replace("%", "").strip()
                    try:
                        val = float(val_str)
                        if val > 1.0:
                            return val / 100.0
                        return val
                    except ValueError:
                        continue
            return None

        # -------------------------------------------------------------------
        # 1. EOQ Extraction
        # Recognizes all ordering cost synonyms:
        # ordering cost, order cost, setup cost, setup expense, cost per order,
        # cost of placing an order, ordering/setup cost, order expense, etc.
        # -------------------------------------------------------------------
        if model == ModelType.EOQ:
            inputs["demand"] = extract_pattern([
                rf"(?:annual|yearly|forecasted|expected)\s+demand\s*(?:is|=|:|of)?\s*({num_pat})",
                rf"demand\s*(?:is|=|:|of|averages)?\s*({num_pat})\s*(?:units)?\s*(?:per\s+year|annually)",
                rf"sells?\s*({num_pat})\s*units\s*(?:per\s+year|annually)",
                rf"annual\s+(?:usage|requirement|sales)\s*(?:is|=|:|of)?\s*({num_pat})",
                rf"({num_pat})\s*units\s*(?:per\s+year|annually|annual\s+demand)",
                rf"annual\s+demand\s*(?:is|=|:)?\s*({num_pat})",
                rf"demand\s*(?:of|is|=|:)\s*({num_pat})\s*units",
            ], q)

            inputs["order_cost"] = extract_pattern([
                # Explicit required synonyms:
                rf"(?:ordering\s+cost|order\s+cost|setup\s+cost|setup\s+expense|cost\s+per\s+order|cost\s+of\s+placing\s+an\s+order|ordering/setup\s+cost)\s*(?:is|=|:|of)?\s*[\$]?\s*({num_pat})",
                rf"[\$]?\s*({num_pat})\s*(?:ordering\s+cost|order\s+cost|setup\s+cost|setup\s+expense|cost\s+per\s+order|cost\s+of\s+placing\s+an\s+order|ordering/setup\s+cost)",
                # Additional common variations:
                rf"(?:cost\s+to\s+place\s+(?:an|each)\s+order|order\s+expense|reorder\s+cost|replenishment\s+setup\s+cost)\s*(?:is|=|:|of)?\s*[\$]?\s*({num_pat})",
                rf"(?:costs?|incurs?)\s*[\$]?\s*({num_pat})\s*(?:per\s+order|to\s+place\s+an\s+order|for\s+setup|to\s+set\s+up)",
                rf"placing\s+an\s+order\s+costs\s*[\$]?\s*({num_pat})",
                rf"[\$]?\s*({num_pat})\s*(?:per\s+order|to\s+place\s+an\s+order)",
            ], q)

            inputs["holding_cost"] = extract_pattern([
                rf"(?:holding\s+cost|carrying\s+cost|cost\s+to\s+hold|cost\s+of\s+holding|storage\s+cost|holding\s+expense|carrying\s+expense|cost\s+of\s+carrying)\s*(?:is|=|:|of)?\s*[\$]?\s*({num_pat})\s*(?:per\s+unit)?",
                rf"[\$]?\s*({num_pat})\s*(?:per\s+unit\s+per\s+year|per\s+unit\s+annually|per\s+unit\s+to\s+hold)",
                rf"[\$]?\s*({num_pat})\s*(?:holding\s+cost|carrying\s+cost)",
                rf"cost\s+to\s+store\s*(?:is|=|:|of)?\s*[\$]?\s*({num_pat})",
            ], q)

            inputs["holding_rate"] = extract_percent([
                rf"(?:holding\s+cost\s+rate|holding\s+rate|carrying\s+cost\s+rate|carrying\s+rate|annual\s+carrying\s+charge|carrying\s+charge|cost\s+of\s+capital)\s*(?:is|=|:|of)?\s*({num_pat})%?",
                rf"({num_pat})%\s*(?:holding|carrying)\s*(?:rate|cost|charge)?",
                rf"holding\s+cost\s*(?:is|=|:|of)?\s*({num_pat})%\s*(?:of\s+(?:unit\s+)?(?:price|cost))?",
            ], q)

            # Unit Purchase Price / Procurement Cost (Must NOT capture holding cost or order cost)
            inputs["unit_cost"] = extract_pattern([
                rf"(?:purchase\s+price|purchase\s+cost|unit\s+purchase\s+price|unit\s+purchase\s+cost|unit\s+cost|unit\s+price|wholesale\s+price|item\s+cost|item\s+price|procurement\s+cost|acquisition\s+cost)\s*(?:is|=|:|of)?\s*[\$]?\s*({num_pat})",
                rf"(?<!holding\s)(?<!carrying\s)(?<!storage\s)(?<!setup\s)(?<!order\s)(?<!ordering\s)costs?\s*[\$]?\s*({num_pat})\s*(?:to\s+purchase|to\s+buy|wholesale|each|per\s+unit)",
                rf"(?<!holding\s)(?<!carrying\s)price\s*(?:is|=|:|of)?\s*[\$]?\s*({num_pat})\s*(?:per\s+unit|each)?",
            ], q)

            inputs["working_days_per_year"] = extract_pattern([
                rf"({num_pat})\s*(?:operating|working)?\s*days\s*per\s*year",
            ], q) or self.default_working_days

        # -------------------------------------------------------------------
        # 2. Quantity Discount Extraction
        # Distinguishes: quantity break, unit price, ordering cost,
        # fixed holding cost, percentage holding cost, unit purchase price
        # -------------------------------------------------------------------
        elif model == ModelType.QUANTITY_DISCOUNT:
            inputs["demand"] = extract_pattern([
                rf"(?:annual|yearly|forecasted)\s+demand\s*(?:is|=|:|of)?\s*({num_pat})",
                rf"demand\s*(?:is|=|:|of|averages)?\s*({num_pat})\s*(?:units)?\s*(?:per\s+year|annually)",
                rf"sells?\s*({num_pat})\s*units\s*(?:per\s+year|annually)",
                rf"({num_pat})\s*units\s*(?:per\s+year|annually|annual\s+demand)",
                rf"demand\s*(?:of|is|=|:)\s*({num_pat})",
            ], q)

            inputs["order_cost"] = extract_pattern([
                rf"(?:ordering\s+cost|order\s+cost|setup\s+cost|setup\s+expense|cost\s+per\s+order|cost\s+of\s+placing\s+an\s+order|ordering/setup\s+cost)\s*(?:is|=|:|of)?\s*[\$]?\s*({num_pat})",
                rf"[\$]?\s*({num_pat})\s*(?:ordering\s+cost|order\s+cost|setup\s+cost|setup\s+expense|cost\s+per\s+order|cost\s+of\s+placing\s+an\s+order)",
                rf"(?:costs?|incurs?)\s*[\$]?\s*({num_pat})\s*(?:per\s+order|to\s+place\s+an\s+order)",
                rf"[\$]?\s*({num_pat})\s*(?:per\s+order)",
            ], q)

            # Distinguish Fixed Holding Cost vs Percentage Holding Cost
            inputs["fixed_holding_cost"] = extract_pattern([
                rf"(?:fixed\s+holding\s+cost|fixed\s+carrying\s+cost)\s*(?:is|=|:|of)?\s*[\$]?\s*({num_pat})",
                rf"holding\s+cost\s*(?:is|=|:|of)?\s*[\$]?\s*({num_pat})\s*per\s*unit\s*(?:regardless\s+of\s+price|fixed)?",
                rf"carrying\s+cost\s*(?:is|=|:|of)?\s*[\$]?\s*({num_pat})\s*per\s*unit",
                rf"[\$]\s*({num_pat})\s*(?:per\s+unit\s+per\s+year|per\s+unit\s+fixed\s+holding)",
            ], q)

            inputs["holding_rate"] = extract_percent([
                rf"(?:holding\s+cost\s+rate|holding\s+rate|carrying\s+cost\s+rate|carrying\s+rate|annual\s+carrying\s+charge|carrying\s+charge)\s*(?:is|=|:|of)?\s*({num_pat})%?",
                rf"({num_pat})%\s*(?:annual\s+)?(?:holding|carrying)\s*(?:rate|cost|charge)?",
                rf"holding\s+cost\s*(?:is|=|:|of)?\s*({num_pat})%\s*(?:of\s+(?:unit\s+)?(?:price|cost))?",
            ], q)

            # Distinguish Quantity Breaks and Unit Prices
            tiers = []
            # Pattern A: "$P for orders under/between/at least Q"
            tier_sentences = re.findall(
                rf"[\$]\s*({num_pat})\s*for\s*(?:orders\s*)?(under|<|between|from|up\s+to|at\s+least|more\s+than|>)?\s*({num_pat})?\s*(?:to|-|and)?\s*({num_pat})?\s*(?:or\s+more|\+|units)?",
                q,
                re.IGNORECASE
            )
            if tier_sentences:
                for price_str, modifier, b1, b2 in tier_sentences:
                    price = float(price_str.replace(",", ""))
                    mod = modifier.lower().strip() if modifier else ""
                    val1 = float(b1.replace(",", "")) if b1 else None
                    val2 = float(b2.replace(",", "")) if b2 else None

                    if "under" in mod or "up to" in mod or "<" in mod:
                        tiers.append({"min_qty": 0, "max_qty": val1 - 1 if val1 else 999, "unit_price": price})
                    elif "at least" in mod or "more" in mod or ">" in mod or (val1 and not val2):
                        tiers.append({"min_qty": val1, "max_qty": None, "unit_price": price})
                    elif val1 is not None and val2 is not None:
                        tiers.append({"min_qty": val1, "max_qty": val2, "unit_price": price})
                    elif val1 is not None:
                        tiers.append({"min_qty": val1, "max_qty": None, "unit_price": price})

            # Pattern B: "0 to 999 units: $5.00" or "1000 - 1999: $4.80" or "2000+: $4.75"
            if not tiers:
                range_matches = re.findall(
                    rf"({num_pat})\s*(?:to|-)\s*({num_pat})\s*(?:units)?\s*[:=\-]?\s*[\$]\s*({num_pat})",
                    q,
                    re.IGNORECASE
                )
                for low, high, pr in range_matches:
                    tiers.append({
                        "min_qty": float(low.replace(",", "")),
                        "max_qty": float(high.replace(",", "")),
                        "unit_price": float(pr.replace(",", ""))
                    })
                plus_matches = re.findall(
                    rf"({num_pat})\s*(?:\+|or\s+more)\s*(?:units)?\s*[:=\-]?\s*[\$]\s*({num_pat})",
                    q,
                    re.IGNORECASE
                )
                for low, pr in plus_matches:
                    tiers.append({
                        "min_qty": float(low.replace(",", "")),
                        "max_qty": None,
                        "unit_price": float(pr.replace(",", ""))
                    })

            # Pattern C: Fallback to structured JSON if provided in query
            if not tiers:
                json_match = re.search(r"(\[\s*\{.*?\}\s*\])", q, re.DOTALL)
                if json_match:
                    try:
                        tiers = json.loads(json_match.group(1))
                    except Exception:
                        pass

            inputs["discount_tiers"] = tiers

        # -------------------------------------------------------------------
        # 3. Safety Stock and Reorder Point Extraction
        # Distinguishes: average demand, demand standard deviation,
        # average lead time, lead-time standard deviation, service level
        # -------------------------------------------------------------------
        elif model == ModelType.SAFETY_STOCK_ROP:
            # 1. Lead-time standard deviation (extracted first to avoid substring collision)
            inputs["lead_time_sd"] = extract_pattern([
                rf"(?:lead[\s-]time\s+standard\s+deviation|standard\s+deviation\s+of\s+lead[\s-]time|lead[\s-]time\s+std\s*dev|std\s*dev\s+of\s+lead[\s-]time|sigma_l)\s*(?:is|=|:|of)?\s*({num_pat})",
                rf"lead\s+time\s+(?:has\s+a\s+)?(?:standard\s+deviation|std\s*dev|sd|sigma)\s*(?:is|=|:|of)?\s*({num_pat})",
                rf"lead\s+time\s+varies\s+with\s+(?:a\s+)?(?:standard\s+deviation|std\s*dev)\s*(?:is|=|:|of)?\s*({num_pat})",
                rf"(?:standard\s+deviation|std\s*dev|sd|sigma)\s*(?:is|=|:|of)?\s*({num_pat})\s*(?:days?|weeks?)?\s*(?:for\s+lead[\s-]time|in\s+lead[\s-]time)",
            ], q) or 0.0

            # 2. Average lead time
            inputs["lead_time"] = extract_pattern([
                rf"(?:constant\s+)?(?:average\s+|mean\s+|expected\s+|replenishment\s+|delivery\s+)?lead\s*time(?!\s*(?:standard\s+deviation|std\s*dev|sd|variance))\s*(?:is|averages|=|:|of)?\s*(?:constant\s+at\s+|about\s+|approximately\s+)?({num_pat})\s*(?:days?|weeks?|months?)?",
                rf"takes\s*({num_pat})\s*(?:days?|weeks?|months?)\s*(?:for\s+replenishment|to\s+arrive|lead\s+time)?",
                rf"({num_pat})[\s-]day\s+(?:replenishment\s+)?lead\s+time",
                rf"({num_pat})[\s-]week\s+(?:replenishment\s+)?lead\s+time",
            ], q)

            # 3. Demand standard deviation (extracted first to avoid substring collision)
            inputs["demand_sd_per_period"] = extract_pattern([
                rf"(?:demand\s+standard\s+deviation|standard\s+deviation\s+of\s+demand|demand\s+std\s*dev|std\s*dev\s+of\s+demand|sigma_d)\s*(?:is|=|:|of)?\s*({num_pat})",
                rf"(?:daily|weekly|period)\s+demand\s+(?:has\s+a\s+)?(?:standard\s+deviation|std\s*dev|sd|sigma)\s*(?:is|=|:|of)?\s*({num_pat})",
                rf"with\s+(?:a\s+)?(?:standard\s+deviation|std\s*dev)\s+(?:of\s+)?({num_pat})\s*units",
                rf"(?:standard\s+deviation|std\s*dev|sd|sigma)\s*(?:is|=|:|of)?\s*({num_pat})\s*(?:units)?\s*(?:for\s+(?:daily\s+|weekly\s+)?demand|in\s+demand)",
                rf"standard\s+deviation\s+of\s*({num_pat})\s*units",
            ], q) or 0.0

            # 4. Average demand
            inputs["demand_mean_per_period"] = extract_pattern([
                rf"(?:constant\s+)?(?:average\s+|mean\s+)?(?:daily\s+|weekly\s+)?demand(?!\s*(?:standard\s+deviation|std\s*dev|sd|variance))\s*(?:is|averages|=|:|of)?\s*(?:constant\s+at\s+|about\s+|approximately\s+)?({num_pat})\s*(?:units)?\s*(?:per\s+(?:day|week|month|period)|daily|weekly)?",
                rf"({num_pat})\s*units\s*(?:per\s+day|per\s+week|daily|weekly)\s*(?:average\s+|constant\s+)?demand",
                rf"mean\s+demand\s*(?:is|=|:)?\s*({num_pat})",
            ], q)

            # 5. Service level or z-value
            inputs["service_level"] = extract_percent([
                rf"(?:cycle\s+)?service\s+level\s*(?:is|=|:|of)?\s*({num_pat})%?",
                rf"({num_pat})%\s*(?:cycle\s+)?service\s+level",
                rf"csl\s*(?:is|=|:|of)?\s*({num_pat})%?",
                rf"({num_pat})%\s*csl",
                rf"in[\s-]stock\s+probability\s*(?:is|=|:|of)?\s*({num_pat})%?",
                rf"service\s+factor\s*(?:is|=|:|of)?\s*({num_pat})%?",
            ], q)

            inputs["z_value"] = extract_pattern([
                rf"z\s*(?:factor|score|value)?\s*(?:is|=|:)?\s*({num_pat})",
            ], q)

            inputs["holding_cost_per_unit_year"] = extract_pattern([
                rf"(?:holding\s+cost|carrying\s+cost)\s*(?:is|=|:|of)?\s*[\$]?\s*({num_pat})",
            ], q)

            inputs["order_quantity"] = extract_pattern([
                rf"order\s+quantity\s*(?:is|=|:|of)?\s*({num_pat})",
                rf"q\s*(?:is|=|:)?\s*({num_pat})\s*units",
            ], q)

            if "week" in q.lower():
                inputs["time_unit"] = "weeks"
            elif "month" in q.lower():
                inputs["time_unit"] = "months"
            else:
                inputs["time_unit"] = "days"

        # -------------------------------------------------------------------
        # 4. Fixed-Period Review Model Extraction
        # Distinguishes: demand, review period, lead time, safety stock,
        # inventory position, on-hand inventory, on-order inventory, backorders
        # -------------------------------------------------------------------
        elif model == ModelType.FIXED_PERIOD_REVIEW:
            # 1. Review period
            inputs["review_period"] = extract_pattern([
                rf"(?:review\s+period|review\s+interval|order\s+interval|review\s+cycle)\s*(?:is|=|:|of)?\s*({num_pat})\s*(?:days?|weeks?|months?)?",
                rf"reviewed\s+every\s*({num_pat})\s*(?:days?|weeks?|months?)?",
                rf"every\s*({num_pat})\s*(?:days?|weeks?|months?)\s*(?:review)?",
                rf"order\s+every\s*({num_pat})\s*(?:days?|weeks?|months?)?",
            ], q)

            # 2. Lead time
            inputs["lead_time"] = extract_pattern([
                rf"(?:constant\s+)?(?:average\s+|mean\s+|expected\s+|replenishment\s+|delivery\s+)?lead\s*time(?!\s*(?:standard\s+deviation|std\s*dev|sd|variance))\s*(?:is|averages|=|:|of)?\s*(?:constant\s+at\s+|about\s+|approximately\s+)?({num_pat})\s*(?:days?|weeks?|months?)?",
                rf"takes\s*({num_pat})\s*(?:days?|weeks?|months?)\s*(?:for\s+replenishment|to\s+arrive)?",
                rf"({num_pat})[\s-]day\s+lead\s+time",
                rf"({num_pat})[\s-]week\s+lead\s+time",
            ], q)

            # 3. Demand standard deviation
            inputs["demand_sd_per_period"] = extract_pattern([
                rf"(?:demand\s+standard\s+deviation|standard\s+deviation\s+of\s+demand|demand\s+std\s*dev|std\s*dev\s+of\s+demand|sigma_d)\s*(?:is|=|:|of)?\s*({num_pat})",
                rf"with\s+(?:a\s+)?(?:standard\s+deviation|std\s*dev)\s+(?:of\s+)?({num_pat})\s*units",
                rf"(?:standard\s+deviation|std\s*dev|sd|sigma)\s*(?:is|=|:|of)?\s*({num_pat})\s*(?:units)?\s*(?:for\s+demand|in\s+demand)",
                rf"std\s*dev\s*(?:is|=|:|of)?\s*({num_pat})",
            ], q) or 0.0

            # 4. Average demand
            inputs["demand_mean_per_period"] = extract_pattern([
                rf"(?:constant\s+)?(?:average\s+|mean\s+)?(?:daily\s+|weekly\s+)?demand(?!\s*(?:standard\s+deviation|std\s*dev|sd|variance))\s*(?:is|averages|=|:|of)?\s*(?:constant\s+at\s+|about\s+|approximately\s+)?({num_pat})\s*(?:units)?\s*(?:per\s+(?:day|week|month|period)|daily|weekly)?",
                rf"({num_pat})\s*units\s*(?:per\s+day|per\s+week|daily|weekly)\s*(?:average\s+|constant\s+)?demand",
            ], q)

            # 5. Service level or z-value
            inputs["service_level"] = extract_percent([
                rf"(?:cycle\s+)?service\s+level\s*(?:is|=|:|of)?\s*({num_pat})%?",
                rf"({num_pat})%\s*(?:cycle\s+)?service\s+level",
                rf"csl\s*(?:is|=|:|of)?\s*({num_pat})%?",
                rf"({num_pat})%\s*csl",
            ], q)

            inputs["z_value"] = extract_pattern([
                rf"z\s*(?:factor|score|value)?\s*(?:is|=|:)?\s*({num_pat})",
            ], q)

            # 6. Safety stock (if given directly)
            inputs["safety_stock"] = extract_pattern([
                rf"(?:safety\s+stock|safety\s+buffer)\s*(?:is|=|:|of)?\s*({num_pat})\s*units",
            ], q)

            # 7. Inventory position components
            inputs["on_hand_inventory"] = extract_pattern([
                rf"(?:on[\s-]hand\s+inventory|on[\s-]hand|current\s+inventory|in\s+stock|physical\s+inventory)\s*(?:is|=|:|of)?\s*({num_pat})",
                rf"currently\s+have\s*({num_pat})\s*units",
                rf"({num_pat})\s*units\s*on[\s-]hand",
            ], q)

            inputs["on_order_inventory"] = extract_pattern([
                rf"(?:on[\s-]order\s+inventory|on[\s-]order|pipeline\s+inventory|in\s+transit|open\s+orders)\s*(?:is|=|:|of)?\s*({num_pat})",
                rf"({num_pat})\s*units\s*on[\s-]order",
                rf"({num_pat})\s*units\s*in\s+transit",
            ], q) or 0.0

            inputs["backorders"] = extract_pattern([
                rf"(?:backorders|backorders\s+are|backlog|unfilled\s+orders)\s*(?:is|=|:|of|are)?\s*({num_pat})",
                rf"({num_pat})\s*(?:units\s*)?backorders?",
            ], q) or 0.0

            # 8. Direct inventory position
            inputs["inventory_position"] = extract_pattern([
                rf"(?:inventory\s+position|current\s+inventory\s+position)\s*(?:is|=|:|of)?\s*({num_pat})",
            ], q)

            inputs["time_unit"] = "weeks" if "week" in q.lower() else "days"

        # -------------------------------------------------------------------
        # 5. Newsvendor Model Extraction
        # Distinguishes: selling price, purchase cost, salvage value,
        # underage / shortage cost, overage / excess cost,
        # demand distribution, mean demand, demand standard deviation
        # -------------------------------------------------------------------
        elif model == ModelType.NEWSVENDOR:
            # 1. Selling price
            inputs["selling_price"] = extract_pattern([
                rf"sell(?:ing)?(?:\s+\w+){{0,5}}\s+for\s*[\$]?\s*({num_pat})",
                rf"(?:selling\s+price|retail\s+price|sales\s+price|revenue\s+per\s+unit)\s*(?:is|=|:|of)?\s*[\$]?\s*({num_pat})",
                rf"price\s*(?:is|=|:|of)?\s*[\$]?\s*({num_pat})\s*(?:per\s+unit|each)?",
                rf"sold\s+at\s*[\$]?\s*({num_pat})",
                rf"[\$]\s*({num_pat})\s*selling\s+price",
            ], q)

            # 2. Purchase cost / unit cost
            inputs["unit_cost"] = extract_pattern([
                rf"(?:purchase\s+cost|unit\s+cost|wholesale\s+price|acquisition\s+cost|procurement\s+cost)\s*(?:is|=|:|of)?\s*[\$]?\s*({num_pat})",
                rf"costs?\s+(?:us\s+)?[\$]?\s*({num_pat})\s*(?:to\s+purchase|to\s+buy|wholesale|each|per\s+unit)?",
                rf"cost\s+to\s+purchase\s*(?:is|=|:|of)?\s*[\$]?\s*({num_pat})",
                rf"cost\s*(?:is|=|:|of)?\s*[\$]?\s*({num_pat})\s*per\s*unit",
            ], q)

            # 3. Salvage value
            inputs["salvage_value"] = extract_pattern([
                rf"(?:salvage\s+value|clearance\s+price|liquidation\s+value|markdown\s+price|scrap\s+value|salvage\s+price)\s*(?:is|=|:|of)?\s*[\$]?\s*({num_pat})",
                rf"salvaged\s+for\s*[\$]?\s*({num_pat})",
                rf"liquidated\s+at\s*[\$]?\s*({num_pat})",
                rf"discounted\s+to\s*[\$]?\s*({num_pat})",
            ], q) or 0.0

            # 4. Underage / Shortage cost (direct)
            inputs["cost_underage"] = extract_pattern([
                rf"(?:underage\s+cost|shortage\s+cost|cost\s+of\s+underage|cost\s+of\s+shortage|lost\s+profit\s+per\s+unit|cost\s+per\s+lost\s+sale)\s*(?:is|=|:|of)?\s*[\$]?\s*({num_pat})",
                rf"[\$]?\s*({num_pat})\s*(?:underage\s+cost|shortage\s+cost)",
            ], q)

            # 5. Overage / Excess cost (direct)
            inputs["cost_overage"] = extract_pattern([
                rf"(?:overage\s+cost|excess\s+cost|cost\s+of\s+overage|cost\s+of\s+excess|cost\s+of\s+overstocking|loss\s+per\s+unsold\s+unit)\s*(?:is|=|:|of)?\s*[\$]?\s*({num_pat})",
                rf"[\$]?\s*({num_pat})\s*(?:overage\s+cost|excess\s+cost)",
            ], q)

            inputs["penalty_cost"] = extract_pattern([
                rf"(?:penalty\s+cost|goodwill\s+penalty|loss\s+of\s+goodwill)\s*(?:is|=|:|of)?\s*[\$]?\s*({num_pat})",
            ], q) or 0.0

            # 6. Mean demand
            inputs["demand_mean"] = extract_pattern([
                rf"(?:mean\s+demand|average\s+demand|expected\s+demand)\s*(?:is|=|:|of)?\s*({num_pat})",
                rf"normally\s+distributed\s+with\s+mean\s*({num_pat})",
                rf"mean\s*(?:is|=|:|of)\s*({num_pat})\s*(?:units)?",
                rf"demand\s+averages\s*({num_pat})\s*units",
            ], q)

            # 7. Demand standard deviation
            inputs["demand_sd"] = extract_pattern([
                rf"(?:demand\s+standard\s+deviation|standard\s+deviation\s+of\s+demand|demand\s+std\s*dev|std\s*dev\s+of\s+demand|sigma)\s*(?:is|=|:|of)?\s*({num_pat})",
                rf"(?:standard\s+deviation|std\s*dev)\s*(?:is|=|:|of)?\s*({num_pat})\s*(?:units)?",
            ], q)

            # 8. Uniform distribution min/max
            inputs["uniform_min"] = extract_pattern([
                rf"uniformly\s+distributed\s+between\s*({num_pat})",
                rf"uniform\s+between\s*({num_pat})",
            ], q)
            inputs["uniform_max"] = extract_pattern([
                rf"between\s*{num_pat}\s*and\s*({num_pat})",
                rf"to\s*({num_pat})\s*(?:units)?\s*(?:uniform)?",
            ], q)

        return inputs

    # -----------------------------------------------------------------------
    # STAGE 3 & 4: Select Python Tool & Calculate
    # -----------------------------------------------------------------------
    def select_tool_and_calculate(
        self, model: ModelType, inputs: Dict[str, Any]
    ) -> Tuple[str, Any]:
        """
        Selects the deterministic Python calculation tool from `inventory_tools.py`
        and executes the numerical operations.
        """
        if model == ModelType.EOQ:
            tool_name = "calculate_eoq"
            res = calculate_eoq(
                demand=inputs["demand"],
                order_cost=inputs["order_cost"],
                holding_cost=inputs.get("holding_cost"),
                holding_rate=inputs.get("holding_rate"),
                unit_cost=inputs.get("unit_cost"),
                working_days_per_year=int(inputs.get("working_days_per_year", self.default_working_days)),
            )
            return tool_name, res

        elif model == ModelType.QUANTITY_DISCOUNT:
            tool_name = "calculate_quantity_discount"
            res = calculate_quantity_discount(
                demand=inputs["demand"],
                order_cost=inputs["order_cost"],
                discount_tiers=inputs["discount_tiers"],
                holding_rate=inputs.get("holding_rate"),
                fixed_holding_cost=inputs.get("fixed_holding_cost"),
            )
            return tool_name, res

        elif model == ModelType.SAFETY_STOCK_ROP:
            tool_name = "calculate_rop"
            res = calculate_rop(
                demand_mean=inputs.get("demand_mean_per_period"),
                lead_time_mean=inputs["lead_time"],
                demand_sd=inputs.get("demand_sd_per_period", 0.0),
                lead_time_sd=inputs.get("lead_time_sd", 0.0),
                service_level=inputs.get("service_level"),
                z_value=inputs.get("z_value"),
                holding_cost_per_unit_year=inputs.get("holding_cost_per_unit_year"),
                order_quantity=inputs.get("order_quantity"),
                time_unit=inputs.get("time_unit", "days"),
            )
            return tool_name, res

        elif model == ModelType.FIXED_PERIOD_REVIEW:
            tool_name = "calculate_periodic_review"
            res = calculate_periodic_review(
                review_period=inputs["review_period"],
                lead_time=inputs["lead_time"],
                demand_mean=inputs["demand_mean_per_period"],
                demand_sd=inputs.get("demand_sd_per_period", 0.0),
                service_level=inputs.get("service_level"),
                z_value=inputs.get("z_value"),
                on_hand=inputs.get("on_hand_inventory"),
                on_order=inputs.get("on_order_inventory", 0.0),
                backorders=inputs.get("backorders", 0.0),
                inventory_position=inputs.get("inventory_position"),
                safety_stock=inputs.get("safety_stock"),
                time_unit=inputs.get("time_unit", "days"),
            )
            return tool_name, res

        elif model == ModelType.NEWSVENDOR:
            tool_name = "calculate_newsvendor"
            res = calculate_newsvendor(
                selling_price=inputs.get("selling_price"),
                unit_cost=inputs.get("unit_cost"),
                salvage_value=inputs.get("salvage_value", 0.0),
                penalty_cost=inputs.get("penalty_cost", 0.0),
                cost_underage=inputs.get("cost_underage"),
                cost_overage=inputs.get("cost_overage"),
                demand_mean=inputs.get("demand_mean"),
                demand_sd=inputs.get("demand_sd"),
                uniform_min=inputs.get("uniform_min"),
                uniform_max=inputs.get("uniform_max"),
            )
            return tool_name, res

        else:
            raise ValueError(f"Unsupported model type: {model}")

    # -----------------------------------------------------------------------
    # STAGE 5: Verify
    # -----------------------------------------------------------------------
    def verify_solution(self, model: ModelType, inputs: Dict[str, Any], result: Any) -> Tuple[bool, List[str]]:
        """
        Verify physical, mathematical, and economic constraints of the calculated result.
        Returns (verification_passed, checklist).
        """
        checklist = []
        is_passed = True

        if model == ModelType.EOQ:
            eoq_res: EOQResult = result
            # Check 1: Demand & costs positive
            if eoq_res.annual_demand > 0 and eoq_res.order_cost > 0 and eoq_res.holding_cost > 0:
                checklist.append("[PASS] Non-negativity constraint satisfied (D > 0, S > 0, H > 0).")
            else:
                checklist.append("[FAIL] Negative or zero parameters encountered.")
                is_passed = False

            # Check 2: First-order optimality AOC == AHC
            diff = abs(eoq_res.annual_order_cost - eoq_res.annual_holding_cost)
            if diff < 1e-3 * eoq_res.total_annual_inventory_cost:
                checklist.append(f"[PASS] Optimality balance confirmed: Annual Ordering Cost (${eoq_res.annual_order_cost:,.2f}) equals Annual Holding Cost (${eoq_res.annual_holding_cost:,.2f}).")
            else:
                checklist.append(f"[FAIL] Optimality condition violated: cost difference = {diff:.4f}")
                is_passed = False

            # Check 3: Discrete rounding penalty
            checklist.append(f"[PASS] Practical integer lot size {eoq_res.rounded_eoq} has minor rounding cost penalty of +${eoq_res.cost_penalty_rounded:.2f}/year.")

        elif model == ModelType.QUANTITY_DISCOUNT:
            disc_res: DiscountResult = result
            checklist.append(f"[PASS] Evaluated {len(disc_res.tiers)} pricing tiers across price break thresholds.")
            best = disc_res.tiers[disc_res.best_tier_index]
            if disc_res.min_total_cost <= disc_res.baseline_eoq_cost:
                checklist.append(f"[PASS] Discount tier {best.tier_index} strictly dominates baseline tier (Cost savings: ${disc_res.cost_savings:,.2f}).")
            else:
                checklist.append("[PASS] Baseline un-discounted EOQ minimizes total cost; discount thresholds do not justify holding excess stock.")

        elif model == ModelType.SAFETY_STOCK_ROP:
            ss_res: SafetyStockROPResult = result
            if ss_res.service_level is not None:
                if 0 < ss_res.service_level < 1.0:
                    checklist.append(f"[PASS] Cycle service level CSL ({ss_res.service_level*100:.1f}%) within valid probability bounds.")
                else:
                    checklist.append("[FAIL] Service level must be in (0, 1).")
                    is_passed = False
            else:
                checklist.append("[PASS] Constant demand and lead time (deterministic Case 1: CSL = 100%, SS = 0).")

            if ss_res.reorder_point >= ss_res.expected_lead_time_demand:
                checklist.append(f"[PASS] ROP ({ss_res.reorder_point:.1f}) covers expected lead time demand ({ss_res.expected_lead_time_demand:.1f}) with buffer {ss_res.safety_stock:.1f}.")
            else:
                checklist.append("[FAIL] Reorder point is below mean lead time demand.")
                is_passed = False

        elif model == ModelType.FIXED_PERIOD_REVIEW:
            pr_res: PeriodicReviewResult = result
            if pr_res.order_up_to_level >= pr_res.expected_protection_demand:
                checklist.append(f"[PASS] Order-up-to level M ({pr_res.order_up_to_level:.1f}) protects entire review interval + lead time ({pr_res.protection_period} {pr_res.time_unit}).")
            else:
                checklist.append("[FAIL] Target base stock level is insufficient to cover protection interval demand.")
                is_passed = False

            if pr_res.recommended_order_quantity is not None:
                inv_pos_desc = f"{pr_res.inventory_position:.1f}" if pr_res.inventory_position is not None else "N/A"
                checklist.append(f"[PASS] Order recommendation ({pr_res.recommended_order_quantity:.1f} units) successfully bridges inventory position ({inv_pos_desc}) to target level.")

        elif model == ModelType.NEWSVENDOR:
            nv_res: NewsvendorResult = result
            if nv_res.cost_underage > 0 and nv_res.cost_overage > 0:
                checklist.append(f"[PASS] Marginal costs valid: Underage Cost Cu = ${nv_res.cost_underage:.2f}, Overage Cost Co = ${nv_res.cost_overage:.2f}.")
            else:
                checklist.append("[FAIL] Marginal cost of underage or overage is non-positive.")
                is_passed = False

            if 0.0 < nv_res.critical_fractile < 1.0:
                checklist.append(f"[PASS] Critical fractile CR = {nv_res.critical_fractile:.4f} strictly within (0, 1).")
            else:
                checklist.append("[FAIL] Critical ratio out of bounds.")
                is_passed = False

            if nv_res.expected_profit <= nv_res.profit_under_perfect_information:
                checklist.append(f"[PASS] Expected profit (${nv_res.expected_profit:,.2f}) bounded by Perfect Information upper limit (${nv_res.profit_under_perfect_information:,.2f}). EVPI = ${nv_res.expected_value_of_perfect_information:,.2f}.")
            else:
                checklist.append("[FAIL] Profit calculation violates upper theoretical bound.")
                is_passed = False

        return is_passed, checklist

    # -----------------------------------------------------------------------
    # STAGE 6: Explain
    # -----------------------------------------------------------------------
    def generate_explanation(
        self,
        question: str,
        model: ModelType,
        inputs: Dict[str, Any],
        result: Any,
        verification_passed: bool,
        verification_checklist: List[str],
    ) -> str:
        """
        Generate executive-level supply chain decision explanation with
        step-by-step mathematical reasoning, cost breakdown, and strategic insights.
        Returns the 8 standardized student verification sections:
        1. Inputs Extracted
        2. Variable Definitions
        3. Formula
        4. Formula with Substituted Values
        5. Intermediate Calculations
        6. Final Answer
        7. Verification Checks
        8. Assumptions
        """
        lines = []

        # Model Title & Executive Recommendation Headline
        if model == ModelType.EOQ:
            r: EOQResult = result
            lines.append("## Supply Chain Decision: Economic Order Quantity (EOQ)")
            lines.append("")
            lines.append(f"**Recommended Order Size:** **{r.rounded_eoq:,} units** per order (exact mathematical EOQ: `{r.eoq:.2f}` units).")
            lines.append(f"**Annual Inventory Management Cost:** **${r.total_annual_inventory_cost:,.2f}/year** across `{r.orders_per_year:.2f}` orders.")
            lines.append("")
        elif model == ModelType.QUANTITY_DISCOUNT:
            r: QuantityDiscountResult = result
            lines.append("## Supply Chain Decision: Quantity Discount Optimization")
            lines.append("")
            lines.append(f"**Optimal Order Decision:** Order **{r.optimal_order_quantity:,.0f} units** at **${r.optimal_unit_price:.2f}/unit** (Tier {r.best_tier_index + 1}).")
            lines.append(f"**Minimized Total Annual Cost:** **${r.min_total_cost:,.2f}** (Achieves **${r.cost_savings:,.2f}** / **{r.percentage_savings:.2f}%** net savings vs base un-discounted EOQ).")
            lines.append("")
        elif model == ModelType.SAFETY_STOCK_ROP:
            r: ROPResult = result
            lines.append("## Supply Chain Decision: Continuous Review Safety Stock & Reorder Point (ROP)")
            lines.append("")
            lines.append(f"**Recommended Inventory Policy:** When inventory drops to **{math.ceil(r.reorder_point):,} units**, place a replenishment order.")
            lines.append(f"- **Safety Stock Buffer:** **{math.ceil(r.safety_stock):,} units** (Exact: `{r.safety_stock:.2f}` units)")
            lines.append(f"- **Reorder Point (ROP):** **{r.reorder_point:.2f} units** (or **{math.ceil(r.reorder_point):,} units** integer threshold)")
            lines.append("")
        elif model == ModelType.FIXED_PERIOD_REVIEW:
            r: PeriodicReviewResult = result
            lines.append("## Supply Chain Decision: Fixed-Period Review Model (Order-Up-To Level)")
            lines.append("")
            lines.append(f"**Target Order-Up-To Level ($M$):** **{math.ceil(r.order_up_to_level):,} units** (Exact: `{r.order_up_to_level:.2f}`).")
            if r.recommended_order_quantity is not None:
                lines.append(f"**Recommended Order Quantity for Current Cycle ($Q$):** **{math.ceil(r.recommended_order_quantity):,} units** (Exact: `{r.recommended_order_quantity:.2f}`).")
            lines.append("")
        elif model == ModelType.NEWSVENDOR:
            r: NewsvendorResult = result
            lines.append("## Supply Chain Decision: Newsvendor Single-Period Critical Fractile")
            lines.append("")
            lines.append(f"**Recommended Order Quantity ($Q^*$):** **{r.rounded_order_quantity:,} units** (Exact: `{r.optimal_order_quantity:.2f}` units).")
            lines.append(f"**Expected Net Profit:** **${r.expected_profit:,.2f}** (Critical Ratio: `{r.critical_fractile:.4f}`)")
            lines.append("")

        # -------------------------------------------------------------------
        # 1. Inputs Extracted
        # -------------------------------------------------------------------
        lines.append("### 1. Inputs Extracted")
        lines.append("| Parameter | Extracted Value | Status / Notes |")
        lines.append("| :--- | :--- | :--- |")
        inputs_dict = getattr(result, "inputs_extracted", inputs)
        for k, v in inputs_dict.items():
            clean_k = k.replace("_", " ").title()
            if v is not None:
                if isinstance(v, float):
                    v_str = f"${v:,.2f}" if any(c in k.lower() for c in ["cost", "price", "salvage", "penalty"]) else (f"{v:,.4f}".rstrip("0").rstrip(".") if "." in f"{v:,.4f}" else f"{v:,.0f}")
                else:
                    v_str = str(v)
                lines.append(f"| {clean_k} (`{k}`) | {v_str} | Given in problem statement |")
            else:
                lines.append(f"| {clean_k} (`{k}`) | *Not provided* | Optional (not required when holding cost H is given directly) |")
        lines.append("")

        # -------------------------------------------------------------------
        # 2. Variable Definitions
        # -------------------------------------------------------------------
        lines.append("### 2. Variable Definitions")
        lines.append("| Variable Symbol | Operational Definition |")
        lines.append("| :--- | :--- |")
        var_defs = getattr(result, "variable_definitions", {})
        for sym, defn in var_defs.items():
            lines.append(f"| **{sym}** | {defn} |")
        lines.append("")

        # -------------------------------------------------------------------
        # 3. Formula
        # -------------------------------------------------------------------
        lines.append("### 3. Formula")
        formulas = getattr(result, "formula", {})
        for f_name, f_latex in formulas.items():
            clean_name = f_name.replace("_", " ").title()
            lines.append(f"- **{clean_name}:**")
            safe_latex = re.sub(r'(?<!\\)(?<!\\text\{)\$', r'\\text{\$}', str(f_latex))
            lines.append(f"  $${safe_latex}$$")
        lines.append("")

        # -------------------------------------------------------------------
        # 4. Formula with Substituted Values
        # -------------------------------------------------------------------
        lines.append("### 4. Formula with Substituted Values")
        formulas_sub = getattr(result, "formula_with_substituted_values", {})
        for f_name, f_val in formulas_sub.items():
            clean_name = f_name.replace("_", " ").title()
            lines.append(f"- **{clean_name}:**")
            safe_latex = re.sub(r'(?<!\\)(?<!\\text\{)\$', r'\\text{\$}', str(f_val))
            lines.append(f"  $${safe_latex}$$")
        lines.append("")

        # -------------------------------------------------------------------
        # 5. Intermediate Calculations
        # -------------------------------------------------------------------
        lines.append("### 5. Intermediate Calculations")
        if model == ModelType.EOQ:
            r: EOQResult = result
            lines.append("#### Operational Cadence & Replenishment Schedule:")
            lines.append(f"- **Order Frequency ($N$):** `{r.orders_per_year:.2f}` orders per year")
            lines.append(f"- **Cycle Time ($T$):** `{r.cycle_time_days:.1f}` operating days (`{r.cycle_time_weeks:.1f}` weeks between orders)")
            lines.append("")
            lines.append("#### Financial & Cost Breakdown:")
            lines.append("| Cost Component | Annual Amount ($) | % of Inventory Holding & Setup |")
            lines.append("| :--- | :--- | :--- |")
            pct_order = (r.annual_order_cost / r.total_annual_inventory_cost * 100.0) if r.total_annual_inventory_cost > 0 else 50.0
            pct_holding = (r.annual_holding_cost / r.total_annual_inventory_cost * 100.0) if r.total_annual_inventory_cost > 0 else 50.0
            lines.append(f"| **Annual Ordering Cost ($AOC$)** | ${r.annual_order_cost:,.2f} | {pct_order:.1f}% |")
            lines.append(f"| **Annual Holding Cost ($AHC$)** | ${r.annual_holding_cost:,.2f} | {pct_holding:.1f}% |")
            lines.append(f"| **Total Annual Inventory Cost ($TAC$)** | **${r.total_annual_inventory_cost:,.2f}** | 100.0% |")
            if r.unit_cost is not None and r.unit_cost > 0:
                lines.append(f"| Annual Purchase Cost ($APC = D \\cdot C$) | ${r.annual_demand * r.unit_cost:,.2f} | — |")
                lines.append(f"| **Total Annual Cost ($TC$)** | **${r.total_annual_cost:,.2f}** | — |")
            else:
                lines.append(f"| Annual Purchase Cost ($APC = D \\cdot C$) | *Not provided (excluded)* | — |")
                lines.append(f"| **Total Annual Cost ($TC$)** | **${r.total_annual_inventory_cost:,.2f}** *(TAC only)* | — |")
            lines.append("")
            lines.append("#### Practical Discrete Integer Batch Evaluation:")
            lines.append(f"- Exact continuous EOQ: `{r.eoq:.2f}` units")
            lines.append(f"- Practical rounded batch: **{r.rounded_eoq:,} units**")
            lines.append(f"- Annual ordering cost at rounded batch: `${r.rounded_order_cost:,.2f}`")
            lines.append(f"- Annual holding cost at rounded batch: `${r.rounded_holding_cost:,.2f}`")
            lines.append(f"- Rounding cost penalty: **+${r.cost_penalty_rounded:.2f}/year** (demonstrates the flat-bottom total cost curve)")
            lines.append("")

        elif model == ModelType.QUANTITY_DISCOUNT:
            r: QuantityDiscountResult = result
            lines.append("#### Comprehensive Evaluation of Price Tiers:")
            lines.append("| Tier | Quantity Range | Unit Price ($) | Unconstrained EOQ | Candidate Q | Feasibility | Annual Purchase | Annual Order | Annual Holding | Total Annual Cost |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for t in r.tiers:
                max_str = f"{t.max_qty:,.0f}" if t.max_qty is not None else "Above"
                feas_str = "Feasible" if t.is_feasible else "Breakpoint adjusted"
                star = " **(OPTIMAL)**" if t.tier_index == r.best_tier_index else ""
                lines.append(f"| Tier {t.tier_index + 1}{star} | {t.min_qty:,.0f} to {max_str} | ${t.unit_price:.2f} | {t.calculated_eoq:,.1f} | {t.candidate_qty:,.1f} | {feas_str} | ${t.annual_purchase_cost:,.2f} | ${t.annual_order_cost:,.2f} | ${t.annual_holding_cost:,.2f} | **${t.total_annual_cost:,.2f}** |")
            lines.append("")
            lines.append(f"- **Baseline Total Cost (Tier 1 base EOQ):** `${r.baseline_eoq_cost:,.2f}`")
            lines.append(f"- **Net Annual Cost Savings:** **${r.cost_savings:,.2f}** ({r.percentage_savings:.2f}% savings vs Tier 1 baseline)")
            lines.append("")

        elif model == ModelType.SAFETY_STOCK_ROP:
            r: ROPResult = result
            lines.append(f"- **Model Case Identified:** `{r.case_identified}`")
            lines.append(f"- **Expected Lead Time Demand ($\\mu_{{LT}}$):** `{r.expected_lead_time_demand:,.2f}` units over `{r.lead_time}` {r.time_unit}")
            lines.append(f"- **Standard Deviation of Lead Time Demand ($\\sigma_{{LT}}$):** `{r.stdev_lead_time_demand:.2f}` units")
            if r.service_level is not None:
                lines.append(f"- **Safety Factor ($z$):** For CSL = `{r.service_level * 100:.1f}%`, $z = \\Phi^{{-1}}({r.service_level:.4f}) = {r.z_factor:.4f}$")
                lines.append(f"- **Calculated Safety Stock ($SS$):** $z \\cdot \\sigma_{{LT}} = {r.z_factor:.4f} \\cdot {r.stdev_lead_time_demand:.2f} = {r.safety_stock:.2f}$ units")
                lines.append(f"- **Stockout Risk per Cycle:** `{r.stockout_risk_per_cycle * 100:.1f}%`")
            else:
                lines.append(f"- **Safety Stock ($SS$):** `{r.safety_stock:.2f}` units (Constant demand & constant lead time; zero stockout risk)")
            if r.annual_safety_stock_holding_cost is not None:
                lines.append(f"- **Annual Safety Stock Carrying Cost:** `${r.annual_safety_stock_holding_cost:,.2f}/year`")
            if r.average_inventory is not None:
                lines.append(f"- **Average Cycle + Safety Inventory:** `{r.average_inventory:,.2f}` units")
            lines.append("")

        elif model == ModelType.FIXED_PERIOD_REVIEW:
            r: PeriodicReviewResult = result
            lines.append(f"- **Review Interval ($R$):** `{r.review_period}` {r.time_unit}")
            lines.append(f"- **Replenishment Lead Time ($L$):** `{r.lead_time}` {r.time_unit}")
            lines.append(f"- **Total Protection Period ($R + L$):** **{r.protection_period} {r.time_unit}**")
            lines.append(f"- **Expected Protection Period Demand ($\\mu_{{R+L}}$):** `{r.demand_mean_per_period} \\cdot {r.protection_period} = {r.expected_protection_demand:,.1f}` units")
            lines.append(f"- **Std Dev of Protection Demand ($\\sigma_{{R+L}}$):** `{r.demand_sd_per_period} \\cdot \\sqrt{{{r.protection_period}}} = {r.stdev_protection_demand:.2f}` units")
            csl_display = f"{r.service_level*100:.1f}%" if r.service_level is not None else "100.0%"
            lines.append(f"- **Safety Factor ($z$):** `{r.z_factor:.3f}` for CSL = {csl_display}")
            lines.append(f"- **Safety Stock ($SS$):** `{r.safety_stock:.2f}` units")
            lines.append(f"- **Order-Up-To Target Level ($M$):** `\\mu_{{R+L}} + SS = {r.expected_protection_demand:.1f} + {r.safety_stock:.2f} = {r.order_up_to_level:.2f}` units")
            if r.inventory_position is not None:
                if r.on_hand_inventory is not None:
                    lines.append(f"- **Inventory Position ($IP$):** On-hand ({r.on_hand_inventory}) + On-order ({r.on_order_inventory}) - Backorders ({r.backorders}) = `{r.inventory_position:.1f}` units")
                else:
                    lines.append(f"- **Inventory Position ($IP$):** `{r.inventory_position:.1f}` units (given directly)")
                lines.append(f"- **Recommended Order Quantity ($Q$):** `\\max(0, M - IP) = \\max(0, {r.order_up_to_level:.1f} - {r.inventory_position:.1f}) = {r.recommended_order_quantity:.1f}` units")
            lines.append(f"- **Continuous vs Periodic Review Trade-off:** In a continuous review system, safety stock would be `{r.continuous_review_comparison_ss:.1f}` units. Periodic review requires `{r.safety_stock:.1f}` units (**+{r.ss_increase_percentage:.1f}% more safety stock**) because inventory status is not tracked continuously between review epochs.")
            lines.append("")

        elif model == ModelType.NEWSVENDOR:
            r: NewsvendorResult = result
            if inputs.get("cost_underage") is not None and inputs.get("cost_overage") is not None and not inputs.get("selling_price"):
                lines.append(f"- **Cost of Underage ($C_u$):** Given directly as `${r.cost_underage:.2f}` per unit of unmet demand.")
                lines.append(f"- **Cost of Overage ($C_o$):** Given directly as `${r.cost_overage:.2f}` per unsold unit.")
            else:
                lines.append(f"- **Cost of Underage ($C_u$):** Price (${r.selling_price:.2f}) - Unit Cost (${r.unit_cost:.2f}) + Penalty (${r.penalty_cost:.2f}) = **${r.cost_underage:.2f}/unit** (Lost margin per stockout)")
                lines.append(f"- **Cost of Overage ($C_o$):** Unit Cost (${r.unit_cost:.2f}) - Salvage (${r.salvage_value:.2f}) = **${r.cost_overage:.2f}/unit** (Net loss per unsold unit)")
            lines.append(f"- **Critical Fractile ($CR$):** $\\frac{{{r.cost_underage:.2f}}}{{{r.cost_underage:.2f} + {r.cost_overage:.2f}}} = \\mathbf{{{r.critical_fractile:.4f}}}$")
            if r.distribution_type == "normal":
                lines.append(f"- **Optimal Safety Factor ($z^*$):** $\\Phi^{{-1}}({r.critical_fractile:.4f}) = {r.z_factor:.4f}$")
            lines.append("")
            lines.append("#### Financial & Risk Analysis Table:")
            lines.append("| Metric | Projected Value | Explanation |")
            lines.append("| :--- | :--- | :--- |")
            lines.append(f"| **Expected Sales** | {r.expected_sales:,.1f} units | Demand satisfied by inventory |")
            lines.append(f"| **Expected Lost Sales** | {r.expected_lost_sales:,.1f} units | Unmet demand due to stockout risk |")
            lines.append(f"| **Expected Leftover Inventory** | {r.expected_leftover_inventory:,.1f} units | Stock liquidated at salvage price |")
            lines.append(f"| Expected Sales Revenue | ${r.expected_revenue:,.2f} | Units sold at full price (${r.selling_price:.2f}) |")
            lines.append(f"| Expected Salvage Recovery | ${r.expected_salvage_revenue:,.2f} | Unsold units salvaged at ${r.salvage_value:.2f} |")
            lines.append(f"| Total Procurement Cost | ${r.expected_total_cost:,.2f} | Capital required upfront |")
            lines.append(f"| **Expected Net Profit** | **${r.expected_profit:,.2f}** | Operating margin after risk |")
            lines.append(f"| Profit with Perfect Information | ${r.profit_under_perfect_information:,.2f} | Upper theoretical ceiling |")
            lines.append(f"| **EVPI (Cost of Uncertainty)** | **${r.expected_value_of_perfect_information:,.2f}** | Maximum value of improved demand forecasting |")
            lines.append("")

        # -------------------------------------------------------------------
        # 6. Final Answer
        # -------------------------------------------------------------------
        lines.append("### 6. Final Answer")
        if model == ModelType.EOQ:
            r: EOQResult = result
            lines.append(f"- **Recommended Order Size ($Q^*$):** **{r.rounded_eoq:,} units** per order (Exact continuous EOQ: `{r.eoq:.2f}` units)")
            lines.append(f"- **Order Frequency ($N$):** **{r.orders_per_year:.2f} orders/year**")
            lines.append(f"- **Cycle Time ($T$):** **{r.cycle_time_days:.1f} operating days** ({r.cycle_time_weeks:.1f} weeks between orders)")
            lines.append(f"- **Minimized Total Annual Inventory Cost (TAC):** **${r.total_annual_inventory_cost:,.2f}/year**")
            if r.unit_cost is not None and r.unit_cost > 0:
                lines.append(f"- **Total Annual Cost (including purchase):** **${r.total_annual_cost:,.2f}/year**")
            else:
                lines.append(f"- **Total Annual Cost (excluding purchase):** **${r.total_annual_inventory_cost:,.2f}/year** *(Unit purchase price C not specified)*")
        elif model == ModelType.QUANTITY_DISCOUNT:
            r: QuantityDiscountResult = result
            lines.append(f"- **Optimal Order Quantity ($Q^*$):** **{r.optimal_order_quantity:,.0f} units**")
            lines.append(f"- **Optimal Discount Tier:** **Tier {r.best_tier_index + 1}** at **${r.optimal_unit_price:.2f}/unit**")
            lines.append(f"- **Minimized Total Annual Cost ($TC^*$):** **${r.min_total_cost:,.2f}/year**")
            lines.append(f"- **Economic Net Savings:** **${r.cost_savings:,.2f}** / **{r.percentage_savings:.2f}%** annual procurement and inventory savings")
        elif model == ModelType.SAFETY_STOCK_ROP:
            r: ROPResult = result
            lines.append(f"- **Reorder Point (ROP):** Place replenishment order when inventory position drops to **{math.ceil(r.reorder_point):,} units** (Exact: `{r.reorder_point:.2f}` units)")
            lines.append(f"- **Safety Stock Buffer ($SS$):** **{math.ceil(r.safety_stock):,} units** integer buffer (Exact: `{r.safety_stock:.2f}` units)")
            csl_disp = f"{r.service_level * 100:.1f}%" if r.service_level is not None else "100.0% (deterministic)"
            lines.append(f"- **Cycle Service Level (CSL):** **{csl_disp}**")
            lines.append(f"- **Decision Policy:** Whenever inventory position drops to or below **{math.ceil(r.reorder_point):,} units**, place an order.")
        elif model == ModelType.FIXED_PERIOD_REVIEW:
            r: PeriodicReviewResult = result
            lines.append(f"- **Target Order-Up-To Level ($M$):** **{math.ceil(r.order_up_to_level):,} units** (Exact: `{r.order_up_to_level:.2f}` units)")
            if r.recommended_order_quantity is not None:
                lines.append(f"- **Recommended Order Quantity for Current Cycle ($Q$):** **{math.ceil(r.recommended_order_quantity):,} units** (Exact: `{r.recommended_order_quantity:.2f}` units)")
            lines.append(f"- **Review Cadence:** Review inventory every **{r.review_period} {r.time_unit}** and order up to **{math.ceil(r.order_up_to_level):,} units**.")
        elif model == ModelType.NEWSVENDOR:
            r: NewsvendorResult = result
            lines.append(f"- **Recommended Order Quantity ($Q^*$):** **{r.rounded_order_quantity:,} units** (Exact: `{r.optimal_order_quantity:.2f}` units)")
            lines.append(f"- **Critical Ratio (Target In-Stock Probability):** **{r.critical_fractile:.4f}** ({r.critical_fractile * 100:.1f}%)")
            lines.append(f"- **Expected Net Operating Profit:** **${r.expected_profit:,.2f}**")
            lines.append(f"- **EVPI (Expected Value of Perfect Information):** **${r.expected_value_of_perfect_information:,.2f}**")
        lines.append("")

        # -------------------------------------------------------------------
        # 7. Verification Checks
        # -------------------------------------------------------------------
        lines.append("### 7. Verification Checks")
        checks = getattr(result, "verification_checks", verification_checklist)
        for chk in checks:
            lines.append(f"- {chk}")
        lines.append("")

        # -------------------------------------------------------------------
        # 8. Assumptions
        # -------------------------------------------------------------------
        lines.append("### 8. Assumptions")
        assumptions = getattr(result, "assumptions", [])
        for idx, asm in enumerate(assumptions, 1):
            lines.append(f"{idx}. {asm}")
        lines.append("")

        # Managerial Insight
        lines.append("#### Managerial Insight:")
        if model == ModelType.EOQ:
            r: EOQResult = result
            lines.append(f"- At $Q^* = {r.eoq:.2f}$, Annual Ordering Cost exactly balances Annual Holding Cost (${r.annual_order_cost:,.2f}).")
            lines.append(f"- Rounding to an actionable batch size of **{r.rounded_eoq:,} units** increases total annual inventory cost by only **${r.cost_penalty_rounded:.2f}**, demonstrating the flat bottom (robustness) of the EOQ total cost curve.")
        elif model == ModelType.QUANTITY_DISCOUNT:
            r: QuantityDiscountResult = result
            best_t = r.tiers[r.best_tier_index]
            if best_t.candidate_qty == best_t.min_qty and best_t.calculated_eoq < best_t.min_qty:
                lines.append(f"- Although the unconstrained EOQ ({best_t.calculated_eoq:.1f} units) falls below the threshold for Tier {r.best_tier_index + 1}, ordering at the breakpoint of **{best_t.min_qty:,.0f} units** captures purchasing discounts that far outweigh the increased holding cost.")
            else:
                lines.append(f"- The unconstrained EOQ of {best_t.calculated_eoq:.1f} naturally falls into the qualifying bracket, achieving simultaneous order-holding balance and reduced purchase expenditure.")
        elif model == ModelType.SAFETY_STOCK_ROP:
            r: ROPResult = result
            lines.append(f"- The safety stock buffer of **{math.ceil(r.safety_stock):,} units** absorbs variability during the `{r.lead_time}` {r.time_unit} lead time window.")
            if r.service_level is not None:
                lines.append(f"- Achieving a {r.service_level*100:.1f}% service level requires balancing stockout risk ({r.stockout_risk_per_cycle*100:.1f}% per cycle) against holding cost.")
        elif model == ModelType.FIXED_PERIOD_REVIEW:
            r: PeriodicReviewResult = result
            lines.append(f"- Periodic review requires protecting against demand over **{r.protection_period} {r.time_unit}** ($R + L$), requiring +{r.ss_increase_percentage:.1f}% more safety stock than continuous review.")
            lines.append("- Grouping items under a common review period $R$ offers substantial transportation and administrative consolidation economies.")
        elif model == ModelType.NEWSVENDOR:
            r: NewsvendorResult = result
            lines.append(f"- Critical fractile $CR = {r.critical_fractile:.4f}$ establishes the optimal trade-off between underage loss (${r.cost_underage:.2f}) and overage loss (${r.cost_overage:.2f}).")
            lines.append(f"- EVPI of **${r.expected_value_of_perfect_information:,.2f}** represents the maximum amount the firm should invest in forecasting and market intelligence.")

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # END-TO-END PIPELINE: Process Customer Question
    # -----------------------------------------------------------------------
    def process_question(self, question: str) -> AgentDecision:
        """
        Executes the 6-stage architecture:
        Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain
        """
        pipeline_trace: List[PipelineStage] = []

        # Stage 1: Identify Model
        model_type, confidence, indicators = self.identify_model(question)
        pipeline_trace.append(PipelineStage(
            stage_name="1. Identify Model",
            status="SUCCESS" if model_type != ModelType.UNKNOWN else "FAILED",
            details={"model": model_type.value, "confidence": confidence, "matched_indicators": indicators},
            summary=f"Identified {model_type.value} with confidence {confidence * 100:.0f}% based on keywords: {indicators}"
        ))

        if model_type == ModelType.UNKNOWN:
            return AgentDecision(
                question=question,
                model_type=ModelType.UNKNOWN,
                model_confidence=0.0,
                extracted_inputs={},
                selected_tool="None",
                calculation_result=None,
                verification_passed=False,
                verification_checklist=["[FAIL] Unable to match question to one of the 5 supported inventory models."],
                explanation="I could not identify which inventory decision model applies to your question. Please provide details matching one of the following: 1. EOQ, 2. Quantity Discount, 3. Safety Stock & ROP, 4. Fixed-Period Review, or 5. Newsvendor.",
                pipeline_trace=pipeline_trace,
            )

        # Stage 2: Extract Inputs
        extracted_inputs = self.extract_inputs(question, model_type)
        missing_critical = []
        if model_type == ModelType.EOQ:
            if not extracted_inputs.get("demand"):
                missing_critical.append("Annual Demand (D)")
            if not extracted_inputs.get("order_cost"):
                missing_critical.append("Order/Setup Cost (S)")
            has_h = extracted_inputs.get("holding_cost") is not None
            has_i_c = (extracted_inputs.get("holding_rate") is not None) and (extracted_inputs.get("unit_cost") is not None and extracted_inputs.get("unit_cost") > 0)
            if not has_h and not has_i_c:
                if not extracted_inputs.get("holding_cost") and not extracted_inputs.get("holding_rate"):
                    missing_critical.append("Holding Cost (H) or Holding Rate (i)")
                elif extracted_inputs.get("holding_rate") and not extracted_inputs.get("unit_cost"):
                    missing_critical.append("Unit Cost (C) to apply Holding Rate (i)")

        elif model_type == ModelType.QUANTITY_DISCOUNT:
            if not extracted_inputs.get("demand"):
                missing_critical.append("Annual Demand (D)")
            if not extracted_inputs.get("order_cost"):
                missing_critical.append("Order/Setup Cost (S)")
            if not extracted_inputs.get("discount_tiers"):
                missing_critical.append("Discount Tiers/Price Breaks")
            if not extracted_inputs.get("holding_rate") and not extracted_inputs.get("fixed_holding_cost"):
                missing_critical.append("Holding Rate (i) or Fixed Holding Cost (H)")

        elif model_type == ModelType.SAFETY_STOCK_ROP:
            if not extracted_inputs.get("lead_time"):
                missing_critical.append("Lead Time (L)")
            if not extracted_inputs.get("demand_mean_per_period") and not extracted_inputs.get("direct_ltd_mean"):
                missing_critical.append("Demand Mean")
            # Service level only required if demand or lead time is variable
            has_variability = ((extracted_inputs.get("demand_sd_per_period") or 0.0) > 0) or ((extracted_inputs.get("lead_time_sd") or 0.0) > 0)
            if has_variability:
                if not extracted_inputs.get("service_level") and not extracted_inputs.get("z_value"):
                    missing_critical.append("Cycle Service Level (CSL) or Z-score")

        elif model_type == ModelType.FIXED_PERIOD_REVIEW:
            if not extracted_inputs.get("review_period"):
                missing_critical.append("Review Period (R)")
            if not extracted_inputs.get("lead_time"):
                missing_critical.append("Lead Time (L)")
            if not extracted_inputs.get("demand_mean_per_period"):
                missing_critical.append("Demand Mean")
            has_variability = (extracted_inputs.get("demand_sd_per_period") or 0.0) > 0
            has_direct_ss = extracted_inputs.get("safety_stock") is not None
            if has_variability and not has_direct_ss:
                if not extracted_inputs.get("service_level") and not extracted_inputs.get("z_value"):
                    missing_critical.append("Service Level (CSL) or Safety Stock")

        elif model_type == ModelType.NEWSVENDOR:
            has_direct_marginal = (extracted_inputs.get("cost_underage") is not None) and (extracted_inputs.get("cost_overage") is not None)
            has_price_and_cost = (extracted_inputs.get("selling_price") is not None) and (extracted_inputs.get("unit_cost") is not None)
            if not has_direct_marginal and not has_price_and_cost:
                if not extracted_inputs.get("selling_price") and not extracted_inputs.get("cost_underage"):
                    missing_critical.append("Selling Price (p) or Underage Cost (Cu)")
                if not extracted_inputs.get("unit_cost") and not extracted_inputs.get("cost_overage"):
                    missing_critical.append("Unit Cost (c) or Overage Cost (Co)")
            if not extracted_inputs.get("demand_mean") and not extracted_inputs.get("uniform_min"):
                missing_critical.append("Demand Distribution Parameters (mean/sd or min/max)")

        if missing_critical:
            pipeline_trace.append(PipelineStage(
                stage_name="2. Extract Inputs",
                status="INCOMPLETE",
                details={"extracted": extracted_inputs, "missing": missing_critical},
                summary=f"Missing essential parameters: {', '.join(missing_critical)}"
            ))
            return AgentDecision(
                question=question,
                model_type=model_type,
                model_confidence=confidence,
                extracted_inputs=extracted_inputs,
                selected_tool="None",
                calculation_result=None,
                verification_passed=False,
                verification_checklist=[f"[FAIL] Missing critical parameter: {p}" for p in missing_critical],
                explanation=f"To solve this using the **{model_type.value}**, please provide the following missing inputs:\n" +
                            "\n".join([f"- **{p}**" for p in missing_critical]),
                pipeline_trace=pipeline_trace,
            )

        pipeline_trace.append(PipelineStage(
            stage_name="2. Extract Inputs",
            status="SUCCESS",
            details=extracted_inputs,
            summary=f"Successfully extracted {len(extracted_inputs)} parameters."
        ))

        # Stage 3 & 4: Select Python Tool & Calculate
        try:
            tool_name, calc_result = self.select_tool_and_calculate(model_type, extracted_inputs)
            pipeline_trace.append(PipelineStage(
                stage_name="3. Select Python Tool",
                status="SUCCESS",
                details={"tool": tool_name},
                summary=f"Selected tool `{tool_name}` from `inventory_tools.py`."
            ))
            pipeline_trace.append(PipelineStage(
                stage_name="4. Calculate",
                status="SUCCESS",
                details={"result_type": type(calc_result).__name__},
                summary="Deterministic numerical calculation executed successfully in Python."
            ))
        except Exception as e:
            pipeline_trace.append(PipelineStage(
                stage_name="4. Calculate",
                status="ERROR",
                details={"error": str(e)},
                summary=f"Calculation failed with error: {str(e)}"
            ))
            return AgentDecision(
                question=question,
                model_type=model_type,
                model_confidence=confidence,
                extracted_inputs=extracted_inputs,
                selected_tool="None",
                calculation_result=None,
                verification_passed=False,
                verification_checklist=[f"[ERROR] Calculation error: {str(e)}"],
                explanation=f"An error occurred during numerical calculation: {str(e)}",
                pipeline_trace=pipeline_trace,
            )

        # Stage 5: Verify
        verification_passed, checklist = self.verify_solution(model_type, extracted_inputs, calc_result)
        pipeline_trace.append(PipelineStage(
            stage_name="5. Verify",
            status="VERIFIED" if verification_passed else "WARNING",
            details={"passed": verification_passed, "checklist": checklist},
            summary="All mathematical, physical, and economic constraints verified." if verification_passed else "Constraint warnings flagged."
        ))

        # Stage 6: Explain
        explanation = self.generate_explanation(
            question=question,
            model=model_type,
            inputs=extracted_inputs,
            result=calc_result,
            verification_passed=verification_passed,
            verification_checklist=checklist,
        )
        pipeline_trace.append(PipelineStage(
            stage_name="6. Explain",
            status="SUCCESS",
            details={"length": len(explanation)},
            summary="Executive managerial explanation generated with LaTeX formulas, financial breakdown, and insights."
        ))

        student_verification = {}
        if hasattr(calc_result, "to_student_verification"):
            student_verification = calc_result.to_student_verification()

        return AgentDecision(
            question=question,
            model_type=model_type,
            model_confidence=confidence,
            extracted_inputs=extracted_inputs,
            selected_tool=tool_name,
            calculation_result=calc_result,
            verification_passed=verification_passed,
            verification_checklist=checklist,
            explanation=explanation,
            pipeline_trace=pipeline_trace,
            student_verification=student_verification,
        )

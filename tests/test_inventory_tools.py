"""
Unit and Integration Tests for Supply Chain Analytics Inventory Tools
=====================================================================
Comprehensive tests verifying all 5 inventory tools:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop()
4. calculate_periodic_review()
5. calculate_newsvendor()

Verifies:
- Strict input validation (never inventing missing numerical inputs; raising ValueError)
- Structured results with .intermediate_calculations, .final_answer, .assumptions, .to_dict()
- All 4 variance cases in calculate_rop()
- Fixed and percentage holding costs in calculate_quantity_discount()
- Protection period, target inventory, and order quantity in calculate_periodic_review()
- Underage, overage, critical ratio, and recommended order quantity in calculate_newsvendor()
- End-to-end question processing through InventoryDecisionAgent
"""

import math
import unittest

from inventory_tools import (
    calculate_eoq,
    calculate_quantity_discount,
    calculate_rop,
    calculate_safety_stock_rop,
    calculate_periodic_review,
    calculate_fixed_period_review,
    calculate_newsvendor,
    standard_normal_cdf,
    standard_normal_inv_cdf,
    unit_normal_loss,
    EOQResult,
    QuantityDiscountResult,
    ROPResult,
    PeriodicReviewResult,
    NewsvendorResult,
)


class TestNormalDistributionHelpers(unittest.TestCase):
    """Verify statistical distribution primitives."""

    def test_standard_normal_cdf(self):
        self.assertAlmostEqual(standard_normal_cdf(0.0), 0.50, places=4)
        self.assertAlmostEqual(standard_normal_cdf(1.64485), 0.95, places=4)
        self.assertAlmostEqual(standard_normal_cdf(1.95996), 0.975, places=4)
        self.assertAlmostEqual(standard_normal_cdf(2.32635), 0.99, places=4)

    def test_standard_normal_inv_cdf(self):
        self.assertAlmostEqual(standard_normal_inv_cdf(0.50), 0.0, places=4)
        self.assertAlmostEqual(standard_normal_inv_cdf(0.95), 1.64485, places=4)
        self.assertAlmostEqual(standard_normal_inv_cdf(0.975), 1.95996, places=4)
        self.assertAlmostEqual(standard_normal_inv_cdf(0.99), 2.32635, places=4)

    def test_standard_normal_inv_cdf_bounds(self):
        with self.assertRaises(ValueError):
            standard_normal_inv_cdf(0.0)
        with self.assertRaises(ValueError):
            standard_normal_inv_cdf(1.0)
        with self.assertRaises(ValueError):
            standard_normal_inv_cdf(-0.1)

    def test_unit_normal_loss(self):
        # L(0) = 1 / sqrt(2 * pi) approx 0.398942
        self.assertAlmostEqual(unit_normal_loss(0.0), 1.0 / math.sqrt(2.0 * math.pi), places=4)


# ===========================================================================
# 1. calculate_eoq() Tests
# ===========================================================================
class TestCalculateEOQ(unittest.TestCase):
    """Tests for calculate_eoq()."""

    def test_eoq_textbook_benchmark(self):
        # D = 10,000, S = 50, H = 2.0, C = 10.0
        # Q = sqrt(2 * 10000 * 50 / 2) = sqrt(500,000) approx 707.1068
        res = calculate_eoq(demand=10000, order_cost=50, holding_cost=2.0, unit_cost=10.0)

        self.assertIsInstance(res, EOQResult)
        self.assertAlmostEqual(res.eoq, 707.1068, places=3)
        self.assertEqual(res.rounded_eoq, 707)

        # Intermediate calculations
        self.assertIn("annual_ordering_cost", res.intermediate_calculations)
        self.assertIn("annual_holding_cost", res.intermediate_calculations)
        self.assertAlmostEqual(res.intermediate_calculations["annual_ordering_cost"], 707.1068, places=3)
        self.assertAlmostEqual(res.intermediate_calculations["annual_holding_cost"], 707.1068, places=3)
        self.assertAlmostEqual(res.intermediate_calculations["total_annual_inventory_cost"], 1414.2136, places=3)

        # Final answer
        self.assertIn("optimal_order_quantity_continuous", res.final_answer)
        self.assertAlmostEqual(res.final_answer["optimal_order_quantity_continuous"], 707.1068, places=3)

        # Assumptions identified
        self.assertTrue(len(res.assumptions) >= 4)
        self.assertTrue(any("constant" in a.lower() for a in res.assumptions))

        # to_dict() serialization
        d = res.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("intermediate_calculations", d)

    def test_eoq_with_holding_rate(self):
        # D = 12000, S = 100, i = 0.25, C = 40 -> H = 10 -> Q = sqrt(2 * 12000 * 100 / 10) = 489.8979
        res = calculate_eoq(demand=12000, order_cost=100, holding_rate=0.25, unit_cost=40.0)
        self.assertAlmostEqual(res.holding_cost, 10.0, places=4)
        self.assertAlmostEqual(res.eoq, 489.8979, places=3)

    def test_eoq_never_invents_missing_numerical_inputs(self):
        # Missing demand
        with self.assertRaises(ValueError):
            calculate_eoq(order_cost=50, holding_cost=2.0)
        # Missing order_cost
        with self.assertRaises(ValueError):
            calculate_eoq(demand=10000, holding_cost=2.0)
        # Missing holding cost and holding rate
        with self.assertRaises(ValueError):
            calculate_eoq(demand=10000, order_cost=50)
        # Negative demand
        with self.assertRaises(ValueError):
            calculate_eoq(demand=-1000, order_cost=50, holding_cost=2.0)
        # Zero order cost
        with self.assertRaises(ValueError):
            calculate_eoq(demand=1000, order_cost=0, holding_cost=2.0)


# ===========================================================================
# 2. calculate_quantity_discount() Tests
# ===========================================================================
class TestCalculateQuantityDiscount(unittest.TestCase):
    """Tests for calculate_quantity_discount()."""

    def setUp(self):
        self.tiers = [
            {"min_qty": 0, "max_qty": 999, "unit_price": 5.00},
            {"min_qty": 1000, "max_qty": 1999, "unit_price": 4.80},
            {"min_qty": 2000, "max_qty": None, "unit_price": 4.75},
        ]

    def test_quantity_discount_percentage_holding_cost(self):
        # D = 5000, S = 49, i = 0.20
        # Tier 0 (0-999, $5.00): H=1.00 -> EOQ=700 (feasible) -> TC=25,700
        # Tier 1 (1000-1999, $4.80): H=0.96 -> EOQ=714.4 -> candidate=1000 -> TC=24,725
        # Tier 2 (2000+, $4.75): H=0.95 -> EOQ=718.2 -> candidate=2000 -> TC=24,822.50
        # Optimal: Tier 1, Q=1000, TC=24,725
        res = calculate_quantity_discount(
            demand=5000,
            order_cost=49,
            discount_tiers=self.tiers,
            holding_rate=0.20,
        )

        self.assertIsInstance(res, QuantityDiscountResult)
        self.assertEqual(res.best_tier_index, 1)
        self.assertAlmostEqual(res.optimal_order_quantity, 1000.0, places=1)
        self.assertAlmostEqual(res.optimal_unit_price, 4.80, places=2)
        self.assertAlmostEqual(res.min_total_cost, 24725.0, places=2)
        self.assertAlmostEqual(res.cost_savings, 975.0, places=2)

        # Intermediate calculations
        self.assertIn("tiers_evaluated", res.intermediate_calculations)
        self.assertEqual(len(res.intermediate_calculations["tiers_evaluated"]), 3)
        self.assertAlmostEqual(res.intermediate_calculations["baseline_cost_tier_0"], 25700.0, places=2)

        # Final answer
        self.assertEqual(res.final_answer["best_tier_index"], 1)
        self.assertEqual(res.final_answer["optimal_order_quantity"], 1000.0)

        # Assumptions
        self.assertTrue(len(res.assumptions) >= 3)

    def test_quantity_discount_fixed_holding_cost(self):
        # D = 5000, S = 49, fixed H = $1.50 across all tiers
        res = calculate_quantity_discount(
            demand=5000,
            order_cost=49,
            discount_tiers=self.tiers,
            fixed_holding_cost=1.50,
        )
        self.assertEqual(res.fixed_holding_cost, 1.50)
        self.assertIsNone(res.holding_rate)
        # All tiers should have holding_cost = 1.50
        for t in res.tiers:
            self.assertEqual(t.holding_cost, 1.50)
        self.assertTrue(res.verification_passed)

    def test_quantity_discount_never_invents_missing_numerical_inputs(self):
        # Missing demand
        with self.assertRaises(ValueError):
            calculate_quantity_discount(order_cost=49, discount_tiers=self.tiers, holding_rate=0.20)
        # Missing order_cost
        with self.assertRaises(ValueError):
            calculate_quantity_discount(demand=5000, discount_tiers=self.tiers, holding_rate=0.20)
        # Missing tiers
        with self.assertRaises(ValueError):
            calculate_quantity_discount(demand=5000, order_cost=49, discount_tiers=[], holding_rate=0.20)
        # Missing both holding_rate and fixed_holding_cost
        with self.assertRaises(ValueError):
            calculate_quantity_discount(demand=5000, order_cost=49, discount_tiers=self.tiers)
        # Invalid tier price
        invalid_tiers = [{"min_qty": 0, "max_qty": 100, "unit_price": -5.0}]
        with self.assertRaises(ValueError):
            calculate_quantity_discount(demand=5000, order_cost=49, discount_tiers=invalid_tiers, holding_rate=0.20)


# ===========================================================================
# 3. calculate_rop() Tests (All 4 Cases)
# ===========================================================================
class TestCalculateROP(unittest.TestCase):
    """Tests for calculate_rop() across all 4 variance structures."""

    def test_case_1_constant_demand_and_constant_lead_time(self):
        """Case 1: sigma_d = 0, sigma_L = 0 -> SS = 0, ROP = d * L."""
        # d = 50, L = 9, sigma_d = 0, sigma_L = 0
        res = calculate_rop(
            demand_mean=50,
            lead_time_mean=9,
            demand_sd=0.0,
            lead_time_sd=0.0,
        )

        self.assertEqual(res.case_identified, "constant demand and constant lead time")
        self.assertEqual(res.safety_stock, 0.0)
        self.assertEqual(res.reorder_point, 450.0)
        self.assertEqual(res.stdev_lead_time_demand, 0.0)
        self.assertEqual(res.intermediate_calculations["expected_lead_time_demand"], 450.0)
        self.assertEqual(res.final_answer["reorder_point"], 450.0)
        self.assertEqual(res.final_answer["safety_stock"], 0.0)
        self.assertTrue(len(res.assumptions) >= 3)

    def test_case_2_variable_demand_and_constant_lead_time(self):
        """Case 2: sigma_d > 0, sigma_L = 0 -> sigma_LT = sigma_d * sqrt(L), SS = z * sigma_LT."""
        # d = 50, sigma_d = 10, L = 9, sigma_L = 0, CSL = 95% (z = 1.64485)
        # mu_LT = 450, sigma_LT = 10 * sqrt(9) = 30
        # SS = 1.64485 * 30 = 49.3456
        # ROP = 450 + 49.3456 = 499.3456
        res = calculate_rop(
            demand_mean=50,
            lead_time_mean=9,
            demand_sd=10.0,
            lead_time_sd=0.0,
            service_level=0.95,
        )

        self.assertEqual(res.case_identified, "variable demand and constant lead time")
        self.assertAlmostEqual(res.expected_lead_time_demand, 450.0, places=2)
        self.assertAlmostEqual(res.stdev_lead_time_demand, 30.0, places=2)
        self.assertAlmostEqual(res.z_factor, 1.64485, places=3)
        self.assertAlmostEqual(res.safety_stock, 49.3456, places=2)
        self.assertAlmostEqual(res.reorder_point, 499.3456, places=2)

    def test_case_3_constant_demand_and_variable_lead_time(self):
        """Case 3: sigma_d = 0, sigma_L > 0 -> sigma_LT = d * sigma_L, SS = z * sigma_LT."""
        # d = 50, sigma_d = 0, L = 9, sigma_L = 2, CSL = 95% (z = 1.64485)
        # mu_LT = 450, sigma_LT = 50 * 2 = 100
        # SS = 1.64485 * 100 = 164.485
        # ROP = 450 + 164.485 = 614.485
        res = calculate_rop(
            demand_mean=50,
            lead_time_mean=9,
            demand_sd=0.0,
            lead_time_sd=2.0,
            service_level=0.95,
        )

        self.assertEqual(res.case_identified, "constant demand and variable lead time")
        self.assertAlmostEqual(res.expected_lead_time_demand, 450.0, places=2)
        self.assertAlmostEqual(res.stdev_lead_time_demand, 100.0, places=2)
        self.assertAlmostEqual(res.safety_stock, 164.485, places=2)
        self.assertAlmostEqual(res.reorder_point, 614.485, places=2)

    def test_case_4_variable_demand_and_variable_lead_time(self):
        """
        Case 4: sigma_d > 0, sigma_L > 0:
        sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
        SS = z * sigma_LT
        ROP = d * L + SS
        """
        # d = 100, sigma_d = 10, L = 4, sigma_L = 1, CSL = 99% (z = 2.32635)
        # mu_LT = 100 * 4 = 400
        # sigma_LT = sqrt(4 * 10^2 + 100^2 * 1^2) = sqrt(400 + 10000) = sqrt(10400) approx 101.98039
        # SS = 2.32635 * 101.98039 approx 237.242
        # ROP = 400 + 237.242 = 637.242
        res = calculate_rop(
            demand_mean=100,
            lead_time_mean=4,
            demand_sd=10.0,
            lead_time_sd=1.0,
            service_level=0.99,
        )

        expected_sigma = math.sqrt(4 * (10.0**2) + (100.0**2) * (1.0**2))
        self.assertEqual(res.case_identified, "variable demand and variable lead time")
        self.assertAlmostEqual(res.expected_lead_time_demand, 400.0, places=2)
        self.assertAlmostEqual(res.stdev_lead_time_demand, expected_sigma, places=3)
        self.assertAlmostEqual(res.safety_stock, 2.32635 * expected_sigma, places=1)
        self.assertAlmostEqual(res.reorder_point, 400.0 + 2.32635 * expected_sigma, places=1)

    def test_rop_never_invents_missing_numerical_inputs(self):
        # Missing demand_mean
        with self.assertRaises(ValueError):
            calculate_rop(lead_time_mean=9, service_level=0.95)
        # Missing lead_time_mean
        with self.assertRaises(ValueError):
            calculate_rop(demand_mean=50, service_level=0.95)
        # Variable demand without service level or z value
        with self.assertRaises(ValueError):
            calculate_rop(demand_mean=50, lead_time_mean=9, demand_sd=10.0)
        # Negative lead time
        with self.assertRaises(ValueError):
            calculate_rop(demand_mean=50, lead_time_mean=-2)


# ===========================================================================
# 4. calculate_periodic_review() Tests
# ===========================================================================
class TestCalculatePeriodicReview(unittest.TestCase):
    """Tests for calculate_periodic_review()."""

    def test_periodic_review_order_up_to_level(self):
        """
        R = 14 days, L = 7 days -> Protection period = 14 + 7 = 21 days
        d = 20, sigma_d = 5, CSL = 98% (z = 2.05375)
        Demand during protection period = 20 * 21 = 420
        sigma_protection = 5 * sqrt(21) approx 22.91288
        SS = 2.05375 * 22.91288 approx 47.057
        Target inventory M = 420 + 47.057 = 467.057
        On-hand = 150, On-order = 50, Backorders = 10 -> IP = 190
        Order quantity Q = max(0, 467.057 - 190) = 277.057
        """
        res = calculate_periodic_review(
            review_period=14,
            lead_time=7,
            demand_mean=20,
            demand_sd=5,
            service_level=0.98,
            on_hand=150,
            on_order=50,
            backorders=10,
        )

        self.assertIsInstance(res, PeriodicReviewResult)
        self.assertEqual(res.protection_period, 21.0)
        self.assertEqual(res.intermediate_calculations["protection_period"], 21.0)
        self.assertAlmostEqual(res.intermediate_calculations["demand_during_protection_period"], 420.0, places=2)
        self.assertAlmostEqual(res.intermediate_calculations["stdev_protection_period"], 5 * math.sqrt(21), places=3)
        self.assertAlmostEqual(res.safety_stock, 47.057, places=1)
        self.assertAlmostEqual(res.target_inventory, 467.057, places=1)
        self.assertEqual(res.intermediate_calculations["inventory_position"], 190.0)
        self.assertAlmostEqual(res.order_quantity, 277.057, places=1)

        # Final answer
        self.assertAlmostEqual(res.final_answer["target_inventory"], 467.057, places=1)
        self.assertAlmostEqual(res.final_answer["order_quantity"], 277.057, places=1)

        # Assumptions
        self.assertTrue(len(res.assumptions) >= 3)
        self.assertTrue(any("protection" in a.lower() for a in res.assumptions))

    def test_periodic_review_order_quantity_zero_when_inventory_position_exceeds_target(self):
        """When on-hand inventory position exceeds target inventory, order quantity is 0."""
        res = calculate_periodic_review(
            review_period=7,
            lead_time=3,
            demand_mean=10,
            demand_sd=2,
            service_level=0.95,
            on_hand=200,  # Far exceeds target inventory of ~108
        )
        self.assertEqual(res.order_quantity, 0.0)

    def test_periodic_review_never_invents_missing_numerical_inputs(self):
        # Missing review_period
        with self.assertRaises(ValueError):
            calculate_periodic_review(lead_time=7, demand_mean=20, service_level=0.95)
        # Missing lead_time
        with self.assertRaises(ValueError):
            calculate_periodic_review(review_period=14, demand_mean=20, service_level=0.95)
        # Missing demand_mean
        with self.assertRaises(ValueError):
            calculate_periodic_review(review_period=14, lead_time=7, service_level=0.95)
        # Stochastic demand without service_level or z_value
        with self.assertRaises(ValueError):
            calculate_periodic_review(review_period=14, lead_time=7, demand_mean=20, demand_sd=5)


# ===========================================================================
# 5. calculate_newsvendor() Tests
# ===========================================================================
class TestCalculateNewsvendor(unittest.TestCase):
    """Tests for calculate_newsvendor()."""

    def test_newsvendor_normal_distribution(self):
        # Selling price p = 100, Cost c = 40, Salvage s = 20, Penalty g = 0
        # Cu = 100 - 40 = 60
        # Co = 40 - 20 = 20
        # Critical ratio CR = 60 / (60 + 20) = 0.75
        # Demand ~ Normal(mean = 1000, sd = 200)
        # z = inv_cdf(0.75) = 0.67449
        # Q* = 1000 + 0.67449 * 200 = 1134.898 units (rounded 1135)
        res = calculate_newsvendor(
            selling_price=100.0,
            unit_cost=40.0,
            salvage_value=20.0,
            demand_mean=1000.0,
            demand_sd=200.0,
        )

        self.assertIsInstance(res, NewsvendorResult)
        self.assertAlmostEqual(res.cost_underage, 60.0, places=2)
        self.assertAlmostEqual(res.cost_overage, 20.0, places=2)
        self.assertAlmostEqual(res.critical_ratio, 0.75, places=4)
        self.assertAlmostEqual(res.z_factor, 0.67449, places=3)
        self.assertAlmostEqual(res.recommended_order_quantity, 1134.898, places=1)
        self.assertEqual(res.rounded_order_quantity, 1135)

        # Intermediate calculations
        self.assertIn("underage_cost", res.intermediate_calculations)
        self.assertIn("overage_cost", res.intermediate_calculations)
        self.assertIn("critical_ratio", res.intermediate_calculations)
        self.assertAlmostEqual(res.intermediate_calculations["underage_cost"], 60.0)
        self.assertAlmostEqual(res.intermediate_calculations["overage_cost"], 20.0)
        self.assertAlmostEqual(res.intermediate_calculations["critical_ratio"], 0.75)
        self.assertGreater(res.expected_profit, 0)
        self.assertLess(res.expected_profit, res.profit_under_perfect_information)

        # Final answer
        self.assertAlmostEqual(res.final_answer["recommended_order_quantity"], 1134.898, places=1)
        self.assertEqual(res.final_answer["rounded_order_quantity"], 1135)

        # Assumptions
        self.assertTrue(len(res.assumptions) >= 3)
        self.assertTrue(any("single" in a.lower() for a in res.assumptions))

    def test_newsvendor_with_penalty_cost(self):
        # p = 100, c = 40, s = 20, g = 20 (goodwill loss)
        # Cu = 100 - 40 + 20 = 80
        # Co = 40 - 20 = 20
        # CR = 80 / (80 + 20) = 0.80
        res = calculate_newsvendor(
            selling_price=100.0,
            unit_cost=40.0,
            salvage_value=20.0,
            penalty_cost=20.0,
            demand_mean=1000.0,
            demand_sd=200.0,
        )
        self.assertAlmostEqual(res.cost_underage, 80.0, places=2)
        self.assertAlmostEqual(res.critical_ratio, 0.80, places=4)
        self.assertAlmostEqual(res.z_factor, standard_normal_inv_cdf(0.80), places=3)

    def test_newsvendor_uniform_distribution(self):
        # Uniform on [100, 300], p = 50, c = 30, s = 10
        # Cu = 20, Co = 20 -> CR = 0.50 -> Q* = 100 + 0.50 * 200 = 200
        res = calculate_newsvendor(
            selling_price=50.0,
            unit_cost=30.0,
            salvage_value=10.0,
            uniform_min=100.0,
            uniform_max=300.0,
        )
        self.assertAlmostEqual(res.critical_ratio, 0.50, places=4)
        self.assertAlmostEqual(res.recommended_order_quantity, 200.0, places=2)

    def test_newsvendor_never_invents_missing_numerical_inputs(self):
        # Missing selling_price
        with self.assertRaises(ValueError):
            calculate_newsvendor(unit_cost=40.0, demand_mean=1000.0, demand_sd=200.0)
        # Missing unit_cost
        with self.assertRaises(ValueError):
            calculate_newsvendor(selling_price=100.0, demand_mean=1000.0, demand_sd=200.0)
        # Missing demand distribution
        with self.assertRaises(ValueError):
            calculate_newsvendor(selling_price=100.0, unit_cost=40.0)
        # Selling price below unit cost
        with self.assertRaises(ValueError):
            calculate_newsvendor(selling_price=30.0, unit_cost=40.0, demand_mean=1000.0, demand_sd=200.0)
        # Unit cost below salvage value
        with self.assertRaises(ValueError):
            calculate_newsvendor(selling_price=100.0, unit_cost=20.0, salvage_value=30.0, demand_mean=1000.0, demand_sd=200.0)


# ===========================================================================
# 6. End-to-End Decision Agent Tests
# ===========================================================================
class TestInventoryAgentEndToEnd(unittest.TestCase):
    """Verifies the 6-stage architecture of InventoryDecisionAgent across all 5 models."""

    def setUp(self):
        from inventory_agent import InventoryDecisionAgent, ModelType
        self.agent = InventoryDecisionAgent()
        self.ModelType = ModelType

    def test_agent_eoq_question(self):
        q = "Our annual demand is 10,000 units, order cost is $50 per order, and holding cost is $2 per unit. What is our EOQ?"
        decision = self.agent.process_question(q)
        self.assertEqual(decision.model_type, self.ModelType.EOQ)
        self.assertEqual(decision.selected_tool, "calculate_eoq")
        self.assertTrue(decision.verification_passed)
        self.assertAlmostEqual(decision.calculation_result.eoq, 707.1068, places=3)
        self.assertIn("Q^*", decision.explanation)
        self.assertEqual(len(decision.pipeline_trace), 6)

    def test_agent_quantity_discount_question(self):
        q = ("Annual demand is 5000 units, order cost is $49, holding rate is 20%. "
             "Price is $5.00 for orders under 1000, $4.80 for 1000 to 1999 units, and $4.75 for 2000 or more units. What should we order?")
        decision = self.agent.process_question(q)
        self.assertEqual(decision.model_type, self.ModelType.QUANTITY_DISCOUNT)
        self.assertEqual(decision.selected_tool, "calculate_quantity_discount")
        self.assertTrue(decision.verification_passed)
        self.assertAlmostEqual(decision.calculation_result.optimal_order_quantity, 1000.0, places=1)
        self.assertIn("1,000", decision.explanation)

    def test_agent_safety_stock_rop_question(self):
        q = "Daily demand averages 50 units with standard deviation 10 units. Lead time is 9 days. What safety stock and reorder point do we need for a 95% cycle service level?"
        decision = self.agent.process_question(q)
        self.assertEqual(decision.model_type, self.ModelType.SAFETY_STOCK_ROP)
        self.assertEqual(decision.selected_tool, "calculate_rop")
        self.assertTrue(decision.verification_passed)
        self.assertAlmostEqual(decision.calculation_result.safety_stock, 49.3456, places=2)
        self.assertAlmostEqual(decision.calculation_result.reorder_point, 499.3456, places=2)

    def test_agent_fixed_period_review_question(self):
        q = ("We review inventory every 14 days with a lead time of 7 days. Daily demand is 20 units with std dev 5 units. "
             "Service level is 98%. We currently have 150 units on-hand, 50 units on-order, and 10 backorders. What is our order-up-to level?")
        decision = self.agent.process_question(q)
        self.assertEqual(decision.model_type, self.ModelType.FIXED_PERIOD_REVIEW)
        self.assertEqual(decision.selected_tool, "calculate_periodic_review")
        self.assertTrue(decision.verification_passed)
        self.assertAlmostEqual(decision.calculation_result.order_up_to_level, 467.057, places=1)
        self.assertAlmostEqual(decision.calculation_result.recommended_order_quantity, 277.057, places=1)

    def test_agent_newsvendor_question(self):
        q = ("We sell a seasonal jacket for $100, which costs us $40 to purchase. Any unsold jacket at the end of the season "
             "is salvaged for $20. Expected demand is normally distributed with mean 1000 and standard deviation 200. How many should we order?")
        decision = self.agent.process_question(q)
        self.assertEqual(decision.model_type, self.ModelType.NEWSVENDOR)
        self.assertEqual(decision.selected_tool, "calculate_newsvendor")
        self.assertTrue(decision.verification_passed)
        self.assertAlmostEqual(decision.calculation_result.optimal_order_quantity, 1134.898, places=1)
        self.assertEqual(decision.calculation_result.rounded_order_quantity, 1135)


# ===========================================================================
# 7. Natural-Language Input Extraction & Synonym Disambiguation Tests
# ===========================================================================
class TestInventoryAgentNLPExtraction(unittest.TestCase):
    """
    Verifies robust natural language input extraction, recognition of domain synonyms,
    and parameter disambiguation across all inventory models.
    """

    def setUp(self):
        from inventory_agent import InventoryDecisionAgent, ModelType
        self.agent = InventoryDecisionAgent()
        self.ModelType = ModelType

    def test_eoq_ordering_cost_synonyms(self):
        """
        Verify that all 7 required ordering cost synonyms are accurately mapped:
        1. ordering cost
        2. order cost
        3. setup cost
        4. setup expense
        5. cost per order
        6. cost of placing an order
        7. ordering/setup cost
        """
        synonyms = [
            "ordering cost",
            "order cost",
            "setup cost",
            "setup expense",
            "cost per order",
            "cost of placing an order",
            "ordering/setup cost",
        ]
        for syn in synonyms:
            with self.subTest(synonym=syn):
                q = f"Our annual demand is 10,000 units, {syn} is $50, and holding cost is $2 per unit. What is the EOQ?"
                decision = self.agent.process_question(q)
                self.assertEqual(decision.model_type, self.ModelType.EOQ, f"Failed to identify EOQ for {syn}")
                self.assertIsNotNone(decision.calculation_result, f"Calculation failed for {syn}: {decision.explanation}")
                self.assertEqual(decision.extracted_inputs.get("order_cost"), 50.0, f"Incorrect order_cost for {syn}")
                self.assertAlmostEqual(decision.calculation_result.eoq, 707.1068, places=3)
                self.assertTrue(decision.verification_passed)

    def test_safety_stock_rop_parameter_distinctions(self):
        """
        Verify distinction between:
        - average demand vs demand standard deviation
        - average lead time vs lead-time standard deviation
        - service level
        Even when standard deviations appear before the averages in the question.
        """
        q = ("Lead-time standard deviation is 2 days. Demand standard deviation is 15 units. "
             "Average daily demand is 100 units. Average lead time is 6 days. Target cycle service level is 95%.")
        decision = self.agent.process_question(q)
        self.assertEqual(decision.model_type, self.ModelType.SAFETY_STOCK_ROP)
        self.assertIsNotNone(decision.calculation_result, decision.explanation)

        inputs = decision.extracted_inputs
        self.assertEqual(inputs["demand_mean_per_period"], 100.0)
        self.assertEqual(inputs["demand_sd_per_period"], 15.0)
        self.assertEqual(inputs["lead_time"], 6.0)
        self.assertEqual(inputs["lead_time_sd"], 2.0)
        self.assertEqual(inputs["service_level"], 0.95)

        # Expected: sigma_LT = sqrt(6 * 15^2 + 100^2 * 2^2) = sqrt(1350 + 40000) = sqrt(41350) ~ 203.347
        # z = 1.64485 -> SS = 1.64485 * 203.347 ~ 334.475 -> ROP = 600 + 334.475 = 934.475
        self.assertAlmostEqual(decision.calculation_result.expected_lead_time_demand, 600.0, places=1)
        self.assertAlmostEqual(decision.calculation_result.safety_stock, 334.47, places=1)
        self.assertAlmostEqual(decision.calculation_result.reorder_point, 934.47, places=1)
        self.assertEqual(decision.calculation_result.case_identified, "variable demand and variable lead time")

    def test_safety_stock_rop_constant_demand_lead_time_no_csl(self):
        """
        Verify that constant demand and constant lead time do not falsely require service level.
        """
        q = "Demand is constant at 80 units per day and replenishment lead time is 5 days. What is the reorder point?"
        decision = self.agent.process_question(q)
        self.assertEqual(decision.model_type, self.ModelType.SAFETY_STOCK_ROP)
        self.assertIsNotNone(decision.calculation_result, decision.explanation)
        self.assertEqual(decision.calculation_result.safety_stock, 0.0)
        self.assertEqual(decision.calculation_result.reorder_point, 400.0)
        self.assertEqual(decision.calculation_result.case_identified, "constant demand and constant lead time")

    def test_quantity_discount_parameter_distinctions(self):
        """
        Verify distinction between:
        - quantity break
        - unit price
        - ordering cost
        - fixed holding cost vs percentage holding cost
        - unit purchase price
        """
        # Case A: Fixed holding cost
        q_fixed = ("Annual demand is 2,000 units. Order cost is $40. Fixed holding cost is $3 per unit. "
                   "The price is $20 for 1 to 499 units, $18 for 500 to 999 units, and $16 for 1000 or more units.")
        dec_fixed = self.agent.process_question(q_fixed)
        self.assertEqual(dec_fixed.model_type, self.ModelType.QUANTITY_DISCOUNT)
        self.assertIsNotNone(dec_fixed.calculation_result, dec_fixed.explanation)
        self.assertEqual(dec_fixed.extracted_inputs["fixed_holding_cost"], 3.0)
        self.assertIsNone(dec_fixed.extracted_inputs["holding_rate"])
        self.assertEqual(dec_fixed.extracted_inputs["order_cost"], 40.0)
        self.assertEqual(dec_fixed.extracted_inputs["demand"], 2000.0)
        self.assertEqual(len(dec_fixed.extracted_inputs["discount_tiers"]), 3)

        # Case B: Percentage holding cost with setup cost synonym
        q_pct = ("Annual demand is 5,000 units, setup expense is $49, and percentage holding cost is 20%. "
                 "Price is $5.00 for orders under 1000, $4.80 for 1000 to 1999 units, and $4.75 for 2000 or more units.")
        dec_pct = self.agent.process_question(q_pct)
        self.assertEqual(dec_pct.model_type, self.ModelType.QUANTITY_DISCOUNT)
        self.assertIsNotNone(dec_pct.calculation_result, dec_pct.explanation)
        self.assertEqual(dec_pct.extracted_inputs["holding_rate"], 0.20)
        self.assertIsNone(dec_pct.extracted_inputs["fixed_holding_cost"])
        self.assertEqual(dec_pct.extracted_inputs["order_cost"], 49.0)

    def test_fixed_period_review_parameter_distinctions(self):
        """
        Verify distinction between:
        - demand
        - review period
        - lead time
        - safety stock
        - inventory position
        - on-hand inventory
        - on-order inventory
        - backorders
        """
        # Case A: Detailed inventory position components
        q_comp = ("Review period is 10 days, lead time is 4 days, daily demand averages 25 units with demand standard deviation of 6 units. "
                  "Cycle service level is 95%. On-hand inventory is 180 units, on-order inventory is 60 units, and backorders are 10 units.")
        dec_comp = self.agent.process_question(q_comp)
        self.assertEqual(dec_comp.model_type, self.ModelType.FIXED_PERIOD_REVIEW)
        self.assertIsNotNone(dec_comp.calculation_result, dec_comp.explanation)

        inputs = dec_comp.extracted_inputs
        self.assertEqual(inputs["review_period"], 10.0)
        self.assertEqual(inputs["lead_time"], 4.0)
        self.assertEqual(inputs["demand_mean_per_period"], 25.0)
        self.assertEqual(inputs["demand_sd_per_period"], 6.0)
        self.assertEqual(inputs["service_level"], 0.95)
        self.assertEqual(inputs["on_hand_inventory"], 180.0)
        self.assertEqual(inputs["on_order_inventory"], 60.0)
        self.assertEqual(inputs["backorders"], 10.0)
        # Inventory position: 180 + 60 - 10 = 230
        self.assertEqual(dec_comp.calculation_result.inventory_position, 230.0)

        # Case B: Direct inventory position
        q_direct_ip = ("We review inventory every 7 days with a replenishment lead time of 3 days. "
                       "Mean demand is 30 units per day with std dev 4 units. Service level is 90%. Current inventory position is 200 units.")
        dec_ip = self.agent.process_question(q_direct_ip)
        self.assertEqual(dec_ip.model_type, self.ModelType.FIXED_PERIOD_REVIEW)
        self.assertIsNotNone(dec_ip.calculation_result, dec_ip.explanation)
        self.assertEqual(dec_ip.extracted_inputs["inventory_position"], 200.0)
        self.assertEqual(dec_ip.calculation_result.inventory_position, 200.0)

        # Case C: Direct safety stock provided without needing service level
        q_direct_ss = ("Periodic review interval is 14 days, lead time is 7 days, daily demand is 50 units. "
                       "We maintain a safety stock of 100 units. On-hand inventory is 300 units.")
        dec_ss = self.agent.process_question(q_direct_ss)
        self.assertEqual(dec_ss.model_type, self.ModelType.FIXED_PERIOD_REVIEW)
        self.assertIsNotNone(dec_ss.calculation_result, dec_ss.explanation)
        self.assertEqual(dec_ss.extracted_inputs["safety_stock"], 100.0)
        self.assertEqual(dec_ss.calculation_result.safety_stock, 100.0)
        self.assertEqual(dec_ss.calculation_result.target_inventory, 1150.0)  # 50 * 21 + 100 = 1150
        self.assertEqual(dec_ss.calculation_result.recommended_order_quantity, 850.0)  # 1150 - 300 = 850

    def test_newsvendor_parameter_distinctions(self):
        """
        Verify distinction between:
        - selling price
        - purchase cost
        - salvage value
        - underage / shortage cost
        - overage / excess cost
        - demand distribution (mean demand, demand standard deviation)
        """
        # Case A: Direct underage cost and overage cost (no selling price or purchase cost needed)
        q_direct = ("Underage cost is $18 per unit and overage cost is $6 per unit. "
                    "Demand is normally distributed with mean demand of 500 units and demand standard deviation of 80 units. What is the optimal order quantity?")
        dec_direct = self.agent.process_question(q_direct)
        self.assertEqual(dec_direct.model_type, self.ModelType.NEWSVENDOR)
        self.assertIsNotNone(dec_direct.calculation_result, dec_direct.explanation)
        self.assertEqual(dec_direct.extracted_inputs["cost_underage"], 18.0)
        self.assertEqual(dec_direct.extracted_inputs["cost_overage"], 6.0)
        self.assertEqual(dec_direct.extracted_inputs["demand_mean"], 500.0)
        self.assertEqual(dec_direct.extracted_inputs["demand_sd"], 80.0)
        self.assertAlmostEqual(dec_direct.calculation_result.critical_ratio, 0.75, places=4)
        self.assertTrue(dec_direct.verification_passed)

        # Case B: Full retail price, unit purchase cost, and salvage value
        q_retail = ("The retail selling price is $120, unit purchase cost is $50, and salvage value is $15. "
                    "Demand averages 300 units with demand standard deviation of 50 units.")
        dec_retail = self.agent.process_question(q_retail)
        self.assertEqual(dec_retail.model_type, self.ModelType.NEWSVENDOR)
        self.assertIsNotNone(dec_retail.calculation_result, dec_retail.explanation)
        self.assertEqual(dec_retail.extracted_inputs["selling_price"], 120.0)
        self.assertEqual(dec_retail.extracted_inputs["unit_cost"], 50.0)
        self.assertEqual(dec_retail.extracted_inputs["salvage_value"], 15.0)
        # Cu = 120 - 50 = 70, Co = 50 - 15 = 35 -> CR = 70 / 105 = 2/3
        self.assertAlmostEqual(dec_retail.calculation_result.critical_ratio, 70.0 / 105.0, places=4)
        self.assertTrue(dec_retail.verification_passed)


# ===========================================================================
# 8. Student Verification Outputs Tests (All 5 Inventory Models)
# ===========================================================================
class TestStudentVerificationOutputs(unittest.TestCase):
    """
    Verify that every calculation tool and agent decision returns:
    1. Inputs Extracted
    2. Variable Definitions
    3. Formula
    4. Formula with Substituted Values
    5. Intermediate Calculations
    6. Final Answer
    7. Verification Checks
    8. Assumptions
    """

    EXPECTED_KEYS = [
        "1_inputs_extracted",
        "2_variable_definitions",
        "3_formula",
        "4_formula_with_substituted_values",
        "5_intermediate_calculations",
        "6_final_answer",
        "7_verification_checks",
        "8_assumptions",
    ]

    EXPECTED_SECTION_HEADERS = [
        "### 1. Inputs Extracted",
        "### 2. Variable Definitions",
        "### 3. Formula",
        "### 4. Formula with Substituted Values",
        "### 5. Intermediate Calculations",
        "### 6. Final Answer",
        "### 7. Verification Checks",
        "### 8. Assumptions",
    ]

    def setUp(self):
        from inventory_agent import InventoryDecisionAgent
        self.agent = InventoryDecisionAgent()

    def _verify_tool_verification_object(self, res, tool_name: str):
        # 1. Check attributes exist and are non-empty
        self.assertTrue(hasattr(res, "inputs_extracted"), f"{tool_name} missing inputs_extracted")
        self.assertTrue(bool(res.inputs_extracted), f"{tool_name} inputs_extracted is empty")

        self.assertTrue(hasattr(res, "variable_definitions"), f"{tool_name} missing variable_definitions")
        self.assertTrue(bool(res.variable_definitions), f"{tool_name} variable_definitions is empty")

        self.assertTrue(hasattr(res, "formula"), f"{tool_name} missing formula")
        self.assertTrue(bool(res.formula), f"{tool_name} formula is empty")

        self.assertTrue(hasattr(res, "formula_with_substituted_values"), f"{tool_name} missing formula_with_substituted_values")
        self.assertTrue(bool(res.formula_with_substituted_values), f"{tool_name} formula_with_substituted_values is empty")

        self.assertTrue(hasattr(res, "intermediate_calculations"), f"{tool_name} missing intermediate_calculations")
        self.assertTrue(bool(res.intermediate_calculations), f"{tool_name} intermediate_calculations is empty")

        self.assertTrue(hasattr(res, "final_answer"), f"{tool_name} missing final_answer")
        self.assertTrue(bool(res.final_answer), f"{tool_name} final_answer is empty")

        self.assertTrue(hasattr(res, "verification_checks"), f"{tool_name} missing verification_checks")
        self.assertTrue(bool(res.verification_checks), f"{tool_name} verification_checks is empty")

        self.assertTrue(hasattr(res, "assumptions"), f"{tool_name} missing assumptions")
        self.assertTrue(bool(res.assumptions), f"{tool_name} assumptions is empty")

        # 2. Check .to_student_verification() returns dict with exact 8 numbered keys
        self.assertTrue(hasattr(res, "to_student_verification"), f"{tool_name} missing to_student_verification")
        sv = res.to_student_verification()
        self.assertIsInstance(sv, dict)
        self.assertEqual(list(sv.keys()), self.EXPECTED_KEYS)
        for k in self.EXPECTED_KEYS:
            self.assertTrue(bool(sv[k]), f"{tool_name} student verification key {k} is empty")

    def test_eoq_student_verification_elements(self):
        res = calculate_eoq(demand=10000, order_cost=50, holding_cost=2.0, unit_cost=10.0)
        self._verify_tool_verification_object(res, "calculate_eoq")
        self.assertIn("optimal_order_quantity_Q*", res.formula)
        self.assertIn("optimal_order_quantity_Q*", res.formula_with_substituted_values)
        self.assertIn("D", res.variable_definitions)
        self.assertIn("S", res.variable_definitions)
        self.assertIn("H", res.variable_definitions)

    def test_quantity_discount_student_verification_elements(self):
        tiers = [
            {"min_qty": 0, "max_qty": 999, "unit_price": 5.0},
            {"min_qty": 1000, "max_qty": 1999, "unit_price": 4.8},
            {"min_qty": 2000, "unit_price": 4.75},
        ]
        res = calculate_quantity_discount(demand=5000, order_cost=49, discount_tiers=tiers, holding_rate=0.20)
        self._verify_tool_verification_object(res, "calculate_quantity_discount")
        self.assertIn("unconstrained_eoq_tier_j", res.formula)
        self.assertIn("Optimal_Decision", res.formula_with_substituted_values)
        self.assertIn("P_j", res.variable_definitions)
        self.assertIn("q_j", res.variable_definitions)

    def test_rop_student_verification_elements(self):
        res = calculate_rop(demand_mean=50, lead_time=9, demand_sd=10, service_level=0.95)
        self._verify_tool_verification_object(res, "calculate_rop")
        self.assertIn("Safety_Stock", res.formula)
        self.assertIn("Reorder_Point", res.formula)
        self.assertIn("Reorder_Point", res.formula_with_substituted_values)
        self.assertIn("sigma_LT", res.variable_definitions)
        self.assertIn("z", res.variable_definitions)

    def test_periodic_review_student_verification_elements(self):
        res = calculate_periodic_review(
            demand_mean=20,
            review_period=14,
            lead_time=7,
            demand_sd=5,
            service_level=0.98,
            on_hand_inventory=150,
            on_order_inventory=50,
            backorders=10,
        )
        self._verify_tool_verification_object(res, "calculate_periodic_review")
        self.assertIn("Protection_Period", res.formula)
        self.assertIn("Order_Up_To_Level_M", res.formula)
        self.assertIn("Order_Quantity_Q", res.formula_with_substituted_values)
        self.assertIn("R + L", res.variable_definitions)
        self.assertIn("IP", res.variable_definitions)

    def test_newsvendor_student_verification_elements(self):
        res = calculate_newsvendor(
            selling_price=100,
            unit_cost=40,
            salvage_value=20,
            demand_mean=1000,
            demand_sd=200,
        )
        self._verify_tool_verification_object(res, "calculate_newsvendor")
        self.assertIn("Critical_Ratio_CR", res.formula)
        self.assertIn("Optimal_Order_Quantity_Q*", res.formula_with_substituted_values)
        self.assertIn("Cu", res.variable_definitions)
        self.assertIn("Co", res.variable_definitions)
        self.assertIn("CR", res.variable_definitions)

    def test_agent_explanation_contains_all_8_sections_for_all_5_models(self):
        benchmark_questions = [
            # 1. EOQ
            "Annual demand is 10,000 units, ordering cost is $50 per order, and holding cost is $2 per unit per year.",
            # 2. Quantity Discount
            ("Annual demand is 5,000 units, order cost is $49, holding rate is 20%. "
             "Prices are $5.00 for orders under 1000, $4.80 for 1000 to 1999 units, and $4.75 for 2000 or more units."),
            # 3. Safety Stock & ROP
            "Daily demand averages 50 units with std dev 10 units. Lead time is 9 days. Service level is 95%.",
            # 4. Fixed-Period Review
            ("Review period is 14 days, lead time is 7 days, daily demand averages 20 units with std dev 5 units. "
             "Service level is 98%. On hand is 150, on order is 50, backorders 10."),
            # 5. Newsvendor
            "Selling price is $100, unit cost is $40, salvage value is $20. Mean demand is 1,000 with std dev 200.",
        ]

        for q in benchmark_questions:
            decision = self.agent.process_question(q)
            self.assertTrue(decision.verification_passed, f"Verification failed for question: {q}")
            self.assertIsNotNone(decision.calculation_result, f"Calculation result None for question: {q}")

            # Check decision.student_verification dict
            self.assertIsInstance(decision.student_verification, dict)
            self.assertEqual(list(decision.student_verification.keys()), self.EXPECTED_KEYS)

            # Check explanation contains all 8 section headers in order
            for header in self.EXPECTED_SECTION_HEADERS:
                self.assertIn(header, decision.explanation, f"Missing {header} in explanation for model {decision.model_type.value}")

            # Verify that each header appears in sequential order
            last_pos = -1
            for header in self.EXPECTED_SECTION_HEADERS:
                pos = decision.explanation.find(header)
                self.assertTrue(pos > last_pos, f"Header {header} out of order in explanation: {pos} <= {last_pos}")
                last_pos = pos


# ===========================================================================
# 9. CSV Data Analysis Tests (Streamlit Integration)
# ===========================================================================
class TestCSVDataAnalysis(unittest.TestCase):
    """Verify CSV validation and pandas calculations for Streamlit data analysis."""

    def test_missing_required_columns(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed in current test interpreter")

        from app import analyze_inventory_dataframe

        # Missing 'Category' and 'Unit Cost'
        df_invalid = pd.DataFrame({"Product": ["Widget A"], "Units": [100]})
        missing, results = analyze_inventory_dataframe(df_invalid)
        self.assertIn("Category", missing)
        self.assertIn("Unit Cost", missing)
        self.assertEqual(results, {})

    def test_valid_inventory_calculations(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed in current test interpreter")

        from app import analyze_inventory_dataframe

        data = {
            "Product": ["SKU-1", "SKU-2", "SKU-3", "SKU-4"],
            "Category": ["Hardware", "Hardware", "Tools", "Tools"],
            "Units": [200, 800, 300, 100],
            "Unit Cost": [15.0, 10.0, 50.0, 25.0],
        }
        df = pd.DataFrame(data)
        missing, results = analyze_inventory_dataframe(df)

        self.assertEqual(missing, [])
        # 1. Total Units: 200 + 800 + 300 + 100 = 1400
        self.assertEqual(results["total_units"], 1400.0)
        # 2. Average Unit Cost: (15 + 10 + 50 + 25) / 4 = 25.0
        self.assertEqual(results["avg_unit_cost"], 25.0)
        # 3. Product with highest units: SKU-2 (800)
        self.assertEqual(results["highest_units_product"], "SKU-2")
        self.assertEqual(results["highest_units_val"], 800.0)
        # 4. Total units by Category: Hardware=1000, Tools=400
        cat_dict = dict(zip(results["units_by_category"]["Category"], results["units_by_category"]["Units"]))
        self.assertEqual(cat_dict["Hardware"], 1000.0)
        self.assertEqual(cat_dict["Tools"], 400.0)
        # 5. Inventory Value = Units * Unit Cost: 200*15=3000, 800*10=8000, 300*50=15000, 100*25=2500 -> sum = 28500
        self.assertIn("Inventory Value", results["dataframe"].columns)
        self.assertEqual(results["total_inventory_value"], 28500.0)


if __name__ == "__main__":
    unittest.main()

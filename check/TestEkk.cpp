#include <algorithm>
#include <cmath>

#include "HCheckConfig.h"
#include "Highs.h"
#include "SpecialLps.h"
#include "catch.hpp"
#include "lp_data/HConst.h"

const bool dev_run = false;

void ekk_solve(Highs& highs, std::string presolve,
               const HighsModelStatus require_model_status,
               const double require_optimal_objective = 0) {
  SpecialLps special_lps;
  if (!dev_run) highs.setOptionValue("output_flag", false);
  const HighsInfo& info = highs.getInfo();

  REQUIRE(highs.setOptionValue("simplex_strategy", kSimplexStrategyDual) ==
          HighsStatus::kOk);

  REQUIRE(highs.setOptionValue("presolve", presolve) == HighsStatus::kOk);

  REQUIRE(highs.setBasis() == HighsStatus::kOk);

  REQUIRE(highs.run() == HighsStatus::kOk);

  REQUIRE(highs.getModelStatus() == require_model_status);

  if (require_model_status == HighsModelStatus::kOptimal) {
    REQUIRE(special_lps.objectiveOk(info.objective_function_value,
                                    require_optimal_objective, dev_run));
  }

  REQUIRE(highs.resetOptions() == HighsStatus::kOk);
  highs.setOptionValue("output_flag", dev_run);

  highs.resetThreadScheduler(true);
}

void ekk_distillation(Highs& highs) {
  SpecialLps special_lps;
  special_lps.reportLpName("distillation", dev_run);
  HighsLp lp;
  HighsModelStatus require_model_status;
  double optimal_objective;
  special_lps.distillationLp(lp, require_model_status, optimal_objective);
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
  ekk_solve(highs, "on", require_model_status, optimal_objective);
}

void ekk_blending(Highs& highs) {
  SpecialLps special_lps;
  special_lps.reportLpName("blending", dev_run);
  HighsLp lp;
  HighsModelStatus require_model_status;
  double optimal_objective;
  special_lps.blendingLp(lp, require_model_status, optimal_objective);
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
  ekk_solve(highs, "on", require_model_status, optimal_objective);
}

void ekk_scipLpi3(Highs& highs) {
  SpecialLps special_lps;
  special_lps.reportLpName("scipLpi3", dev_run);
  HighsLp lp;
  HighsModelStatus require_model_status;
  special_lps.scipLpi3Lp(lp, require_model_status);
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
  ekk_solve(highs, "off", require_model_status);
}

TEST_CASE("Ekk", "[highs_test_ekk]") {
  Highs highs;
  if (!dev_run) highs.setOptionValue("output_flag", false);
  HighsLp lp;
  const bool from_file = true;
  if (from_file) {
    std::string model_file =
        std::string(HIGHS_DIR) + "/check/instances/25fv47.mps";
    // "/check/instances/adlittle.mps";
    REQUIRE(highs.readModel(model_file) == HighsStatus::kOk);

    REQUIRE(highs.setOptionValue("simplex_strategy", kSimplexStrategyDual) ==
            HighsStatus::kOk);
    if (dev_run)
      highs.setOptionValue("log_dev_level", kHighsLogDevLevelDetailed);
    REQUIRE(highs.run() == HighsStatus::kOk);
  } else {
    //    ekk_distillation(highs);
    ekk_blending(highs);
    //    ekk_scipLpi3(highs);
  }

  highs.resetThreadScheduler(true);
}

TEST_CASE("EkkPrimal-all", "[highs_test_ekk]") {
  Highs highs;
  if (!dev_run) highs.setOptionValue("output_flag", false);
  ekk_distillation(highs);
  ekk_blending(highs);
}

TEST_CASE("Ekk-excessive-primal-value", "[highs_test_ekk]") {
  std::string log;
  Highs highs;
  REQUIRE(highs.setOptionValue("log_to_console", dev_run) == HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("log_dev_level", kHighsLogDevLevelInfo) ==
          HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("presolve", "off") == HighsStatus::kOk);
  REQUIRE(highs.setCallback(
              [&log](int callback_type, const std::string& message,
                     const HighsCallbackOutput*, HighsCallbackInput*, void*) {
                if (callback_type == kCallbackLogging) log += message;
              }) == HighsStatus::kOk);
  REQUIRE(highs.startCallback(kCallbackLogging) == HighsStatus::kOk);

  HighsLp lp;
  lp.num_col_ = 2;
  lp.num_row_ = 3;
  lp.col_cost_ = {0, 0};
  lp.col_lower_ = {-kHighsInf, -kHighsInf};
  lp.col_upper_ = {kHighsInf, kHighsInf};
  lp.row_lower_ = {-kHighsInf, -kHighsInf, -kHighsInf};
  lp.row_upper_ = {0, 0, -1e16};
  lp.a_matrix_.start_ = {0, 2, 5};
  lp.a_matrix_.index_ = {0, 1, 0, 1, 2};
  lp.a_matrix_.value_ = {1e-6, -1e6, -2, -1, 1};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
  REQUIRE(highs.run() == HighsStatus::kError);
  REQUIRE(log.find("Excessive primal value") != std::string::npos);

  highs.resetThreadScheduler(true);
}

// After a dual simplex solve from a non-logical basis that is far from
// optimal, the steepest edge weights of rows that were never primal
// infeasible are unknown, and those computed on request are exact
TEST_CASE("Ekk-unknown-dual-edge-weights", "[highs_test_ekk]") {
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  REQUIRE(highs.setOptionValue("presolve", "off") == HighsStatus::kOk);
  // Without scaling, the weights are the squared norms of the rows of
  // the basis inverse that getBasisInverseRow returns
  REQUIRE(highs.setOptionValue("simplex_scale_strategy", 0) ==
          HighsStatus::kOk);
  const std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/80bau3b.mps";
  REQUIRE(highs.readModel(filename) == HighsStatus::kOk);
  const HighsLp lp = highs.getLp();
  const HighsInt num_row = lp.num_row_;

  // Start from the optimal basis of the LP with perturbed row bounds
  HighsLp perturbed_lp = lp;
  for (HighsInt iRow = 0; iRow < num_row; iRow++) {
    const double factor = 1.0 + 0.3 * std::sin(1.7 * iRow + 0.3);
    perturbed_lp.row_lower_[iRow] *= factor;
    perturbed_lp.row_upper_[iRow] *= factor;
    if (perturbed_lp.row_lower_[iRow] > perturbed_lp.row_upper_[iRow])
      perturbed_lp.row_upper_[iRow] = perturbed_lp.row_lower_[iRow];
  }
  REQUIRE(highs.passModel(perturbed_lp) == HighsStatus::kOk);
  REQUIRE(highs.run() == HighsStatus::kOk);
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
  const HighsBasis basis = highs.getBasis();
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
  REQUIRE(highs.setBasis(basis) == HighsStatus::kOk);
  REQUIRE(highs.run() == HighsStatus::kOk);
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
  REQUIRE(highs.getInfo().simplex_iteration_count > 0);

  const double* known_weight = highs.getDualEdgeWeights();
  REQUIRE(known_weight != nullptr);
  HighsInt num_unknown_weight = 0;
  for (HighsInt iRow = 0; iRow < num_row; iRow++)
    if (known_weight[iRow] < 0) num_unknown_weight++;
  REQUIRE(num_unknown_weight > 0);

  std::vector<HighsInt> basic_index(num_row);
  for (HighsInt iRow = 0; iRow < num_row; iRow++) basic_index[iRow] = iRow;
  const double* weight = highs.getDualEdgeWeights(basic_index);
  REQUIRE(weight != nullptr);
  std::vector<double> row_vector(num_row);
  double max_relative_error = 0;
  for (HighsInt iRow = 0; iRow < num_row; iRow++) {
    REQUIRE(highs.getBasisInverseRow(iRow, row_vector.data()) ==
            HighsStatus::kOk);
    double exact_weight = 0;
    for (const double value : row_vector) exact_weight += value * value;
    max_relative_error =
        std::max(std::fabs(weight[iRow] - exact_weight) / exact_weight,
                 max_relative_error);
  }
  REQUIRE(max_relative_error < 1e-6);

  highs.resetThreadScheduler(true);
}

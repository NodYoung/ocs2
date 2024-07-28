#include <cmath>
#include <iostream>
#include <string>
#include <thread>
#include <glog/logging.h>

#include <ocs2_core/augmented_lagrangian/AugmentedLagrangian.h>
#include <ocs2_core/cost/QuadraticStateCost.h>
#include <ocs2_core/penalties/Penalties.h>
#include <ocs2_ddp/ILQR.h>
#include <ocs2_ddp/SLQ.h>
#include <ocs2_oc/synchronized_module/SolverObserver.h>

#include "ocs2_cartpole/CartPoleInterface.h"
#include "ocs2_cartpole/package_path.h"

enum class PenaltyType {
  SlacknessSquaredHingePenalty,
  ModifiedRelaxedBarrierPenalty,
};

int TestCartpoleDDP(ocs2::ddp::Algorithm algo, PenaltyType penalty) {
  std::string task_file = ocs2::cartpole::getPath() + "/config/mpc/task.info";
  const std::string lib_folder = ocs2::cartpole::getPath() + "/auto_generated";
  ocs2::cartpole::CartPoleInterface cart_pole_interface(task_file, lib_folder, true);
  // Since the problem only uses final cost for swing-up, the final cost should be scaled proportional to the increase of time horizon. We fist remove the final cost and then add a scaled version
  const ocs2::scalar_t time_horizon = 15.0;
  {
    const std::string final_cost_name = "finalCost";
    if (!cart_pole_interface.optimalControlProblem().finalCostPtr->erase(final_cost_name)) {
      throw std::runtime_error("[Cartpole::Cartpole]: " + final_cost_name + " was not found!");
    }
    ocs2::matrix_t Qf(ocs2::cartpole::STATE_DIM, ocs2::cartpole::STATE_DIM);
    ocs2::loadData::loadEigenMatrix(task_file, "Q_final", Qf);
    Qf *= (time_horizon / cart_pole_interface.mpcSettings().timeHorizon_);  // scale cost
    cart_pole_interface.optimalControlProblem().finalCostPtr->add(final_cost_name, std::make_unique<ocs2::QuadraticStateCost>(Qf));
  }
  // initial command
  ocs2::TargetTrajectories init_target_trajectories;
  init_target_trajectories.timeTrajectory.push_back(0.0);
  init_target_trajectories.stateTrajectory.push_back(cart_pole_interface.getInitialTarget());
  init_target_trajectories.inputTrajectory.push_back(ocs2::vector_t::Zero(ocs2::cartpole::INPUT_DIM));
  // construct solver
  std::unique_ptr<ocs2::GaussNewtonDDP> ddp_ptr;
  {
    cart_pole_interface.ddpSettings().algorithm_ = algo;
    cart_pole_interface.ddpSettings().maxNumIterations_ = 100;
    cart_pole_interface.ddpSettings().minRelCost_ = 1e-12;  // to avoid early termination
    cart_pole_interface.ddpSettings().displayInfo_ = false;
    cart_pole_interface.ddpSettings().displayShortSummary_ = true;
    const std::string input_limits_name = "InputLimits";
    // remove InputLimits as it will be added later
    std::unique_ptr<ocs2::StateInputConstraint> input_limits_constraint;
    {
      auto termLagrangianPtr = cart_pole_interface.optimalControlProblem().inequalityLagrangianPtr->extract(input_limits_name);
      if (termLagrangianPtr == nullptr) {
        throw std::runtime_error("[Cartpole::popInequalityLagrangian]: " + input_limits_name + " was not found!");
      }
      auto termStateInpuLagrangianPtr = dynamic_cast<ocs2::StateInputAugmentedLagrangian*>(termLagrangianPtr.get());
      if (termStateInpuLagrangianPtr == nullptr) {
        throw std::runtime_error("[Cartpole::popInequalityLagrangian]: term " + input_limits_name + " is not of type StateInputAugmentedLagrangian!");
      }
      input_limits_constraint = std::unique_ptr<ocs2::StateInputConstraint>(termStateInpuLagrangianPtr->get().clone());
    }
    // getPenalty
    std::unique_ptr<ocs2::augmented::AugmentedPenaltyBase> augmented_penalty;
    {
      switch (penalty) {
        case PenaltyType::SlacknessSquaredHingePenalty: {
          using penalty_type = ocs2::augmented::SlacknessSquaredHingePenalty;
          penalty_type::Config boundsConfig;
          ocs2::loadData::loadPenaltyConfig(task_file, "bounds_penalty_config", boundsConfig, false);
          augmented_penalty = penalty_type::create(boundsConfig);
          break;
        }
        case PenaltyType::ModifiedRelaxedBarrierPenalty: {
          using penalty_type = ocs2::augmented::ModifiedRelaxedBarrierPenalty;
          penalty_type::Config boundsConfig;
          ocs2::loadData::loadPenaltyConfig(task_file, "bounds_penalty_config", boundsConfig, false);
          augmented_penalty = penalty_type::create(boundsConfig);
          break;
        }
        default:
          throw std::runtime_error("[Cartpole::getPenalty] undefined penaltyType");
      }
    }
    // add InputLimits by penalty
    cart_pole_interface.getOptimalControlProblem().inequalityLagrangianPtr->add(input_limits_name, ocs2::create(std::unique_ptr<ocs2::StateInputConstraint>(input_limits_constraint->clone()), std::move(augmented_penalty)));
    // getAlgorithm
    switch (algo) {
      case ocs2::ddp::Algorithm::SLQ:
        ddp_ptr = std::make_unique<ocs2::SLQ>(std::move(cart_pole_interface.ddpSettings()), cart_pole_interface.getRollout(), cart_pole_interface.getOptimalControlProblem(), cart_pole_interface.getInitializer());
        break;
      case ocs2::ddp::Algorithm::ILQR:
        ddp_ptr = std::make_unique<ocs2::ILQR>(std::move(cart_pole_interface.ddpSettings()), cart_pole_interface.getRollout(), cart_pole_interface.getOptimalControlProblem(), cart_pole_interface.getInitializer());
        break;
      default:
        throw std::runtime_error("[Cartpole::getAlgorithm] undefined algorithm");
    }
  }
  // set TargetTrajectories
  ddp_ptr->getReferenceManager().setTargetTrajectories(init_target_trajectories);
  // observer for InputLimits violation
  const ocs2::scalar_t constraint_violation_tolerance = 1e-1;
  auto inputLimitsObserverModulePtr = ocs2::SolverObserver::LagrangianTermObserver(
      ocs2::SolverObserver::Type::Intermediate, 
      "InputLimits", 
      [&](const ocs2::scalar_array_t& timeTrajectory, const std::vector<ocs2::LagrangianMetricsConstRef>& termMetrics) {
        for (size_t i = 0; i < timeTrajectory.size(); i++) {
          const ocs2::vector_t constraintViolation = termMetrics[i].constraint.cwiseMin(0.0);
          if (std::fabs(constraintViolation(0)) > constraint_violation_tolerance) {
            LOG(ERROR) << "Input lower limit is violated at time " << std::to_string(timeTrajectory[i]);
          }
          if (std::fabs(constraintViolation(1)) > constraint_violation_tolerance) {
            LOG(ERROR) << "Input upper limit is violated at time " << std::to_string(timeTrajectory[i]);
          }
        }
      }
    );
  ddp_ptr->addSolverObserver(std::move(inputLimitsObserverModulePtr));

  // run solver
  ddp_ptr->run(0.0, cart_pole_interface.getInitialState(), time_horizon);

  // test final state
  const auto& finalState = ddp_ptr->primalSolution(time_horizon).stateTrajectory_.back();
  const auto& desiredState = cart_pole_interface.getInitialTarget();
  LOG(INFO) << "Pole final angle, final_state=" << finalState(0) << ", desired_state=" << desiredState(0);
  LOG(INFO) << "Cart final position, final_state=" << finalState(1) << ", desired_state=" << desiredState(1);
  LOG(INFO) << "Pole final velocity, final_state=" << finalState(2) << ", desired_state=" << desiredState(2);
  LOG(INFO) << "Cart final velocity, final_state=" << finalState(3) << ", desired_state=" << desiredState(3);
  return 0;
}

int main() {
  LOG(INFO) << "CartpoleDemo start";
  TestCartpoleDDP(ocs2::ddp::Algorithm::SLQ, PenaltyType::SlacknessSquaredHingePenalty);
  LOG(INFO) << "CartpoleDemo end";
  return 0;
}

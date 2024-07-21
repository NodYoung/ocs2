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

class Cartpole {
public:
  static constexpr ocs2::scalar_t timeHorizon = 15.0;
  static constexpr ocs2::scalar_t goalViolationTolerance = 1e-1;
  static constexpr ocs2::scalar_t constraintViolationTolerance = 1e-1;

  Cartpole(ocs2::ddp::Algorithm algo, PenaltyType penalty) {
    algorithm = algo;
    penaltyType = penalty;
    // interface
    taskFile = ocs2::cartpole::getPath() + "/config/mpc/task.info";
    const std::string libFolder = ocs2::cartpole::getPath() + "/auto_generated";
    cartPoleInterfacePtr.reset(new ocs2::cartpole::CartPoleInterface(taskFile, libFolder, true /*verbose*/));

    // Since the problem only uses final cost for swing-up, the final cost should be scaled proportional to
    // the increase of time horizon. We fist remove the final cost and then add a scaled version
    const std::string finalCostName = "finalCost";
    if (!cartPoleInterfacePtr->optimalControlProblem().finalCostPtr->erase(finalCostName)) {
      throw std::runtime_error("[Cartpole::Cartpole]: " + finalCostName + " was not found!");
    }
    auto createFinalCost = [&]() {
      ocs2::matrix_t Qf(ocs2::cartpole::STATE_DIM, ocs2::cartpole::STATE_DIM);
      ocs2::loadData::loadEigenMatrix(taskFile, "Q_final", Qf);
      Qf *= (timeHorizon / cartPoleInterfacePtr->mpcSettings().timeHorizon_);  // scale cost
      return std::make_unique<ocs2::QuadraticStateCost>(Qf);
    };
    cartPoleInterfacePtr->optimalControlProblem().finalCostPtr->add(finalCostName, createFinalCost());

    // remove InputLimits as it will be added later in each test
    inputLimitsConstraint = popInequalityLagrangian(inputLimitsName, cartPoleInterfacePtr->optimalControlProblem());

    // initial command
    initTargetTrajectories.timeTrajectory.push_back(0.0);
    initTargetTrajectories.stateTrajectory.push_back(cartPoleInterfacePtr->getInitialTarget());
    initTargetTrajectories.inputTrajectory.push_back(ocs2::vector_t::Zero(ocs2::cartpole::INPUT_DIM));
  }

  std::unique_ptr<ocs2::GaussNewtonDDP> getAlgorithm() const {
    auto ddpSettings = cartPoleInterfacePtr->ddpSettings();
    ddpSettings.algorithm_ = algorithm;
    ddpSettings.maxNumIterations_ = 100;
    ddpSettings.minRelCost_ = 1e-12;  // to avoid early termination
    ddpSettings.displayInfo_ = false;
    ddpSettings.displayShortSummary_ = true;

    switch (algorithm) {
      case ocs2::ddp::Algorithm::SLQ:
        return std::make_unique<ocs2::SLQ>(std::move(ddpSettings), cartPoleInterfacePtr->getRollout(),
                                     createOptimalControlProblem(),
                                     cartPoleInterfacePtr->getInitializer());
      case ocs2::ddp::Algorithm::ILQR:
        return std::make_unique<ocs2::ILQR>(std::move(ddpSettings), cartPoleInterfacePtr->getRollout(),
                                      createOptimalControlProblem(),
                                      cartPoleInterfacePtr->getInitializer());
      default:
        throw std::runtime_error("[Cartpole::getAlgorithm] undefined algorithm");
    }
  }

  std::unique_ptr<ocs2::augmented::AugmentedPenaltyBase> getPenalty() const {
    switch (penaltyType) {
      case PenaltyType::SlacknessSquaredHingePenalty: {
        using penalty_type = ocs2::augmented::SlacknessSquaredHingePenalty;
        penalty_type::Config boundsConfig;
        ocs2::loadData::loadPenaltyConfig(taskFile, "bounds_penalty_config", boundsConfig, false);
        return penalty_type::create(boundsConfig);
      }
      case PenaltyType::ModifiedRelaxedBarrierPenalty: {
        using penalty_type = ocs2::augmented::ModifiedRelaxedBarrierPenalty;
        penalty_type::Config boundsConfig;
        ocs2::loadData::loadPenaltyConfig(taskFile, "bounds_penalty_config", boundsConfig, false);
        return penalty_type::create(boundsConfig);
      }
      default:
        throw std::runtime_error("[Cartpole::getPenalty] undefined penaltyType");
    }
  }

  ocs2::OptimalControlProblem createOptimalControlProblem() const {
    ocs2::OptimalControlProblem problem = cartPoleInterfacePtr->getOptimalControlProblem();
    problem.inequalityLagrangianPtr->add(inputLimitsName, ocs2::create(std::unique_ptr<ocs2::StateInputConstraint>(inputLimitsConstraint->clone()), getPenalty()));
    return problem;
  }

  std::unique_ptr<ocs2::StateInputConstraint> popInequalityLagrangian(const std::string& name, ocs2::OptimalControlProblem& ocp) const {
    auto termLagrangianPtr = ocp.inequalityLagrangianPtr->extract(name);
    if (termLagrangianPtr == nullptr) {
      throw std::runtime_error("[Cartpole::popInequalityLagrangian]: " + name + " was not found!");
    }

    auto termStateInpuLagrangianPtr = dynamic_cast<ocs2::StateInputAugmentedLagrangian*>(termLagrangianPtr.get());
    if (termStateInpuLagrangianPtr == nullptr) {
      throw std::runtime_error("[Cartpole::popInequalityLagrangian]: term " + name + " is not of type StateInputAugmentedLagrangian!");
    }

    return std::unique_ptr<ocs2::StateInputConstraint>(termStateInpuLagrangianPtr->get().clone());
  }

  const std::string inputLimitsName = "InputLimits";
  ocs2::TargetTrajectories initTargetTrajectories;
  std::unique_ptr<ocs2::cartpole::CartPoleInterface> cartPoleInterfacePtr;

  std::string taskFile;
  std::unique_ptr<ocs2::StateInputConstraint> inputLimitsConstraint;

  ocs2::ddp::Algorithm algorithm; 
  PenaltyType penaltyType;
};

int CartpoleDDP(ocs2::ddp::Algorithm algo, PenaltyType penalty) {
  Cartpole cart_pole(algo, penalty);
  // construct solver
  auto ddpPtr = cart_pole.getAlgorithm();

  // set TargetTrajectories
  ddpPtr->getReferenceManager().setTargetTrajectories(cart_pole.initTargetTrajectories);

  // observer for InputLimits violation
  auto inputLimitsObserverModulePtr = ocs2::SolverObserver::LagrangianTermObserver(
      ocs2::SolverObserver::Type::Intermediate, 
      "InputLimits", 
      [&](const ocs2::scalar_array_t& timeTrajectory, const std::vector<ocs2::LagrangianMetricsConstRef>& termMetrics) {
        for (size_t i = 0; i < timeTrajectory.size(); i++) {
          const ocs2::vector_t constraintViolation = termMetrics[i].constraint.cwiseMin(0.0);
          if (std::fabs(constraintViolation(0)) > cart_pole.constraintViolationTolerance) {
            LOG(ERROR) << "Input lower limit is violated at time " << std::to_string(timeTrajectory[i]);
          }
          if (std::fabs(constraintViolation(1)) > cart_pole.constraintViolationTolerance) {
            LOG(ERROR) << "Input upper limit is violated at time " << std::to_string(timeTrajectory[i]);
          }
        }
      }
    );
  ddpPtr->addSolverObserver(std::move(inputLimitsObserverModulePtr));

  // run solver
  ddpPtr->run(0.0, cart_pole.cartPoleInterfacePtr->getInitialState(), cart_pole.timeHorizon);

  // test final state
  const auto& finalState = ddpPtr->primalSolution(cart_pole.timeHorizon).stateTrajectory_.back();
  const auto& desiredState = cart_pole.cartPoleInterfacePtr->getInitialTarget();
  LOG(INFO) << "Pole final angle, final_state=" << finalState(0) << ", desired_state=" << desiredState(0);
  LOG(INFO) << "Cart final position, final_state=" << finalState(1) << ", desired_state=" << desiredState(1);
  LOG(INFO) << "Pole final velocity, final_state=" << finalState(2) << ", desired_state=" << desiredState(2);
  LOG(INFO) << "Cart final velocity, final_state=" << finalState(3) << ", desired_state=" << desiredState(3);
}

std::unique_ptr<ocs2::StateInputConstraint> PopInequalityLagrangian(const std::string& name, ocs2::OptimalControlProblem& ocp) {
  auto termLagrangianPtr = ocp.inequalityLagrangianPtr->extract(name);
  if (termLagrangianPtr == nullptr) {
    throw std::runtime_error("[Cartpole::popInequalityLagrangian]: " + name + " was not found!");
  }
  auto termStateInpuLagrangianPtr = dynamic_cast<ocs2::StateInputAugmentedLagrangian*>(termLagrangianPtr.get());
  if (termStateInpuLagrangianPtr == nullptr) {
    throw std::runtime_error("[Cartpole::popInequalityLagrangian]: term " + name + " is not of type StateInputAugmentedLagrangian!");
  }
  return std::unique_ptr<ocs2::StateInputConstraint>(termStateInpuLagrangianPtr->get().clone());
}

std::unique_ptr<ocs2::augmented::AugmentedPenaltyBase> getPenalty(std::string task_file, PenaltyType penalty) {
  switch (penalty) {
    case PenaltyType::SlacknessSquaredHingePenalty: {
      using penalty_type = ocs2::augmented::SlacknessSquaredHingePenalty;
      penalty_type::Config boundsConfig;
      ocs2::loadData::loadPenaltyConfig(task_file, "bounds_penalty_config", boundsConfig, false);
      return penalty_type::create(boundsConfig);
    }
    case PenaltyType::ModifiedRelaxedBarrierPenalty: {
      using penalty_type = ocs2::augmented::ModifiedRelaxedBarrierPenalty;
      penalty_type::Config boundsConfig;
      ocs2::loadData::loadPenaltyConfig(task_file, "bounds_penalty_config", boundsConfig, false);
      return penalty_type::create(boundsConfig);
    }
    default:
      throw std::runtime_error("[Cartpole::getPenalty] undefined penaltyType");
  }
}

std::unique_ptr<ocs2::GaussNewtonDDP> getAlgorithm(std::string task_file, ocs2::ddp::Algorithm algorithm, PenaltyType penalty, ocs2::cartpole::CartPoleInterface& cart_pole_interface) {
  cart_pole_interface.ddpSettings().algorithm_ = algorithm;
  cart_pole_interface.ddpSettings().maxNumIterations_ = 100;
  cart_pole_interface.ddpSettings().minRelCost_ = 1e-12;  // to avoid early termination
  cart_pole_interface.ddpSettings().displayInfo_ = false;
  cart_pole_interface.ddpSettings().displayShortSummary_ = true;
  // remove InputLimits as it will be added later in each test
  const std::string input_limits_name = "InputLimits";
  std::unique_ptr<ocs2::StateInputConstraint> input_limits_constraint = PopInequalityLagrangian(input_limits_name, cart_pole_interface.optimalControlProblem());
  ocs2::OptimalControlProblem problem = cart_pole_interface.getOptimalControlProblem();
  problem.inequalityLagrangianPtr->add(input_limits_name, ocs2::create(std::unique_ptr<ocs2::StateInputConstraint>(input_limits_constraint->clone()), getPenalty(task_file, penalty)));
  switch (algorithm) {
    case ocs2::ddp::Algorithm::SLQ:
      return std::make_unique<ocs2::SLQ>(std::move(cart_pole_interface.ddpSettings()), cart_pole_interface.getRollout(),
                                   problem,
                                   cart_pole_interface.getInitializer());
    case ocs2::ddp::Algorithm::ILQR:
      return std::make_unique<ocs2::ILQR>(std::move(cart_pole_interface.ddpSettings()), cart_pole_interface.getRollout(),
                                    problem,
                                    cart_pole_interface.getInitializer());
    default:
      throw std::runtime_error("[Cartpole::getAlgorithm] undefined algorithm");
  }
}

int TestCartpoleDDP(ocs2::ddp::Algorithm algo, PenaltyType penalty) {
  std::string task_file = ocs2::cartpole::getPath() + "/config/mpc/task.info";
  const std::string lib_folder = ocs2::cartpole::getPath() + "/auto_generated";
  ocs2::cartpole::CartPoleInterface cart_pole_interface(task_file, lib_folder, true);
  // Since the problem only uses final cost for swing-up, the final cost should be scaled proportional to the increase of time horizon. We fist remove the final cost and then add a scaled version
  const std::string final_cost_name = "finalCost";
  if (!cart_pole_interface.optimalControlProblem().finalCostPtr->erase(final_cost_name)) {
    throw std::runtime_error("[Cartpole::Cartpole]: " + final_cost_name + " was not found!");
  }
  const ocs2::scalar_t time_horizon = 15.0;
  ocs2::matrix_t Qf(ocs2::cartpole::STATE_DIM, ocs2::cartpole::STATE_DIM);
  ocs2::loadData::loadEigenMatrix(task_file, "Q_final", Qf);
  Qf *= (time_horizon / cart_pole_interface.mpcSettings().timeHorizon_);  // scale cost
  cart_pole_interface.optimalControlProblem().finalCostPtr->add(final_cost_name, std::make_unique<ocs2::QuadraticStateCost>(Qf));
  // initial command
  ocs2::TargetTrajectories init_target_trajectories;
  init_target_trajectories.timeTrajectory.push_back(0.0);
  init_target_trajectories.stateTrajectory.push_back(cart_pole_interface.getInitialTarget());
  init_target_trajectories.inputTrajectory.push_back(ocs2::vector_t::Zero(ocs2::cartpole::INPUT_DIM));
  // construct solver
  std::unique_ptr<ocs2::GaussNewtonDDP> ddp_ptr = getAlgorithm(task_file, algo, penalty, cart_pole_interface);

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
  // CartpoleDDP(ocs2::ddp::Algorithm::SLQ, PenaltyType::SlacknessSquaredHingePenalty);
  TestCartpoleDDP(ocs2::ddp::Algorithm::SLQ, PenaltyType::SlacknessSquaredHingePenalty);
  LOG(INFO) << "CartpoleDemo end";
  return 0;
}
